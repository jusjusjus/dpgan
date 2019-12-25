#!/usr/bin/env python

"""
Source: https://github.com/tensorflow/models/blob/master/differential_privacy/privacy_accountant/tf/accountant.py
(Apache License, Version 2.0)

Reference: https://github.com/tensorflow/models/blob/master/differential_privacy/privacy_accountant/tf/accountant.py
"""

import sys
import math
from functools import partial
from collections import namedtuple

import numpy as np
import tensorflow as tf


EpsDelta = namedtuple('EpsDelta', 'spent_eps spent_delta')


def generate_binomial_table(m):
    """return float-64 tensor with Pascal's triangle

    Args:
        m: the size of the table.

    Returns:
        A two dimensional array T[i][j] = (i choose j) for 0<= i, j <=m."""
    table = np.zeros((m + 1, m + 1), dtype=np.float64)
    for i in range(m + 1):
        table[i, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            v = table[i - 1, j] + table[i - 1, j - 1]
            assert not math.isnan(v) and not math.isinf(v)
            table[i, j] = v
    return tf.convert_to_tensor(table)


class GaussianMomentsAccountant(object):

    def __init__(self, total_examples, moment_order=32):
        assert total_examples > 0
        self._total_examples = total_examples
        self._moment_orders = moment_order if np.iterable(moment_order) \
                              else np.arange(1, moment_order + 1)
        self._max_moment_order = max(self._moment_orders)
        assert self._max_moment_order < 100, f"""
        Moment order {self._max_moment_order} is too large"""
        self._log_moments = [
            tf.Variable(np.float64(0.0), trainable=False,
                name=f"log_moments-{order}")
            for order in self._moment_orders
        ]
        self._binomial = generate_binomial_table(self._max_moment_order)

    def _differential_moments(self, sigma, s, t):
        """return 0-to-t-th differential moments of Gaussian variable

        E[(P(x+s)/P(x+s-1)-1)^t] =
            sum_{i=0}^t (t choose i) (-1)^{t-i} E[(P(x+s)/P(x+s-1))^i]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} E[exp(-i*(2*x+2*s-1)/(2*sigma^2))]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} exp(i(i+1-2*s)/(2 sigma^2))

        Args:
          sigma: the noise sigma, in the multiples of the sensitivity.
          s: the shift.
          t: 0 to t-th moment.

        Returns:
          0 to t-th moment as a tensor of shape [t+1]."""

        t1 = t + 1
        assert t <= self._max_moment_order, f"""
        Order {t} is above upper bound {self._max_moment_order}"""
        binomial = tf.slice(self._binomial, begin=[0, 0], size=[t1, t1])
        signs = np.zeros((t1, t1), dtype=np.float64)
        for i in range(t1):
            for j in range(t1):
                signs[i, j] = 1. - 2 * ((i - j) % 2)

        exponents = tf.constant([
            j * (j + 1. - 2. * s) / (2. * sigma**2)
            for j in range(t1)
        ], dtype=tf.float64)

        # x[i, j] = binomial[i, j] * signs[i, j] = (i choose j) * (-1)^{i-j}

        x = tf.multiply(binomial, signs)

        # y[i, j] = x[i, j] * exp(exponents[j])
        #         = (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
        # Note: this computation is done by broadcasting pointwise multiplication
        # between [t+1, t+1] tensor and [t+1] tensor.

        y = tf.multiply(x, tf.exp(exponents))

        # z[i] = sum_j y[i, j]
        #      = sum_j (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))

        return tf.reduce_sum(y, axis=1)

    def _compute_log_moment(self, sigma, q, moment_order):
        """return high moment of privacy loss

        Args:
          sigma: the noise sigma, in the multiples of the sensitivity.
          q: the sampling ratio.
          moment_order: the order of moment.

        Returns:
          log E[exp(moment_order * X)]"""

        assert moment_order <= self._max_moment_order, f"""
        Order {moment_order} is above upper bound {self._max_moment_order}"""

        binomial = tf.slice(self._binomial, begin=[moment_order, 0],
                                  size=[1, moment_order+1])

        # qs = [1 q q^2 ... q^L] = exp([0 1 2 ... L] * log(q))

        qs = tf.exp(tf.constant([i * 1.0 for i in range(moment_order + 1)],
                                dtype=tf.float64) * tf.cast(
                                    tf.math.log(q), dtype=tf.float64))
        moments0 = self._differential_moments(sigma, 0.0, moment_order)
        term0 = tf.reduce_sum(binomial * qs * moments0)
        moments1 = self._differential_moments(sigma, 1.0, moment_order)
        term1 = tf.reduce_sum(binomial * qs * moments1)
        return tf.squeeze(tf.math.log(tf.cast(q * term0 + (1.0 - q) * term1,
                                         tf.float64)))

    def _compute_delta(self, log_moments, eps):
        """Compute delta for given log_moments and eps.

        Args:
          log_moments: the log moments of privacy loss, in the form of pairs
            of (moment_order, log_moment)
          eps: the target epsilon.

        Returns:
          delta"""

        min_delta = 1.0
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                sys.stderr.write(
                        "The %d-th order is inf or Nan\n" % moment_order)
                continue
            if log_moment < moment_order * eps:
                min_delta = min(min_delta,
                            math.exp(log_moment - moment_order * eps))
        return min_delta

    def _compute_eps(self, log_moments, delta):
        min_eps = float('inf')
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
                continue
            min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
        return min_eps

    def accumulate_privacy_spending(self, unused_eps_delta,
                                    sigma, num_examples):
        """return accumulated privacy spending

        In particular, accounts for privacy spending when we assume there are
        num_examples, and we are releasing the vector

        `Normal(0, stddev=l2norm_bound*sigma) + sum_{i=1}^{num_examples} x_i`

        where l2norm_bound is the maximum l2_norm of each example x_i, and the
        num_examples have been randomly selected out of a pool of
        `self.total_examples`.

        Args:
          unused_eps_delta: EpsDelta pair which can be tensors.  Unused
            in this accountant.
          sigma: the noise sigma, in the multiples of the sensitivity (that is,
            if the l2-norm sensitivity is k, then the caller must have added
            Gaussian noise with `stddev=k*sigma` to the result of the query).
          num_examples: the number of examples involved.

        Returns:
          a TensorFlow operation for updating the privacy spending."""

        q = tf.cast(num_examples, tf.float64) / float(self._total_examples)

        moments_accum_ops = [
            tf.compat.v1.assign_add(moment, self._compute_log_moment(
                sigma, q, order))
            for moment, order in zip(self._log_moments, self._moment_orders)
        ]
        return tf.group(*moments_accum_ops)

    def get_privacy_spent(self, sess, eps=None, deltas=None):
        """Compute privacy spending in (e, d)-DP form for a single or list of eps.

        Args:
          sess: the session to run the tensor.
          eps: a list of target epsilons for which we would like to
            compute corresponding delta value.
          deltas: a list of target deltas for which we would like to
            compute the corresponding eps value.

        Caller must specify either `eps` or `deltas`.

        Returns:
          A list of EpsDelta pairs."""

        assert (eps is None) ^ (deltas is None), """
        `get_privacy_spent` expects either eps or deltas"""
        log_moments = sess.run(self._log_moments)
        orders_and_moments = zip(self._moment_orders, log_moments)
        if eps is not None:
            eps = [eps] if not np.iterable(eps) else eps
            deltas = [self._compute_delta(orders_and_moments, e) for e in eps]
        else:
            deltas = [deltas] if not np.iterable(deltas) else deltas
            eps = [self._compute_eps(orders_and_moments, d) for d in deltas]

        return [EpsDelta(*ed) for ed in zip(eps, deltas)]


class DummyAccountant(object):
    """An accountant that does no accounting"""

    def accumulate_privacy_spending(self, *unused_args):
        return tf.no_op()

    def get_privacy_spent(self, unused_sess, **unused_kwargs):
        return [EpsDelta(np.inf, 1.0)]
