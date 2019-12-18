import tensorflow as tf
import numpy as np

from dp.clippers.base import Clipper


class BasicClipper(Clipper):

    def __init__(self, bound, specials=None):
        super(BasicClipper, self).__init__()
        self.bound = bound
        self.specials = set(specials) if specials is not None else {}
        self.keys = set()
        self._bounds = {}

    def clip_grads(self, m):
        clipped = []
        for w, g in m:
            self.keys.add(w)
            if w in self.specials:
                self._bounds[w] = self.specials[w]
                clipped.append(tf.clip_by_norm(g, self.specials[w].get_bound_tensor()))
            else:
                self._bounds[None] = self.bound
                clipped.append(tf.clip_by_norm(g, self._bounds[None].get_bound_tensor()))
        return clipped

    def num_accountant_terms(self, step):
        return len(self.keys)

    def noise_grads(self, m, batch_size, sigma):
        scaled_sigma = sigma / np.sqrt(batch_size)
        noised = {}
        for w, g in m.items():
            assert w in self.keys
            C = self.specials.get(w, self.bound).get_bound_tensor()
            noise = tf.random_normal(shape=w.shape, stddev=C * scaled_sigma)
            noised[w] = g + noise
        return noised

    def info(self):
        return "Basic clipper with bound: %r" % self.bound

    def update_feed_dict(self, sess, steps):
        d = {}
        for k, b in self._bounds.items():
            d.update(b.update_feed_dict(sess, steps))
        return d

