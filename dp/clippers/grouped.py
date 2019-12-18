
import tensorflow as tf
import numpy as np

from dp.clippers.base import Clipper


class GroupedClipper(Clipper):

    def __init__(self, groups, no_noise=False):
        super().__init__()
        self.no_noise = no_noise
        self.group_by_var = {}
        self.variables = {}
        self.bounds = {}

        for i, (variables, bound) in enumerate(groups):
            self.variables[i] = list(variables)
            self.bounds[i] = bound
            for var in variables:
                self.group_by_var[var] = i

    def clip_grads(self, m):
        groups = {i: [] for i in range(len(self.variables))}
        for v, g in m:
            assert v.name in self.group_by_var
            groups[self.group_by_var[v.name]].append((v.name, g))

        # It's questionable if this is really more efficient than to iterate
        # through all grads and apply clip-by-norm one by one.

        clipped_grads = {}
        for group, grouped_gradients in groups.items():
            names = [name for name, _ in grouped_gradients]
            grads = [grad for _, grad in grouped_gradients]
            shapes = [grad.shape for grad in grads]
            grads = [tf.reshape(grad, [-1]) for grad in grads]
            flattened_sizes = [grad.shape[0].value for grad in grads]
            grads = tf.concat(grads, axis=0)
            grads = tf.clip_by_norm(grads, self.bounds[group].get_bound_tensor())
            grads = tf.split(grads, flattened_sizes)
            for shape, name, grad in zip(shapes, names, grads):
                clipped_grads[name] = tf.reshape(grad, shape)

        return [clipped_grads[v.name] for v, _ in m]

    def num_accountant_terms(self, step):
        return len(self.variables)

    def noise_grads(self, m, batch_size, sigma):
        noised = {v: 0 for v in m}
        scaled_sigma = sigma / np.sqrt(batch_size)
        for v, g in m.items():
            assert v.name in self.group_by_var
            c_value = self.bounds[self.group_by_var[v.name]].get_bound_tensor()
            noise = tf.random.normal(shape=v.shape, stddev=c_value * scaled_sigma)
            noised[v] = g + noise if not self.no_noise else g
        return noised

    def info(self):
        f = "GroupedClipper\n"
        r = []
        for group, vars in sorted(self.variables.items(), key=lambda x: x[0]):
            r.append(f"({','.join(vars)}, {self.bounds[group]})")
        return f + '\n'.join(r)

    def update_feed_dict(self, sess, steps):
        d = {}
        for _, b in self.bounds.items():
            d.update(b.update_feed_dict(sess, steps))
        return d
