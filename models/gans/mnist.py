import tensorflow as tf
from tflearn.layers import activation, batch_normalization
from tflearn.layers.conv import conv_2d_transpose
from tflearn.activations import leaky_relu

from utils.ops import conv_2d, fully_connected


def random_labels(num_samples, num_labels):
    assert num_labels < 256
    return tf.random.uniform((num_samples,), maxval=num_labels,
                             name='labels')


def generator_forward(config, noise=None, labels=None, scope="generator",
                      name=None, reuse=False, num_samples=None):
    with tf.compat.v1.variable_scope(scope, name, reuse=reuse):
        if noise is None:
            num_samples = num_samples if num_samples else config.batch_size
            noise = tf.random.normal([num_samples, 128], name="noise")

        if labels is not None:
            labels = tf.one_hot(labels, 10)
            noise = tf.concat([noise, labels], -1)

        output = fully_connected(noise, 4*4*4*config.dim)
        output = batch_normalization(output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4*config.dim])

        output = conv_2d_transpose(output, 2 * config.dim, 5, [8, 8], strides=2)
        output = output[:, :7, :7, :]

        output = conv_2d_transpose(output, config.dim, 5, [14, 14], strides=2)
        output = tf.nn.relu(output)

        output = conv_2d_transpose(output, 1, 5, [28, 28], strides=2)

        output = tf.tanh(output)

    return output if labels is None else (output, labels)


def discriminator_forward(config, images, labels=None,
                      scope="discriminator", name=None, reuse=False):
    with tf.compat.v1.variable_scope(scope, name, reuse=reuse):
        if labels is not None:

            # Append one-hot embedding as additional channels to image.  The
            # embedded representation is expanded as constant vectors across
            # the whole image shape.

            labels = tf.one_hot(labels, 10)
            labels = labels[:, None, None, :]
            labels = tf.compat.v1.tile(labels, (1, *images.shape[1:3], 1))
            images = tf.concat([images, labels], -1)

        output = leaky_relu(conv_2d(images, config.dim, 5, 2), 0.2)
        output = leaky_relu(conv_2d(output, 2 * config.dim, 5, 2), 0.2)
        output = leaky_relu(conv_2d(output, 4 * config.dim, 5, 2), 0.2)

        output = tf.reshape(output, [-1, 4 * 4 * 4 * config.dim])
        output = tf.reshape(fully_connected(output, 1, bias=False), [-1])

    return output
