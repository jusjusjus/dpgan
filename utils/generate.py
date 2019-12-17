#!/usr/bin/env python
from six.moves import xrange

import os
from os import makedirs
from os.path import exists, join

import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image
from tqdm import trange


def generate_steps(config, generator_forward):
    with tf.device("/cpu:0"):
        fake_data = generator_forward(config)
    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, config.load_path)
    print("loaded model from %s." % config.load_path)

    tflearn.is_training(False, sess)
    to_stack = []
    for _ in xrange(config.times):
        generated = sess.run(fake_data)
        to_stack.append(generated)
    stacked = np.concatenate(to_stack, axis=0)

    if config.save_path:
        np.save(config.save_path, stacked, allow_pickle=False)


def generate_steps_png(config, generator_forward):

    with tf.device("/cpu:0"):
        fake_data = generator_forward(config)

    sess = tf.compat.v1.Session()

    if config.params:
        print(f"load model from '{config.params}'..")
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, config.params)
    else:
        sess.run(tf.global_variables_initializer())

    makedirs(config.save_dir, exist_ok=True)

    tflearn.is_training(False, sess)
    for batch_idx in trange(config.times):
        generated = sess.run(fake_data)
        for image_idx, arr in enumerate(generated):
            arr = (127.5 * (arr + 1)).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            img = Image.fromarray(arr, "RGB")
            img.save(join(config.save_dir,
                     f"{batch_idx * config.batch_size + image_idx}.png"))
