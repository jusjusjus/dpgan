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

    with tf.compat.v1.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, config.params)
        print(f"loaded model from {config.params}.")
        tflearn.is_training(False, sess)
        images = [sess.run(fake_data) for _ in range(config.times)]

    # concatenate and transform to uint8 format
    images = np.concatenate(images, axis=0)
    images = (127.5 * (images + 1)).astype(np.uint8)
    return images


def generate_steps_png(config, generator_forward):

    with tf.device("/cpu:0"):
        fake_data = generator_forward(config)

    with tf.compat.v1.Session() as sess:
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
                idx = batch_idx * config.batch_size + image_idx
                img.save(join(config.save_dir, f"{idx:04d}.png"))
