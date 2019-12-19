#!/usr/bin/env python

from sys import path
path.insert(0, '.')
import os
import argparse

import numpy as np
import tflearn
from tqdm import trange
import tensorflow as tf

from models.tasks.mnist import classifier_forward
from utils.data_utils import MNISTLoader


def run_task(config, train_data_loader, eval_data_loader, classifier_forward,
             optimizer):
    image_inputs = tf.placeholder(tf.float32, shape=[None] + train_data_loader.shape(), name="input")
    label_inputs = tf.placeholder(tf.int32, shape=[None, 10], name="labels")
    logits_op = classifier_forward(config, image_inputs, name="classifier")

    classifier_variables = [var for var in tf.all_variables() \
                            if var.name.startswith("classifier")]

    global_step = tf.Variable(0, False)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=label_inputs, logits=logits_op))

    labels_op = tf.cast(tf.argmax(tf.nn.softmax(logits_op), axis=-1), tf.int64)

    step_op = optimizer.minimize(loss_op, global_step=global_step,
        var_list=[var for var in classifier_variables if var in tf.trainable_variables()])

    acc_op = tf.reduce_mean(tf.cast(tf.equal(
        labels_op, tf.argmax(label_inputs, axis=-1)), tf.float32))

    print("graph built.")

    saver = tf.train.Saver(max_to_keep=15)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_step = 0
    for epoch in range(config.num_epoch):
        num_steps = int(config.num_example / config.batch_size)
        bar = trange(num_steps, leave=False)
        for _ in bar:
            images, labels = train_data_loader.next_batch(config.batch_size)

            eval_images, eval_labels = eval_data_loader.next_batch(config.batch_size)

            feed_dict = {image_inputs: images,
                         label_inputs: labels.astype(np.int32)}
            tflearn.is_training(True, sess)
            loss, _ = sess.run([acc_op, step_op], feed_dict=feed_dict)
            tflearn.is_training(False, sess)
            eval_loss = sess.run(acc_op, feed_dict={
                image_inputs: eval_images, label_inputs: eval_labels.astype(np.int32)})
            bar.set_description("train accuracy %.4f, eval accuracy %.4f" % (loss, eval_loss))
            total_step += 1

        if config.save_dir is not None:
            saver.save(sess, os.path.join(config.save_dir, "model"),
                       global_step=global_step)

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/mnist")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("-e", "--num-epoch", dest="num_epoch", type=int, default=15)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("-n", "--num-example", dest="num_example", type=int,
                        default=50000)

    config = parser.parse_args()

    print("config: %r" % config)
    os.makedirs(config.save_dir, exist_ok=True)

    train_data_loader = MNISTLoader(config.data_dir, include_test=False)
    eval_data_loader = MNISTLoader(config.data_dir, include_train=False)

    run_task(config, train_data_loader, eval_data_loader,
             classifier_forward, tf.train.AdamOptimizer())
