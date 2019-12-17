
import os
from os.path import join
from functools import partial

import tflearn
import tensorflow as tf
import numpy as np
from tqdm import trange

from .per_example_flow import train_graph_per_tower, aggregate_flow, gradient_norms_estimate_tower
from utils.data_utils import generate_images


def get_train_ops(config, real_data, fake_data, global_step, discriminator_forward,
                  gen_optimizer, disc_optimizer,
                  supervisor, accountant=None):
    with tf.device("/cpu:0"):
        discriminator_forward(config, tf.zeros(shape=[1] + [d.value for d in real_data.shape[1:]]), scope="discriminator")

    real_data_splits = tf.split(real_data, config.num_gpu, axis=0)
    fake_data_splits = tf.split(fake_data, config.num_gpu, axis=0)
    disc_grads = []
    gen_costs = []
    disc_costs = []

    for g, (real_data, fake_data) in enumerate(zip(real_data_splits, fake_data_splits)):
        with tf.device("/gpu:%d" % g):
            disc_cost, gen_cost, disc_grad = train_graph_per_tower(
                config, discriminator_forward, real_data, fake_data, supervisor)
            disc_costs.append(disc_cost)
            gen_costs.append(gen_cost)
            disc_grads.append(disc_grad)

    if supervisor.sampler is not None:
        supervisor.sampler.set_forward_function(partial(gradient_norms_estimate_tower,
                                                        config, discriminator_forward,
                                                        real_data_splits[0], fake_data_splits[0],
                                                        supervisor))

    return aggregate_flow(config, disc_costs, gen_costs, disc_grads,
                          disc_optimizer=disc_optimizer,
                          gen_optimizer=gen_optimizer,
                          global_step=global_step, supervisor=supervisor,
                          accountant=accountant)


def train_steps(config, dataloader, real_data, fake_data, global_step,
                gen_train_op, gen_cost, supervisor,
                accountant=None):
    init = tf.global_variables_initializer()

    # Load train savers for the whole graph and for the generator
    # specifically.

    saver = tf.compat.v1.train.Saver(max_to_keep=25)
    var_list = [
        var for var in tf.global_variables()
        if var.name.startswith(("generator", "discriminator"))
        and not var.name.endswith("is_training:0")
    ]
    gan_saver = tf.compat.v1.train.Saver(var_list=var_list)

    # Initialize or restore session.  If checkpoint `config.load_path` is
    # given, the whole graph is initialized.  If checkpoint
    # `config.gan_load_path` is given, only the generator is initialized.

    sess = tf.Session()
    # `total_step` may change when loading checkpoints.
    total_step = 1
    if config.load_path:
        print("loading graph from '%s'.." % config.load_path)
        saver.restore(sess, config.load_path)
        total_step = sess.run(global_step)
        print("continue training at step %d.." % total_step)
    elif config.gan_load_path:
        print("loading generator and discriminator from '{}'..".format(
              config.gan_load_path))
        sess.run(init)
        gan_saver.restore(sess, config.gan_load_path)
    else:
        print("initializing graph..")
        sess.run(init)

    # For mnist, `.callback_before_train` just prints out the clipper
    # information.

    supervisor.callback_before_train(sess, total_step)

    early_stop = False
    for epoch in range(config.num_epoch):
        gen_losses = []
        disc_losses = []
        num_steps = dataloader.num_steps(config.batch_size * config.num_gpu)
        bar = trange(num_steps, leave=False)
        for _ in bar:
            if early_stop:
                break
            if config.total_step is not None and total_step > config.total_step:
                break
            tflearn.is_training(True, sess)
            gen_cost_value, _ = sess.run([gen_cost, gen_train_op])
            gen_losses.append(gen_cost_value)

            ret = supervisor.callback_before_iter(sess, total_step)
            num_critic = ret["num_critic"]
            for i in range(num_critic):
                disc_cost_value = supervisor.callback_disc_iter(
                    sess, total_step, i, real_data, dataloader,
                    accountant=accountant)
                if i == num_critic - 1:
                    disc_losses.append(disc_cost_value)
            bar.set_description("gen loss: %.4f, disc loss: %.4f" % (gen_cost_value, disc_cost_value))

            tflearn.is_training(False, sess)
            if total_step % config.image_every == 0 and config.image_dir:
                generated = sess.run(fake_data)
                generate_images(generated, dataloader.mode(),
                                join(config.image_dir, "gen_step_%d.jpg" % total_step))
                generate_images(np.concatenate(
                    [dataloader.next_batch(config.batch_size)[0] for _ in range(config.num_gpu)], axis=0),
                    dataloader.mode(),
                    join(config.image_dir, "real_step_%d.jpg" % total_step))

            if total_step % config.save_every == 0 and config.save_dir:
                saver.save(sess, join(config.save_dir, "model"), write_meta_graph=False,
                           global_step=global_step)

            if total_step % config.log_every == 0 and accountant and config.log_path:
                spent_eps_deltas = accountant.get_privacy_spent(
                    sess, target_eps=config.target_epsilons)

                with open(config.log_path, "a") as log:
                    log.write("privacy log at step: %d\n" % total_step)
                    for spent_eps, spent_delta in spent_eps_deltas:
                        to_print = "spent privacy: eps %.4f delta %.5g" % (spent_eps, spent_delta)
                        log.write(to_print + "\n")
                    log.write("\n")

            if total_step % 10 == 0 and accountant and config.terminate:
                spent_eps_deltas = accountant.get_privacy_spent(
                    sess, target_eps=config.target_epsilons)

                for (spent_eps, spent_delta), target_delta in zip(
                        spent_eps_deltas, config.target_deltas):
                    print("e-d-privacy Delta:")
                    print("\t spent:", spent_delta)
                    print("\t target:", target_delta)
                    if spent_delta > target_delta:
                        early_stop = True
                        print("terminating at step %d.." % total_step)
                        break

            total_step += 1
        bar.close()

    if config.save_dir:
        saver.save(sess, join(config.save_dir, "model"), write_meta_graph=False, global_step=global_step)


def train(config, dataloader, generator_forward, discriminator_forward,
          disc_optimizer, gen_optimizer,
          supervisor, accountant=None):
    print("parameters:", config)

    for folder in (config.image_dir, config.save_dir):
        if folder:
            os.makedirs(folder, exist_ok=True)

    print("building graph...")
    global_step = tf.Variable(0, trainable=False)
    real_data = tf.compat.v1.placeholder(tf.float32, shape=[config.num_gpu * config.batch_size] + dataloader.shape())
    fake_data = generator_forward(config, num_samples=config.num_gpu * config.batch_size)

    gen_train_op, gen_cost = get_train_ops(config, real_data, fake_data, global_step,
                                         discriminator_forward,
                                         disc_optimizer=disc_optimizer,
                                         gen_optimizer=gen_optimizer,
                                         accountant=accountant,
                                         supervisor=supervisor)
    print("graph built.")

    train_steps(config, dataloader, real_data,
                fake_data, global_step, gen_train_op, gen_cost, accountant=accountant,
                supervisor=supervisor)
    print("done with parameters:", config)
