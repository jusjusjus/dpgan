#!/usr/bin/env python

from os import environ as env
from os import makedirs
from os.path import join
from sys import path
path.insert(0, '.')

import numpy as np
import tensorflow as tf

from dp.train import train
from dp.supervisors.basic_mnist import BasicSupervisorMNIST
from utils.parsers import create_dp_parser
from utils.clippers import get_clipper
from utils.schedulers import get_scheduler
from utils.accounting import GaussianMomentsAccountant
from utils.data_utils import MNISTLoader
from models.gans import mnist

data_dir = env.get('DATA', './data')

if __name__ == "__main__":
    parser = create_dp_parser()
    parser.add_argument("--dim", default=64, type=int, dest="dim")
    parser.add_argument("--data-dir", default=join(data_dir, "mnist"))
    parser.add_argument("--sample-seed", type=int, default=1024)
    parser.add_argument("--exclude-test", action="store_true")
    parser.add_argument("--sample-ratio", type=float)
    parser.add_argument("--exclude-train", action="store_true")
    parser.add_argument("--adaptive-rate", action="store_true")
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--gen-learning-rate", default=2e-4, type=float)

    config = parser.parse_args()
    if hasattr(config, 'output'):
        config.image_dir = join(config.output, 'images')
        config.save_dir = join(config.output, 'checkpoints')
        for folder in (config.image_dir, config.save_dir):
            makedirs(folder, exist_ok=True)
        config.log_path = join(config.output, 'logs.txt')
    config.dataset = "mnist"


    np.random.seed()
    if config.enable_accounting:
        if config.delta and config.epsilon:
            config.sigma = np.sqrt(2.0 * np.log(1.25 / config.delta)) \
                           / config.epsilon
            print("Changing sigma to:")
            print(f">  sigma = {config.sigma}")
        else:
            assert config.sigma is not None, f"""
            Either --sigma or --delta and --epsilon have to be provided"""

    datakw = {
        'include_test': not config.exclude_test,
        'include_train': not config.exclude_train
    }
    if config.sample_ratio is not None:
        assert 0.0 < config.sample_ratio < 1.0
        datakw['seed'] = config.sample_seed
        first = int(50000 * (1 - config.sample_ratio))
        last = int(50000 * config.sample_ratio)
        gan_data_loader = MNISTLoader(config.data_dir, first=first, **datakw)
        sample_data_loader = MNISTLoader(config.data_dir, last=last, **datakw)
    else:
        gan_data_loader = MNISTLoader(config.data_dir, **datakw)

    accountant = GaussianMomentsAccountant(gan_data_loader.n, config.moment) \
                 if config.enable_accounting else None

    if config.adaptive_rate:
        lr = tf.compat.v1.placeholder(tf.float32, shape=())
    else:
        lr = config.learning_rate

    optkw = {'beta1': 0.5, 'beta2': 0.9}
    gen_optimizer = tf.compat.v1.train.AdamOptimizer(
                        config.gen_learning_rate, **optkw)
    disc_optimizer = tf.compat.v1.train.AdamOptimizer(lr, **optkw)

    clipper_ret = get_clipper(config.clipper, config)
    if isinstance(clipper_ret, tuple):
        clipper, sampler = clipper_ret
        sampler.set_data_loader(sample_data_loader)
        sampler.keep_memory = False
    else:
        clipper = clipper_ret
        sampler = None

    scheduler = get_scheduler(config.scheduler, config)

    def callback_before_train(_0, _1, _2):
        """called in `dp.train.train_steps` before training starts"""
        print(clipper.info())

    supervisor = BasicSupervisorMNIST(config, clipper, scheduler, sampler=sampler,
                                      callback_before_train=callback_before_train)
    if config.adaptive_rate:
        supervisor.put_key("lr", lr)

    with open(config.log_path, 'w') as fp:
        fp.write("Input Parameters:\n" + str(config) + "\n\n")

    train(config, gan_data_loader, mnist.generator_forward,
          mnist.discriminator_forward, gen_optimizer=gen_optimizer,
          disc_optimizer=disc_optimizer, accountant=accountant,
          supervisor=supervisor)
