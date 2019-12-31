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
config.dataset = "mnist"
if hasattr(config, 'output'):
    config.save_dir = join(config.output, 'checkpoints')
    makedirs(config.save_dir, exist_ok=True)
    config.log_path = join(config.output, 'logs.txt')
    config.log_dir = join(config.output, 'summaries')

config.sigma = config.sigma or 1.0

np.random.seed()
if config.enable_accounting:
    if config.delta and config.epsilon:
        config.sigma = np.sqrt(2.0 * np.log(1.25 / config.delta)) \
                       / config.epsilon
        print("Changing sigma to:")
        print(f">  sigma = {config.sigma}")
    else:
        assert config.sigma is not None, f"""
        Either --sigma, or --delta and --epsilon have to be provided"""

datakw = {
    'include_test': not config.exclude_test,
    'include_train': not config.exclude_train
}
if config.sample_ratio is None:
    dataloader = MNISTLoader(config.data_dir, **datakw)
else:
    assert 0.0 < config.sample_ratio < 1.0
    datakw['seed'] = config.sample_seed
    first = int(50000 * (1 - config.sample_ratio))
    last = int(50000 * config.sample_ratio)
    dataloader = MNISTLoader(config.data_dir, first=first, **datakw)
    sampleloader = MNISTLoader(config.data_dir, last=last, **datakw)

accountant = GaussianMomentsAccountant(dataloader.n, config.moment) \
             if config.enable_accounting else None

lr = tf.compat.v1.placeholder(tf.float32, shape=()) if config.adaptive_rate \
     else config.learning_rate

optkw = {'beta1': 0.5, 'beta2': 0.9}
gen_optimizer = tf.compat.v1.train.AdamOptimizer(
                    config.gen_learning_rate, **optkw)
disc_optimizer = tf.compat.v1.train.AdamOptimizer(lr, **optkw)

clipper_ret = get_clipper(config.clipper, config)
if isinstance(clipper_ret, tuple):
    clipper, sampler = clipper_ret
    sampler.set_data_loader(sampleloader)
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

train(config, dataloader, mnist.generator_forward,
      mnist.discriminator_forward, gen_optimizer=gen_optimizer,
      disc_optimizer=disc_optimizer, accountant=accountant,
      supervisor=supervisor)
