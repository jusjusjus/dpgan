#!/usr/bin/env python

from os import environ as env
from os import makedirs
from os.path import join
from sys import path
path.insert(0, '.')

data_dir = env.get('DATA', './data')

from utils.parsers import create_dp_parser

parser = create_dp_parser()
parser.add_argument("--dim", default=64, type=int, dest="dim")
parser.add_argument("--data-dir", default=join(data_dir, "mnist"))
parser.add_argument("--sample-seed", type=int, default=42)
parser.add_argument("--exclude-test", action="store_true")
parser.add_argument("--sample-ratio", type=float)
parser.add_argument("--exclude-train", action="store_true")
parser.add_argument("--adaptive-rate", action="store_true")
parser.add_argument("--learning-rate", default=2e-4, type=float)
parser.add_argument("--gen-learning-rate", default=2e-4, type=float)

opt = parser.parse_args()

import numpy as np
import tensorflow as tf

from dp.train import train
from dp.supervisors.basic_mnist import BasicSupervisorMNIST
from utils.clippers import get_clipper
from utils.schedulers import get_scheduler
from utils.accounting import GaussianMomentsAccountant
from utils.data_utils import MNISTLoader
from models.gans import mnist


opt.dataset = "mnist"
if hasattr(opt, 'output'):
    opt.save_dir = join(opt.output, 'checkpoints')
    makedirs(opt.save_dir, exist_ok=True)
    opt.log_path = join(opt.output, 'logs.txt')
    opt.log_dir = join(opt.output, 'summaries')

opt.sigma = opt.sigma or 1.0

np.random.seed()
if opt.enable_accounting:
    if opt.delta and opt.epsilon:
        opt.sigma = np.sqrt(2.0 * np.log(1.25 / opt.delta)) \
                       / opt.epsilon
        print("Changing sigma to:")
        print(f">  sigma = {opt.sigma}")
    else:
        assert opt.sigma is not None, f"""
        Either --sigma, or --delta and --epsilon have to be provided"""

datakw = {
    'include_test': not opt.exclude_test,
    'include_train': not opt.exclude_train
}
if opt.sample_ratio is None:
    dataloader = MNISTLoader(opt.data_dir, **datakw)
else:
    assert 0.0 < opt.sample_ratio < 1.0
    datakw['seed'] = opt.sample_seed
    first = int(50000 * (1 - opt.sample_ratio))
    last = int(50000 * opt.sample_ratio)
    dataloader = MNISTLoader(opt.data_dir, first=first, **datakw)
    sampleloader = MNISTLoader(opt.data_dir, last=last, **datakw)

accountant = GaussianMomentsAccountant(dataloader.n, opt.moment) \
             if opt.enable_accounting else None

lr = tf.compat.v1.placeholder(tf.float32, shape=()) if opt.adaptive_rate \
     else opt.learning_rate

optkw = {'beta1': 0.5, 'beta2': 0.9}
gen_optimizer = tf.compat.v1.train.AdamOptimizer(
                    opt.gen_learning_rate, **optkw)
disc_optimizer = tf.compat.v1.train.AdamOptimizer(lr, **optkw)

clipper_ret = get_clipper(opt.clipper, opt)
if isinstance(clipper_ret, tuple):
    clipper, sampler = clipper_ret
    sampler.set_data_loader(sampleloader)
    sampler.keep_memory = False
else:
    clipper = clipper_ret
    sampler = None

scheduler = get_scheduler(opt.scheduler, opt)

def callback_before_train(_0, _1, _2):
    """called in `dp.train.train_steps` before training starts"""
    print(clipper.info())

supervisor = BasicSupervisorMNIST(opt, clipper, scheduler, sampler=sampler,
                                  callback_before_train=callback_before_train)
if opt.adaptive_rate:
    supervisor.put_key("lr", lr)

with open(opt.log_path, 'w') as fp:
    fp.write("Input Parameters:\n" + str(opt) + "\n\n")

train(opt, dataloader, mnist.generator_forward,
      mnist.discriminator_forward, gen_optimizer=gen_optimizer,
      disc_optimizer=disc_optimizer, accountant=accountant,
      supervisor=supervisor)
