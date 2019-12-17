#!/usr/bin/env python

"""generate images from model checkpoint

Images are saved to folder specified by `--save-dir`."""

from os import makedirs
from sys import path
path.insert(0, '.')
from argparse import ArgumentParser 

from utils.generate import generate_steps_png

parser = ArgumentParser()
parser.add_argument("--dim", default=64, type=int)
parser.add_argument("--model", type=str, default='mnist', choices=['mnist', 'celeba', 'lsun'])
parser.add_argument("--times", default=7, type=int)
parser.add_argument("--params", type=str, default=None, help="Path to parameters")
parser.add_argument("--save-dir", type=str, default='./cache/images')
parser.add_argument("--batch-size", default=16, type=int)
opt = parser.parse_args()

if opt.model is 'mnist':
    from models.gans.mnist import generator_forward
elif opt.model in ('celeba', 'lsun'):
    from models.gans.d48_resnet_dcgan import generator_forward
else:
    raise ValueError(f"Unkown model {opt.model}")

generate_steps_png(opt, generator_forward)
