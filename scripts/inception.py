#!/usr/bin/env python

from os.path import join, splitext, isdir
from sys import path
path.insert(0, '.')
from glob import glob
from argparse import ArgumentParser 
from functools import partial
from collections import namedtuple

parser = ArgumentParser()
parser.add_argument("folder_or_params", type=str,
    help="folder with images or path to tensorflow model parameters")
parser.add_argument("--model", type=str, default='mnist',
                    help="pretrained model to use")
parser.add_argument("--model-path", type=str, required=True,
                    help="parameters of pretrained model")
opt = parser.parse_args()

import numpy as np
from PIL import Image

from utils.generate import generate_steps


def read_images(folder):
    filenames = glob(join(folder, '*.png'))
    assert all(splitext(f)[1] in ('.png', '.jpg') for f in filenames)
    print(f"Found {len(filenames)} images in '{opt.folder_or_params}'")
    images = map(Image.open, filenames)
    return map(np.array, images)


def generate_images(params, model='mnist', times=50):

    Config = namedtuple("Config", "dim times params batch_size")
    config = Config(dim=64, times=times, params=params, batch_size=16)

    if model is 'mnist':
        from models.gans.mnist import generator_forward
    elif model in ('celeba', 'lsun'):
        from models.gans.d48_resnet_dcgan import generator_forward

    images = generate_steps(config, generator_forward)
    return images



if isdir(opt.folder_or_params):
    images = read_images(opt.folder_or_params)
else:
    images = generate_images(opt.folder_or_params)

# Compute inception score.

if opt.model == 'imagenet':
    from tasks.inception_score import get_inception_score
elif 'mnist' == opt.model:
    from tasks.mnist_score import get_mnist_score
    get_inception_score = partial(get_mnist_score,
                                  model_path=opt.model_path)
    images = map(lambda x: x[..., 0:1], images)

I, dI = get_inception_score(images)
print(f"Inception score: {I:.3f} +/- {dI:.3f}")
