#!/usr/bin/env python

from os.path import join, splitext
from sys import path
path.insert(0, '.')
from glob import glob
from argparse import ArgumentParser 
from functools import partial

parser = ArgumentParser()
parser.add_argument("folder", type=str, help="folder with images")
parser.add_argument("--model", type=str, default='mnist',
                    help="pretrained model to use")
parser.add_argument("--model-path", type=str,
                    default='./cache/mnist/classifier/model-10153',
                    help="parameters of pretrained model")
opt = parser.parse_args()


import numpy as np
from PIL import Image


def read_images(folder):
    filenames = glob(join(opt.folder, '*.png'))
    assert all(splitext(f)[1] in ('.png', '.jpg') for f in filenames)
    print(f"Found {len(filenames)} images in '{opt.folder}'")
    images = map(Image.open, filenames)
    return map(np.array, images)


images = read_images(opt.folder)

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
