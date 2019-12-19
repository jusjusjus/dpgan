#!/usr/bin/env python

from os.path import join, splitext
from sys import path
path.insert(0, '.')
from glob import glob
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument("folder", type=str, help="folder with images")
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

from tasks.inception_score import get_inception_score

I, dI = get_inception_score(images)
print(f"Inception score: {I:.3f} +/- {dI:.3f}")
