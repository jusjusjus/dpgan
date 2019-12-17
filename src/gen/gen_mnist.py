#!/usr/bin/env python

from sys import path
path.insert(0, '.')

from utils.parsers import create_gen_parser
from utils.generate import generate_steps
from models.gans.mnist import generator_forward


if __name__ == "__main__":
    parser = create_gen_parser()
    parser.add_argument("--dim", dest="dim", default=64, type=int)

    generate_steps(parser.parse_args(), generator_forward)
