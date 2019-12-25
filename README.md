# README #

# Install

```bash
conda create -n dpgan python=3.6 pip
conda activate dpgan
pip install -r requirements.txt
```
## MNIST

To train a generator on MNIST, run:

```bash
./scripts/train.py --image-every 250 --save-every 250 -e 1000 --log-every 250 \
                   -g 2 --accounting --terminate \
                   --target-deltas 1e-5 --target-epsilons 4.0 --sigma 1.086 \
                   --clipper mnist_est --sample-ratio 0.05 \
                   -o ./cache/mnist/gen
```

Next, you want to compute the inception score of the trained generator.
First train an MNIST classifier on the original dataset:

```bash
./src/tasks/baseline_mnist_train.py --save-dir ./cache/mnist/clf
```

After training has finished, locate the generator checkpoints at
"./cache/mnist/gen/checkpoints" and the classifier checkpoints at
"./cache/mnist/clf".  You are ready to compute the inception score for your
MNIST generator:

```bash
./scripts/inception.py ./cache/mnist/gen/checkpoints/<gen-ckpt> \
                       --model-path ./cache/mnist/clf/<clf-ckpt>
```

### DPGAN project ###

Author: Xinyang Zhang
Modified by: Justus Schwabedal
