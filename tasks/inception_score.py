
"""Code derived from
tensorflow/tensorflow/models/image/imagenet/classify_image.py"""

import sys
import tarfile
from os import makedirs, stat
from os.path import exists, join
from six.moves import urllib

import numpy as np
import tensorflow as tf


MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


def check(image):
    assert type(image) == np.ndarray
    assert len(image.shape) == 3
    assert np.max(image) > 10
    assert np.min(image) >= 0
    return np.expand_dims(image.astype(np.float32), 0)


def get_inception_score(images, splits=10):
  """return inception score with bootstrapped standard deviation

  Call this function with list of images. Each of elements should be a numpy
  array with values ranging from 0 to 255."""

  inps = list(map(check, images))
  with tf.compat.v1.Session() as sess:
    preds = []
    for inp in inps:
        sys.stdout.write(".")
        sys.stdout.flush()
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      P_yx = preds[i::splits]
      P_y = np.mean(P_yx, 0, keepdims=True)
      KL_div = np.sum(P_yx * (np.log(P_yx) - np.log(P_y)), axis=1)
      s_G = np.exp(np.mean(KL_div))
      scores.append(s_G)
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  makedirs(MODEL_DIR, exist_ok=True)
  filename = DATA_URL.split('/')[-1]
  filepath = join(MODEL_DIR, filename)
  if not exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  protobuf = join(MODEL_DIR, 'classify_image_graph_def.pb')
  with tf.gfile.GFile(protobuf, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  # This works with an arbitrary minibatch size.

  with tf.compat.v1.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                new_shape.append(None if s == 1 and j == 0 else s)

            o.set_shape(tf.TensorShape(new_shape))

    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [0, 1]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()
