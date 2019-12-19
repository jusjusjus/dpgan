
import tensorflow as tf
import numpy as np
import tflearn

from models.tasks.mnist import classifier_forward


def next_batch(images_iter, batch_size=100):
    count = 0
    elements = []
    for img in images_iter:
        elements.append(img[None])
        count += 1
        if count == batch_size:
            yield np.concatenate(elements, axis=0)
            elements.clear()
            count = 0

    if count > 0:
        yield np.concatenate(elements, axis=0)


def get_mnist_score(images_iter, model_path, batch_size=100, splits=10):
    tf.reset_default_graph()

    incoming = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="input")
    logits = classifier_forward(None, incoming, name="classifier")
    probs = tf.nn.softmax(logits)
    saver = tf.train.Saver([var for var in tf.global_variables()
                            if var.name.startswith("classifier")
                            and not var.name.endswith("is_training:0")])

    preds, scores = [], []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tflearn.is_training(False, sess)
        saver.restore(sess, model_path)

        for images in next_batch(images_iter, batch_size):
            pred = sess.run(probs, feed_dict={incoming: images})
            preds.append(pred)

    preds = np.concatenate(preds, 0)
    for i in range(splits):
        P_yx = preds[i::splits]
        P_y = np.mean(P_yx, 0, keepdims=True)
        P_yx = np.maximum(P_yx, 1e-12)
        P_y = np.maximum(P_y, 1e-12)
        KL_div = np.sum(P_yx * (np.log(P_yx) - np.log(P_y)), 1)
        s_G = np.exp(np.mean(KL_div))
        scores.append(s_G)

    return np.mean(scores), np.std(scores)
