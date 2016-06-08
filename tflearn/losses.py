from __future__ import division, print_function, absolute_import

import tensorflow as tf
from .utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'regularizer')


def L2(tensor, wd=0.001):
    """ L2.

    Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2 * wd

    Arguments:
        tensor: `Tensor`. The tensor to apply regularization.
        wd: `float`. The decay.

    Returns:
        The regularization `Tensor`.

    """
    return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')


def L1(tensor, wd=0.001):
    """ L1.

    Computes the L1 norm of a tensor:

      output = sum(|t|) * wd

    Arguments:
        tensor: `Tensor`. The tensor to apply regularization.
        wd: `float`. The decay.

    Returns:
        The regularization `Tensor`.

    """
    return tf.mul(tf.reduce_sum(tf.abs(tensor)), wd, name='L1-Loss')
    
def weak_cross_entropy_2d_loss(logits, labels, num_classes, epsilon=0.0001, head=None):
    """Calculate the semantic segmentation using weak softmax cross entropy loss.

    Given 2d image shaped logits and corresponding labels, this calculated the
    widely used semantic segmentation loss. Using `tf.nn.softmax_cross_entropy_with_logits`
    is currently not supported. See https://github.com/tensorflow/tensorflow/issues/2327#issuecomment-224491229

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes].
          Weighting the loss of each class.

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        shape = [logits.get_shape()[0], num_classes]
        epsilon = tf.constant(value=epsilon, shape=shape)
        logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits)

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1]))

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
