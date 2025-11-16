# -*- coding:utf-8 -*-
"""
本地可修改版本
"""

import json
import logging
import requests
from collections import namedtuple
from threading import Thread

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda


class NegativeSampler(
    namedtuple('NegativeSampler', ['sampler', 'num_sampled', 'item_name', 'item_count', 'distortion'])):
    """ NegativeSampler
    Args:
        sampler: sampler name,['inbatch', 'uniform', 'frequency' 'adaptive',] .
        num_sampled: negative samples number per one positive sample.
        item_name: pkey of item features .
        item_count: global frequency of item .
        distortion: skew factor of the unigram probability distribution.
    """
    __slots__ = ()

    def __new__(cls, sampler, num_sampled, item_name, item_count=None, distortion=1.0, ):
        if sampler not in ['inbatch', 'uniform', 'frequency', 'adaptive']:
            raise ValueError(' `%s` sampler is not supported ' % sampler)
        if sampler in ['inbatch', 'frequency'] and item_count is None:
            raise ValueError(' `item_count` must not be `None` when using `inbatch` or `frequency` sampler')
        return super(NegativeSampler, cls).__new__(cls, sampler, num_sampled, item_name, item_count, distortion)


def l2_normalize(x, axis=-1):
    return Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)


def inner_product(x, y, temperature=1.0):
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1])) / temperature)([x, y])


def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)


def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)


def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)

