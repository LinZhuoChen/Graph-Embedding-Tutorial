"This file contains common function in recommand system using tf"
import tensorflow as tf
from collections import namedtuple, OrderedDict
from copy import copy
from itertools import chain

from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.layers import Input, Lambda

import tensorflow as tf
from tensorflow.python.keras.layers import Flatten

LINEAR_SCOPE_NAME = 'linear'
DNN_SCOPE_NAME = 'dnn'

def get_train_op_fn(linear_optimizer, dnn_optimizer):
    def _train_op_fn(loss):
        train_ops = []
        try:
            global_step = tf.train.get_global_step()
        except AttributeError:
            global_step = tf.compat.v1.train.get_global_step()
        linear_var_list = get_collection(get_GraphKeys().TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)
        dnn_var_list = get_collection(get_GraphKeys().TRAINABLE_VARIABLES, DNN_SCOPE_NAME)

        if len(dnn_var_list) > 0:
            train_ops.append(
                dnn_optimizer.minimize(
                    loss,
                    var_list=dnn_var_list))
        if len(linear_var_list) > 0:
            train_ops.append(
                linear_optimizer.minimize(
                    loss,
                    var_list=linear_var_list))

        train_op = tf.group(*train_ops)
        with tf.control_dependencies([train_op]):
            try:
                return tf.assign_add(global_step, 1).op
            except AttributeError:
                return tf.compat.v1.assign_add(global_step, 1).op

    return _train_op_fn


def variable_scope(name_or_scope):
    try:
        return tf.variable_scope(name_or_scope)
    except AttributeError:
        return tf.compat.v1.variable_scope(name_or_scope)

def get_collection(key, scope=None):
    try:
        return tf.get_collection(key, scope=scope)
    except AttributeError:
        return tf.compat.v1.get_collection(key, scope=scope)


def get_GraphKeys():
    try:
        return tf.GraphKeys
    except AttributeError:
        return tf.compat.v1.GraphKeys


def get_losses():
    try:
        return tf.compat.v1.losses
    except AttributeError:
        return tf.losses


def input_layer(features, feature_columns):
    try:
        return tf.feature_column.input_layer(features, feature_columns)
    except AttributeError:
        return tf.compat.v1.feature_column.input_layer(features, feature_columns)


def get_metrics():
    try:
        return tf.compat.v1.metrics
    except AttributeError:
        return tf.metrics


def to_float(x, name="ToFloat"):
    try:
        return tf.to_float(x, name)
    except AttributeError:
        return tf.compat.v1.to_float(x, name)


def summary_scalar(name, data):
    try:
        tf.summary.scalar(name, data)
    except AttributeError:  # tf version 2.5.0+:AttributeError: module 'tensorflow._api.v2.summary' has no attribute 'scalar'
        tf.compat.v1.summary.scalar(name, data)

def linear_model(features, linear_feature_columns):
    if tf.__version__ >= '2.0.0':
        linear_logits = tf.compat.v1.feature_column.linear_model(features, linear_feature_columns)
    else:
        linear_logits = tf.feature_column.linear_model(features, linear_feature_columns)
    return linear_logits


def get_linear_logit(features, linear_feature_columns, l2_reg_linear=0):
    with variable_scope(LINEAR_SCOPE_NAME):
        if not linear_feature_columns:
            linear_logits = tf.Variable([[0.0]], name='bias_weights')
        else:

            linear_logits = linear_model(features, linear_feature_columns)

            if l2_reg_linear > 0:
                for var in get_collection(get_GraphKeys().TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)[:-1]:
                    get_losses().add_loss(l2_reg_linear * tf.nn.l2_loss(var, name=var.name.split(":")[0] + "_l2loss"),
                                          get_GraphKeys().REGULARIZATION_LOSSES)
    return linear_logits


def input_from_feature_columns(features, feature_columns, l2_reg_embedding=0.0):
    dense_value_list = []
    sparse_emb_list = []
    for feat in feature_columns:
        if is_embedding(feat):
            sparse_emb = tf.expand_dims(input_layer(features, [feat]), axis=1)
            sparse_emb_list.append(sparse_emb)
            if l2_reg_embedding > 0:
                get_losses().add_loss(l2_reg_embedding * tf.nn.l2_loss(sparse_emb, name=feat.name + "_l2loss"),
                                      get_GraphKeys().REGULARIZATION_LOSSES)

        else:
            dense_value_list.append(input_layer(features, [feat]))

    return sparse_emb_list, dense_value_list


def is_embedding(feature_column):
    try:
        from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn
    except:
        EmbeddingColumn = _EmbeddingColumn
    return isinstance(feature_column, (_EmbeddingColumn, EmbeddingColumn))

# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):


        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x
    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        self.seed = seed
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(self.seed),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[1][-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(self.seed),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])

        return tf.keras.layers.add(inputs)


def add_func(inputs):
    return Add()(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")
