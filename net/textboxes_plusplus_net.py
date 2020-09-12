from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import textboxes_plusplus_config as config


def forward_module(m, inputs, training=False):
    # BatchNormalization and Dropout layers behave differently in training and
    # testing (prediction) phase.
    if isinstance(m, tf.layers.BatchNormalization) or\
            isinstance(m, tf.layers.Dropout):
        return m.apply(inputs, training=training)
    return m.apply(inputs)


class VGG16Backbone(object):
    def __init__(self, data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format
        self._conv_initializer = tf.glorot_uniform_initializer
        # VGG layers
        # 384x384
        self._conv1_block = self.conv_block(2, 64, 3, (1, 1), 'conv1')
        self._pool1 = tf.layers.MaxPooling2D(2,
                                             2,
                                             padding='same',
                                             data_format=self._data_format,
                                             name='pool1')
        # 192x192
        self._conv2_block = self.conv_block(2, 128, 3, (1, 1), 'conv2')
        self._pool2 = tf.layers.MaxPooling2D(2,
                                             2,
                                             padding='same',
                                             data_format=self._data_format,
                                             name='pool2')
        # 96x96
        self._conv3_block = self.conv_block(3, 256, 3, (1, 1), 'conv3')
        self._pool3 = tf.layers.MaxPooling2D(2,
                                             2,
                                             padding='same',
                                             data_format=self._data_format,
                                             name='pool3')
        # 48x48
        self._conv4_block = self.conv_block(3, 512, 3, (1, 1), 'conv4')
        self._pool4 = tf.layers.MaxPooling2D(2,
                                             2,
                                             padding='same',
                                             data_format=self._data_format,
                                             name='pool4')
        # 24x24
        self._conv5_block = self.conv_block(3, 512, 3, (1, 1), 'conv5')
        self._pool5 = tf.layers.MaxPooling2D(3,
                                             1,
                                             padding='same',
                                             data_format=self._data_format,
                                             name='pool5')
        self._conv6 = tf.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=6,
            data_format=self._data_format,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=self._conv_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='fc6',
            _scope='fc6',
            _reuse=None)
        self._conv7 = tf.layers.Conv2D(
            filters=1024,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format=self._data_format,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=self._conv_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='fc7',
            _scope='fc7',
            _reuse=None)
        # 24x24
        # TextBoxes++ layers
        with tf.variable_scope('additional_layers'):
            self._conv8_block = self.ssd_conv_block(256, 2, 'conv8')
            self._conv9_block = self.ssd_conv_block(128, 2, 'conv9')
            self._conv10_block = self.ssd_conv_block(128,
                                                     1,
                                                     'conv10',
                                                     padding='valid')
            self._conv11_block = self.ssd_conv_block(128,
                                                     1,
                                                     'conv11',
                                                     padding='valid')
            self._conv12_block = self.ssd_conv_block(128,
                                                     2,
                                                     'conv12')

    def l2_normalize(self, x, name):
        with tf.name_scope(name, 'l2_normalize', [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            # across_spatial=False
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            # eb = x[i, j, k, 0]
            # eg = x[i, j, k, 1]
            # er = x[i, j, k, 2]
            # return:
            # x[i, j, k, 0] = eb/sqrt(eb^2 + eg^2 + er^2)
            # x[i, j, k, 1] = eg/sqrt(eb^2 + eg^2 + er^2)
            # x[i, j, k, 2] = er/sqrt(eb^2 + eg^2 + er^2)
            return tf.multiply(x, x_inv_norm, name=name)

    def forward(self, inputs, training=False):
        # inputs should in BGR
        feature_layers = []
        # forward vgg layers
        for conv in self._conv1_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool1.apply(inputs)
        for conv in self._conv2_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool2.apply(inputs)
        for conv in self._conv3_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool3.apply(inputs)
        for conv in self._conv4_block:
            inputs = forward_module(conv, inputs, training=training)
        # conv4_3
        with tf.variable_scope('conv4_3_scale'):
            # channel_shared=False
            weight_scale =\
                tf.Variable([20.] * 512, trainable=training, name='weights')
            if self._data_format == 'channels_last':
                weight_scale =\
                    tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            else:
                weight_scale =\
                    tf.reshape(weight_scale, [1, -1, 1, 1], name='reshape')

            feature_layers.append(
                tf.multiply(weight_scale,
                            self.l2_normalize(inputs, name='norm'),
                            name='rescale'))
        inputs = self._pool4.apply(inputs)
        for conv in self._conv5_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool5.apply(inputs)
        # forward fc layers
        # fc6
        inputs = self._conv6.apply(inputs)
        # fc7
        inputs = self._conv7.apply(inputs)
        feature_layers.append(inputs)
        # forward textboxes++ layers
        # conv8
        for layer in self._conv8_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        # conv9
        for layer in self._conv9_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        # conv10
        for layer in self._conv10_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        # conv11
        for layer in self._conv11_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        # conv12
        if config.NUM_FEATURE_LAYERS == 7:
            for layer in self._conv12_block:
                inputs = forward_module(layer, inputs, training=training)
            feature_layers.append(inputs)
        return feature_layers

    def conv_block(self,
                   num_layers_per_block,
                   filters,
                   kernel_size,
                   strides,
                   name,
                   reuse=None):
        with tf.variable_scope(name):
            conv_layers = []
            for ind in range(1, num_layers_per_block + 1):
                conv_layers.append(
                    tf.layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        data_format=self._data_format,
                        activation=tf.nn.relu,
                        use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_{}'.format(name, ind),
                        _scope='{}_{}'.format(name, ind),
                        _reuse=None)
                    )
            return conv_layers

    def ssd_conv_block(self,
                       filters,
                       strides,
                       name,
                       padding='same',
                       reuse=None):
        with tf.variable_scope(name):
            conv_layers = []
            conv_layers.append(
                    tf.layers.Conv2D(
                        filters=filters,
                        kernel_size=1,
                        strides=1,
                        padding=padding,
                        data_format=self._data_format,
                        activation=tf.nn.relu,
                        use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_1'.format(name),
                        _scope='{}_1'.format(name),
                        _reuse=None)
                )
            conv_layers.append(
                    tf.layers.Conv2D(
                        filters=filters * 2,
                        kernel_size=3,
                        strides=strides,
                        padding=padding,
                        data_format=self._data_format,
                        activation=tf.nn.relu,
                        use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_2'.format(name),
                        _scope='{}_2'.format(name),
                        _reuse=None)
                )
            return conv_layers


def multibox_head(feature_layers,
                  num_classes,
                  num_offsets,
                  num_anchors_per_location_all_layers,
                  data_format='channels_first'):
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feature_layer in enumerate(feature_layers):
            loc_preds.append(
                tf.layers.conv2d(
                    feature_layer,
                    num_anchors_per_location_all_layers[ind] * 2 * num_offsets,
                    (3, 5),
                    use_bias=True,
                    name='loc_{}'.format(ind),
                    strides=(1, 1),
                    padding='same',
                    data_format=data_format,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer()))

            cls_preds.append(
                tf.layers.conv2d(
                    feature_layer,
                    num_anchors_per_location_all_layers[ind] * 2 * num_classes,
                    (3, 5),
                    use_bias=True,
                    name='cls_{}'.format(ind),
                    strides=(1, 1),
                    padding='same',
                    data_format=data_format,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer()))

        return loc_preds, cls_preds


def multibox_head_with_inception_module(
    feature_layers,
    num_classes,
    num_offsets,
    num_anchors_per_location_all_layers,
        data_format='channels_first'):
    with tf.variable_scope('multibox_head_with_inception_module'):
        cls_preds = []
        loc_preds = []

        def branch(feature_layer,
                   layer_index,
                   depth,
                   kernel_size,
                   loc_or_cls,
                   branch_index):
            one_by_one =\
                tf.layers.conv2d(
                    feature_layer,
                    depth,
                    (1, 1),
                    use_bias=True,
                    name='{}_{}_{}_one_by_one'.format(loc_or_cls,
                                                      layer_index,
                                                      branch_index),
                    strides=(1, 1),
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer())
            return tf.layers.conv2d(
                    one_by_one,
                    depth * 2,
                    kernel_size,
                    use_bias=True,
                    name='{}_{}_branch_{}'.format(loc_or_cls,
                                                  layer_index,
                                                  branch_index),
                    strides=(1, 1),
                    padding='same',
                    data_format=data_format,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer())
        for ind, feature_layer in enumerate(feature_layers):
            used_kernels = [(1, 6), (6, 1), (3, 5)]
            branches =\
                [branch(feature_layer,
                        ind,
                        num_anchors_per_location_all_layers[ind] * num_offsets,
                        kernel,
                        'loc',
                        i+1) for i, kernel in enumerate(used_kernels)]
            maxpooling =\
                tf.layers.MaxPooling2D(
                    3,
                    1,
                    padding='same',
                    data_format=data_format,
                    name='loc_{}_maxpooling'.format(ind))
            last_branch =\
                tf.layers.conv2d(
                    maxpooling.apply(feature_layer),
                    num_anchors_per_location_all_layers[ind] * 2 * num_offsets,
                    (1, 1),
                    use_bias=True,
                    name='loc_{}_pooling_one_by_one'.format(ind),
                    strides=(1, 1),
                    padding='same',
                    data_format=data_format,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer())
            stacked_inception_branches = tf.stack(
                branches + [last_branch],
                axis=-1)
            loc_preds.append(
                tf.reduce_mean(
                    stacked_inception_branches,
                    axis=-1))

            branches =\
                [branch(feature_layer,
                        ind,
                        num_anchors_per_location_all_layers[ind] * num_classes,
                        kernel,
                        'cls',
                        i+1) for i, kernel in enumerate(used_kernels)]
            maxpooling =\
                tf.layers.MaxPooling2D(
                    3,
                    1,
                    padding='same',
                    data_format=data_format,
                    name='cls_{}_maxpooling'.format(ind))
            last_branch =\
                tf.layers.conv2d(
                    maxpooling.apply(feature_layer),
                    num_anchors_per_location_all_layers[ind] * 2 * num_classes,
                    (1, 1),
                    use_bias=True,
                    name='cls_{}_pooling_one_by_one'.format(ind),
                    strides=(1, 1),
                    padding='same',
                    data_format=data_format,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.zeros_initializer())
            stacked_inception_branches =\
                tf.stack(
                    branches + [last_branch],
                    axis=-1)
            cls_preds.append(
                tf.reduce_mean(
                    stacked_inception_branches,
                    axis=-1))

        return loc_preds, cls_preds
