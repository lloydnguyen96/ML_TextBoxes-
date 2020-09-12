import math

import tensorflow as tf
# import numpy as np

# from tensorflow.contrib.image.python.ops import image_ops


class AnchorProcessor(object):
    def __init__(self,
                 positive_threshold,
                 ignore_threshold,
                 prior_scaling):
        super(AnchorProcessor, self).__init__()
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling

    def center2point(self,
                     center_y,
                     center_x,
                     height,
                     width):
        with tf.name_scope('center2point'):
            return center_y - (height - 1.) / 2.,\
                   center_x - (width - 1.) / 2.,\
                   center_y + (height - 1.) / 2.,\
                   center_x + (width - 1.) / 2.,

    def point2center(self,
                     ymin,
                     xmin,
                     ymax,
                     xmax):
        with tf.name_scope('point2center'):
            height, width = (ymax - ymin + 1.), (xmax - xmin + 1.)
            return (ymin + ymax) / 2., (xmin + xmax) / 2., height, width

    def get_anchors_size_one_layer(self,
                                   anchor_scales_one_layer,
                                   extra_anchor_scales_one_layer,
                                   anchor_ratios_one_layer,
                                   name=None):
        with tf.name_scope(name, 'get_anchors_size_one_layer'):
            num_anchors_per_location_one_layer =\
                len(anchor_scales_one_layer) *\
                len(anchor_ratios_one_layer) +\
                len(extra_anchor_scales_one_layer)

            list_h_on_image_one_layer = []
            list_w_on_image_one_layer = []
            # For square anchors
            for _, scale in enumerate(extra_anchor_scales_one_layer):
                list_h_on_image_one_layer.append(scale)
                list_w_on_image_one_layer.append(scale)
            # For the other aspect ratio scaled anchors.
            for _, scale in enumerate(anchor_scales_one_layer):
                for _, ratio in enumerate(anchor_ratios_one_layer):
                    list_h_on_image_one_layer.append(scale / math.sqrt(ratio))
                    list_w_on_image_one_layer.append(scale * math.sqrt(ratio))
            return tf.constant(list_h_on_image_one_layer, dtype=tf.float32),\
                tf.constant(list_w_on_image_one_layer, dtype=tf.float32),\
                num_anchors_per_location_one_layer

    def get_anchors_size_all_layers(self,
                                    anchor_scales_all_layers,
                                    extra_anchor_scales_all_layers,
                                    anchor_ratios_all_layers,
                                    num_feature_layers,
                                    name=None):
        with tf.name_scope(name, 'get_anchors_size_all_layers'):
            anchor_heights_all_layers = []
            anchor_widths_all_layers = []
            num_anchors_per_location_all_layers = []
            for i in range(num_feature_layers):
                anchor_heights_one_layer,\
                    anchor_widths_one_layer,\
                    num_anchors_per_location_one_layer =\
                    self.get_anchors_size_one_layer(
                        anchor_scales_all_layers[i],
                        extra_anchor_scales_all_layers[i],
                        anchor_ratios_all_layers[i],
                        name='get_anchors_size_one_layer_{}'.format(i))
                anchor_heights_all_layers.append(anchor_heights_one_layer)
                anchor_widths_all_layers.append(anchor_widths_one_layer)
                num_anchors_per_location_all_layers.append(
                    num_anchors_per_location_one_layer)
        return anchor_heights_all_layers,\
            anchor_widths_all_layers,\
            num_anchors_per_location_all_layers

    def get_all_anchors_all_layers(self,
                                   image_shape,
                                   anchor_heights_all_layers,
                                   anchor_witdths_all_layers,
                                   num_anchors_per_location_all_layers,
                                   anchor_offsets,
                                   vertical_offsets,
                                   layer_shapes,
                                   feat_strides,
                                   allowed_borders,
                                   should_clips,
                                   name=None):
        with tf.name_scope(name, 'get_all_anchors_all_layers'):
            image_height, image_width =\
                tf.to_float(image_shape[0]),\
                tf.to_float(image_shape[1])  # to type float32

            anchors_ymin = []
            anchors_xmin = []
            anchors_ymax = []
            anchors_xmax = []
            anchor_allowed_borders = []
            # For each chosen feature layer.
            for i, num_anchors_per_location_one_layer in\
                    enumerate(num_anchors_per_location_all_layers):
                with tf.name_scope('get_all_anchors_one_layer_{}'.format(i)):
                    _anchors_ymin,\
                        _anchors_xmin,\
                        _anchors_ymax,\
                        _anchors_xmax =\
                        self.get_all_anchors_one_layer(
                            anchor_heights_all_layers[i],
                            anchor_witdths_all_layers[i],
                            num_anchors_per_location_one_layer,
                            layer_shapes[i],
                            feat_strides[i],
                            offset=anchor_offsets[i],
                            vertical_offset=vertical_offsets[i])
                    # _anchors_ymin: 2d-tf.Tensor(
                    #       FH * FW * 2,
                    #       num_anchors_per_location_one_layer) tf.float32
                    # _anchors_ymin: 2d-tf.Tensor(
                    #       num_locations_one_layer,
                    #       num_anchors_per_location_one_layer) tf.float32

                    if should_clips[i]:
                        _anchors_ymin = tf.clip_by_value(
                            _anchors_ymin,
                            0.,
                            image_height - 1.)
                        _anchors_xmin = tf.clip_by_value(
                            _anchors_xmin,
                            0.,
                            image_width - 1.)
                        _anchors_ymax = tf.clip_by_value(
                            _anchors_ymax,
                            0.,
                            image_height - 1.)
                        _anchors_xmax = tf.clip_by_value(
                            _anchors_xmax,
                            0.,
                            image_width - 1.)

                    _anchors_ymin = tf.reshape(_anchors_ymin, [-1])
                    _anchors_xmin = tf.reshape(_anchors_xmin, [-1])
                    _anchors_ymax = tf.reshape(_anchors_ymax, [-1])
                    _anchors_xmax = tf.reshape(_anchors_xmax, [-1])
                    # _anchors_ymin: 1d-tf.Tensor(
                    #       num_anchors_one_layer) tf.float32

                    anchors_ymin.append(_anchors_ymin)
                    anchors_xmin.append(_anchors_xmin)
                    anchors_ymax.append(_anchors_ymax)
                    anchors_xmax.append(_anchors_xmax)
                    anchor_allowed_borders.append(
                        tf.ones_like(_anchors_ymin,
                                     dtype=tf.float32) * allowed_borders[i])

            anchors_ymin = tf.concat(anchors_ymin, axis=0)
            anchors_xmin = tf.concat(anchors_xmin, axis=0)
            anchors_ymax = tf.concat(anchors_ymax, axis=0)
            anchors_xmax = tf.concat(anchors_xmax, axis=0)
            anchor_allowed_borders = tf.concat(anchor_allowed_borders, axis=0)
            # anchors_ymin: 1d-tf.Tensor(num_anchors_all_layers) tf.float32
            # anchor_allowed_borders: 1d-tf.Tensor(
            #       num_anchors_all_layers) tf.float32

            inside_mask =\
                tf.logical_and(tf.logical_and(
                    anchors_ymin > -anchor_allowed_borders,
                    anchors_xmin > -anchor_allowed_borders),
                               tf.logical_and(
                    anchors_ymax < image_height - 1. + anchor_allowed_borders,
                    anchors_xmax < image_width - 1. + anchor_allowed_borders))
            #  '>' and '<' used instead of '>=' and '<='
            #  anchors corner points will not lie on borderline, but
            #  inside

            # inside_mask: 1d-tf.Tensor(num_anchors_all_layers) tf.bool
            # e.g., image_height=image_width=anchor_allowed_border=384
            # ==> borderline and 'inside' region will look like this:
            #     -384 o o ... o o -1 0 1 o o ... o o 382 383 o o ... o o 767
            # -384  i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   .   .  . . ... . .  . . . . . ... . .  .   .  . . ... . .  .
            #   .   .  . . ... . .  . . . . . ... . .  .   .  . . ... . .  .
            #   .   .  . . ... . .  . . . . . ... . .  .   .  . . ... . .  .
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #  -1   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #                         ________...____________              o
            #   0   i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #   1   i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #   o   i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #   o   i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #   .   .  . . ... . .  .|. . . . ... . .  .   . |. . ... . .  .
            #   .   .  . . ... . .  .|. . . . ... . .  .   . |. . ... . .  .
            #   .   .  . . ... . .  .|. . . . ... . .  .   . |. . ... . .  .
            #   o   i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #   o   i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #  382  i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #  383  i  i i ... i i  i|i i i i ... i i  i   i |i i ... i i  o
            #                        |________...____________|             o
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   .   .  . . ... . .  . . . . . ... . .  .   .  . . ... . .  .
            #   .   .  . . ... . .  . . . . . ... . .  .   .  . . ... . .  .
            #   .   .  . . ... . .  . . . . . ... . .  .   .  . . ... . .  .
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #   o   i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #  766  i  i i ... i i  i i i i i ... i i  i   i  i i ... i i  o
            #  767  o  o o ... o o -1 0 1 o o ... o o 382 383 o o ... o o  o
            #  in which:
            #  - i means inside (valid value)
            #  - o means outside (invalid value)
            #  for anchor_x and anchor_y
            #  Why did we choose such a large region for valid anchor_x,
            #  anchor_y?

            return anchors_ymin,\
                anchors_xmin,\
                anchors_ymax,\
                anchors_xmax,\
                inside_mask

    def get_all_anchors_one_layer(self,
                                  anchor_heights_one_layer,
                                  anchor_widths_one_layer,
                                  num_anchors_per_location_one_layer,
                                  layer_shape,
                                  feat_stride,
                                  offset=0.5,
                                  vertical_offset=0.5,
                                  name=None):
        with tf.name_scope(name, 'get_all_anchors_one_layer'):
            feat_stride = tf.to_float(feat_stride)
            # feat_stride: 0d-tf.Tensor tf.float32

            # Anchor center coordinates on layer.
            x_on_layer, y_on_layer =\
                tf.meshgrid(tf.range(layer_shape[1]),  # w
                            tf.range(layer_shape[0]))  # h
            # x_on_layer: 2d-tf.Tensor tf.int32
            # y_on_layer: 2d-tf.Tensor tf.int32

            # x_on_layer = [
            #               [0, 1, 2, ..., 47],
            #               [0, 1, 2, ..., 47],
            #               ...
            #               [0, 1, 2, ..., 47],
            #              ]
            # x_on_layer.shape = (48, 48)

            # y_on_layer = [
            #               [0, 0, 0, ..., 0],
            #               [1, 1, 1, ..., 1],
            #               ...
            #               [47, 47, 47, ..., 47],
            #              ]
            # y_on_layer.shape = (48, 48)

            if isinstance(offset, list):
                tf.logging.info(
                    '{}: Using separate offset: height: {}, width: {}.'.format(
                        name,
                        offset[0],
                        offset[1]))
                offset_h = offset[0]
                offset_w = offset[1]
            else:
                offset_h = offset
                offset_w = offset

            # Anchor center coordinates on image.
            y_on_image = (tf.to_float(y_on_layer) + offset_h) * feat_stride
            x_on_image = (tf.to_float(x_on_layer) + offset_w) * feat_stride
            # y_on_image: 2d-tf.Tensor tf.float32
            # x_on_image: 2d-tf.Tensor tf.float32

            # x_on_image =\
            # [
            #  [0.5 x 384/48, 1.5 x 384/48, 2.5 x 384/48, ..., 47.5 x 384/48],
            #  [0.5 x 384/48, 1.5 x 384/48, 2.5 x 384/48, ..., 47.5 x 384/48],
            #  ...
            #  [0.5 x 384/48, 1.5 x 384/48, 2.5 x 384/48, ..., 47.5 x 384/48],
            # ]
            # x_on_image.shape = (48, 48)

            # y_on_image =\
            # [
            #  [0.5 x 384/48, 0.5 x 384/48, ..., 0.5 x 384/48],
            #  [1.5 x 384/48, 1.5 x 384/48, ..., 1.5 x 384/48],
            #  ...
            #  [47.5 x 384/48, 47.5 x 384/48, ..., 47.5 x 384/48],
            # ]
            # y_on_image.shape = (48, 48)

            y_vo_on_image =\
                (tf.to_float(y_on_layer) + offset_h + vertical_offset) *\
                feat_stride
            x_vo_on_image = x_on_image
            # y_vo_on_image: 2d-tf.Tensor tf.float32
            # x_vo_on_image: 2d-tf.Tensor tf.float32

            # y_vo_on_image =\
            # [
            #  [1 x 384/48, 1 x 384/48, 1 x 384/48, ..., 1 x 384/48],
            #  [2 x 384/48, 2 x 384/48, 2 x 384/48, ..., 2 x 384/48],
            #  ...
            #  [48 x 384/48, 48 x 384/48, 48 x 384/48, ..., 48 x 384/48],
            # ]
            # y_vo_on_image.shape = (48, 48)

            y_on_image = tf.stack([y_on_image,
                                  y_vo_on_image], axis=-1)
            x_on_image = tf.stack([x_on_image,
                                  x_vo_on_image], axis=-1)
            # y_on_image: 3d-tf.Tensor(FH, FW, 2) tf.float32
            # x_on_image: 3d-tf.Tensor(FH, FW, 2) tf.float32

            # Anchors one layer.
            anchors_ymin,\
                anchors_xmin,\
                anchors_ymax,\
                anchors_xmax =\
                self.center2point(tf.expand_dims(y_on_image, axis=-1),
                                  # 4d-tf.Tensor(FH, FW, 2, 1) tf.float32
                                  tf.expand_dims(x_on_image, axis=-1),
                                  # 4d-tf.Tensor(FH, FW, 2, 1) tf.float32
                                  anchor_heights_one_layer,
                                  # 1d-tf.constant tf.float32
                                  anchor_widths_one_layer
                                  # 1d-tf.constant tf.float32
                                  )
            # anchors_ymin: 4d-tf.Tensor(
            #       FH, FW, 2,
            #       num_anchors_per_location_one_layer) tf.float32

            # All locations=\
            #       feature layer locations + vertical offsetted locations.
            anchors_ymin = tf.reshape(anchors_ymin,
                                      [-1, num_anchors_per_location_one_layer])
            anchors_xmin = tf.reshape(anchors_xmin,
                                      [-1, num_anchors_per_location_one_layer])
            anchors_ymax = tf.reshape(anchors_ymax,
                                      [-1, num_anchors_per_location_one_layer])
            anchors_xmax = tf.reshape(anchors_xmax,
                                      [-1, num_anchors_per_location_one_layer])
            # anchors_ymin: 2d-tf.Tensor(
            #       FH * FW * 2,
            #       num_anchors_per_location_one_layer) tf.float32

            return anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax

    def count_num_anchors_per_layer(self,
                                    num_anchors_per_location_one_layer,
                                    layer_shape,
                                    name=None):
        with tf.name_scope(name, 'count_num_anchors_per_layer'):
            num_anchors_per_depth = layer_shape[0] * layer_shape[1] * 2
            num_anchors_per_layer =\
                num_anchors_per_depth * num_anchors_per_location_one_layer
            return num_anchors_per_depth, num_anchors_per_layer

    def encode_anchors(self,
                       labels,
                       bboxes,
                       quadrilaterals,
                       anchors_ymin,
                       anchors_xmin,
                       anchors_ymax,
                       anchors_xmax,
                       inside_mask,
                       debug=False):
        '''encode anchors with ground truth on the fly

        We generate prediction targets for all locations of the rpn feature
        map, so this routine is called when the final rpn feature map has been
        generated, so there is a performance bottleneck here but we have no
        idea to fix this because of we must perform multi-scale training.
        Maybe this needs to be placed on CPU, leave this problem to later

        Args:
            bboxes: [num_bboxes, 4] in [ymin, xmin, ymax, xmax] format
            anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax,
                inside_mask: generate by 'get_all_anchors_all_layers'

        Example of arguments:
            labels.get_shape():  (?,)
            bboxes.get_shape():  (?, 4)
            anchors_ymin.get_shape():  (8732,)
            anchors_xmin.get_shape():  (8732,)
            anchors_ymax.get_shape():  (8732,)
            anchors_xmax.get_shape():  (8732,)
            inside_mask.get_shape():  (8732,)
        '''
        with tf.name_scope('encode_anchors'):
            all_anchors = tf.stack([anchors_ymin,
                                    anchors_xmin,
                                    anchors_ymax,
                                    anchors_xmax], axis=-1)
            # e.g., all_anchors.get_shape():  (8732, 4)
            overlap_matrix =\
                iou_matrix(all_anchors, bboxes) *\
                tf.cast(tf.expand_dims(inside_mask, 1), tf.float32)
            matched_gt, gt_scores =\
                do_dual_max_match(overlap_matrix,
                                  self._ignore_threshold,
                                  self._positive_threshold)
            # get all positive matching positions
            matched_gt_mask = matched_gt > -1
            matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
            # temporarily move negatives and ignores to the first object's
            # class

            # e.g., matched_indices.get_shape():  (8732,)
            gt_labels = tf.gather(labels, matched_indices)
            # e.g., gt_labels.get_shape():  (8732,)
            # filter the invalid labels
            gt_labels = gt_labels * tf.to_int64(matched_gt_mask)
            # return negatives to its true class which is rezo (background)
            # set those ignored positions to -1
            # >=1: detection objects
            # =0: background including all negatives considered as background
            # -1: for ignores
            gt_labels = gt_labels + (-1 * tf.to_int64(matched_gt < -1))
            # return ignores to its class (id=-1). Actually we don't have this
            # class

            gt_ymin, gt_xmin, gt_ymax, gt_xmax =\
                tf.unstack(tf.gather(bboxes,
                                     matched_indices),
                           4,
                           axis=-1)

            # transform to center / size.
            gt_cy, gt_cx, gt_h, gt_w =\
                self.point2center(gt_ymin,
                                  gt_xmin,
                                  gt_ymax,
                                  gt_xmax)
            anchor_cy, anchor_cx, anchor_h, anchor_w =\
                self.point2center(anchors_ymin,
                                  anchors_xmin,
                                  anchors_ymax,
                                  anchors_xmax)
            # encode features.
            # the prior_scaling (in fact is 5 and 10) is use for balance the
            # regression loss of center and with(or height)
            gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[0]
            gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[1]
            gt_h = tf.log(gt_h / anchor_h) / self._prior_scaling[2]
            gt_w = tf.log(gt_w / anchor_w) / self._prior_scaling[3]
            # now gt_localizations is our regression object, but also maybe
            # chaos at those non-positive positions
            gt_y1, gt_x1, gt_y2, gt_x2,\
                gt_y3, gt_x3, gt_y4, gt_x4 =\
                tf.unstack(tf.gather(quadrilaterals,
                                     matched_indices),
                           8,
                           axis=-1)

            gt_y1 = (gt_y1 - anchors_ymin) / anchor_h / self._prior_scaling[0]
            gt_x1 = (gt_x1 - anchors_xmin) / anchor_w / self._prior_scaling[1]
            gt_y2 = (gt_y2 - anchors_ymin) / anchor_h / self._prior_scaling[0]
            gt_x2 = (gt_x2 - anchors_xmax) / anchor_w / self._prior_scaling[1]
            gt_y3 = (gt_y3 - anchors_ymax) / anchor_h / self._prior_scaling[0]
            gt_x3 = (gt_x3 - anchors_xmax) / anchor_w / self._prior_scaling[1]
            gt_y4 = (gt_y4 - anchors_ymax) / anchor_h / self._prior_scaling[0]
            gt_x4 = (gt_x4 - anchors_xmin) / anchor_w / self._prior_scaling[1]

            if debug:
                gt_targets =\
                    tf.stack([anchors_ymin,
                              anchors_xmin,
                              anchors_ymax,
                              anchors_xmax], axis=-1)
            else:
                gt_targets = tf.stack([gt_cy, gt_cx, gt_h, gt_w,
                                       gt_y1, gt_x1,
                                       gt_y2, gt_x2,
                                       gt_y3, gt_x3,
                                       gt_y4, gt_x4], axis=-1)

            # set all targets of non-positive positions to 0
            gt_targets =\
                tf.expand_dims(tf.to_float(matched_gt_mask), -1) * gt_targets
            # not use negatives and ignores coordinates

            # return:
            # - gt_scores: [score_0, score_1, ..., score_n_anchors-1] in which
            # + assume oij = overlap(anchor_i, object_j)
            # + then score_i =
            #   score_ia = max([oij for object_j in objects_i])
            #   if objects_i != []
            #   else:
            #   score_ib = max([oij for object_j in objects])
            #   objects_i = [object in objects
            #               if index(anchors,
            #                        max(overlap(object, anchors))) = i]
            # + score_i in [0, 1] = [negatives | ignores | positives] if
            # ignore_between is True
            # + negatives_range = [0, x)
            # + ignores_range = [x, y)
            # + positives_range = [y, 1]
            # + x = y = 0.5 (x = FLAGS.neg_threshold, y =
            # FLAGS.match_threshold) ==> ignores = None
            # - gt_labels = [label_0, ..., label_n_anchors-1]
            # + label_i = label(object_j) >= 0 (=0 for background and >=1 for
            # detection objects) such that overlap(anchor_i, object_j) =
            # score_ia if exists else score_ib in positives_range. Please note
            # that background objects doesn't normally come from labelling data
            # but negative anchors considered as background
            # + label_i = 0 for negatives (score_ib in negatives_range)
            # + label_i = -1 for ignores (score_ib in ignores_range)
            # - gt_targets = [12coordinates_0, ..., 12coordinates_n_anchors-1]
            # + 12coordinates_i = 0 if ignores or negatives
            return gt_targets, gt_labels, gt_scores

    def batch_decode_anchors(self,
                             pred_location,
                             anchors_ymin,
                             anchors_xmin,
                             anchors_ymax,
                             anchors_xmax):
        with tf.name_scope('decode_rpn',
                           values=[pred_location,
                                   anchors_ymin,
                                   anchors_xmin,
                                   anchors_ymax,
                                   anchors_xmax]):

            anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax =\
                tf.expand_dims(anchors_ymin, axis=0),\
                tf.expand_dims(anchors_xmin, axis=0),\
                tf.expand_dims(anchors_ymax, axis=0),\
                tf.expand_dims(anchors_xmax, axis=0)

            anchor_cy, anchor_cx, anchor_h, anchor_w =\
                self.point2center(anchors_ymin,
                                  anchors_xmin,
                                  anchors_ymax,
                                  anchors_xmax)

            # e.g., pred_location.get_shape():  (?, ?, 12)
            pred_cy = pred_location[:, :, 0] * self._prior_scaling[0] *\
                anchor_h + anchor_cy
            pred_cx = pred_location[:, :, 1] * self._prior_scaling[1] *\
                anchor_w + anchor_cx
            pred_h = tf.exp(pred_location[:, :, 2] * self._prior_scaling[2]) *\
                anchor_h
            pred_w = tf.exp(pred_location[:, :, 3] * self._prior_scaling[3]) *\
                anchor_w
            pred_y1 = pred_location[:, :, 4] * self._prior_scaling[0] *\
                anchor_h + anchors_ymin
            pred_x1 = pred_location[:, :, 5] * self._prior_scaling[1] *\
                anchor_w + anchors_xmin
            pred_y2 = pred_location[:, :, 6] * self._prior_scaling[0] *\
                anchor_h + anchors_ymin
            pred_x2 = pred_location[:, :, 7] * self._prior_scaling[1] *\
                anchor_w + anchors_xmax
            pred_y3 = pred_location[:, :, 8] * self._prior_scaling[0] *\
                anchor_h + anchors_ymax
            pred_x3 = pred_location[:, :, 9] * self._prior_scaling[1] *\
                anchor_w + anchors_xmax
            pred_y4 = pred_location[:, :, 10] * self._prior_scaling[0] *\
                anchor_h + anchors_ymax
            pred_x4 = pred_location[:, :, 11] * self._prior_scaling[1] *\
                anchor_w + anchors_xmin

            bboxes_pred =\
                tf.stack(self.center2point(pred_cy,
                                           pred_cx,
                                           pred_h,
                                           pred_w), axis=-1)
            quadrilaterals_pred =\
                tf.stack([pred_y1, pred_x1,
                          pred_y2, pred_x2,
                          pred_y3, pred_x3,
                          pred_y4, pred_x4], axis=-1)
            return bboxes_pred, quadrilaterals_pred
            # return tf.stack(
            #     list(self.center2point(pred_cy, pred_cx, pred_h, pred_w)) +
            #     [pred_y1, pred_x1, pred_y2, pred_x2,
            #      pred_y3, pred_x3, pred_y4, pred_x4],
            #     axis=-1)

    def decode_anchors(self,
                       pred_location,
                       anchors_ymin,
                       anchors_xmin,
                       anchors_ymax,
                       anchors_xmax):
        with tf.name_scope('decode_rpn',
                           values=[pred_location,
                                   anchors_ymin,
                                   anchors_xmin,
                                   anchors_ymax,
                                   anchors_xmax]):
            anchor_cy, anchor_cx, anchor_h, anchor_w =\
                self.point2center(anchors_ymin,
                                  anchors_xmin,
                                  anchors_ymax,
                                  anchors_xmax)

            pred_cy = pred_location[:, 0] * self._prior_scaling[0] *\
                anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] *\
                anchor_w + anchor_cx
            pred_h = tf.exp(pred_location[:, 2] * self._prior_scaling[2]) *\
                anchor_h
            pred_w = tf.exp(pred_location[:, 3] * self._prior_scaling[3]) *\
                anchor_w
            pred_y1 = pred_location[:, 4] * self._prior_scaling[0] *\
                anchor_h + anchors_ymin
            pred_x1 = pred_location[:, 5] * self._prior_scaling[1] *\
                anchor_w + anchors_xmin
            pred_y2 = pred_location[:, 6] * self._prior_scaling[0] *\
                anchor_h + anchors_ymin
            pred_x2 = pred_location[:, 7] * self._prior_scaling[1] *\
                anchor_w + anchors_xmax
            pred_y3 = pred_location[:, 8] * self._prior_scaling[0] *\
                anchor_h + anchors_ymax
            pred_x3 = pred_location[:, 9] * self._prior_scaling[1] *\
                anchor_w + anchors_xmax
            pred_y4 = pred_location[:, 10] * self._prior_scaling[0] *\
                anchor_h + anchors_ymax
            pred_x4 = pred_location[:, 11] * self._prior_scaling[1] *\
                anchor_w + anchors_xmin

            bboxes_pred =\
                tf.stack(self.center2point(pred_cy,
                                           pred_cx,
                                           pred_h,
                                           pred_w), axis=-1)
            quadrilaterals_pred =\
                tf.stack([pred_y1, pred_x1,
                          pred_y2, pred_x2,
                          pred_y3, pred_x3,
                          pred_y4, pred_x4], axis=-1)
            return bboxes_pred, quadrilaterals_pred


def iou_matrix(gt_bboxes, default_bboxes):
    with tf.name_scope('iou_matrix', values=[gt_bboxes, default_bboxes]):
        inter_vol = intersection(gt_bboxes, default_bboxes)
        # e.g., inter_vol.get_shape():  (8732, ?)
        # broadcast
        # e.g., gt_bboxes.get_shape():  (8732, 4)
        areas_gt = areas(gt_bboxes)
        # e.g., areas_gt.get_shape():  (8732, 1)
        union_vol = areas_gt +\
            tf.transpose(areas(default_bboxes), perm=[1, 0]) -\
            inter_vol

        # areas_gt = tf.Print(areas_gt, [areas_gt], summarize=100)
        return tf.where(tf.equal(union_vol, 0.0),
                        tf.zeros_like(inter_vol),
                        tf.truediv(inter_vol, union_vol))


def do_dual_max_match(overlap_matrix,
                      low_thres,  # _ignore_threshold
                      high_thres,  # _positive_threshold
                      ignore_between=True,
                      gt_max_first=True):
    '''do_dual_max_match, but using the transposed overlap matrix, this may be
    faster due to the cache friendly

    Args:
        overlap_matrix: num_anchors * num_gts
    '''
    with tf.name_scope('dual_max_match', values=[overlap_matrix]):
        # e.g.,
        # overlap_matrix =
        #                0     1    2    3    4    n_gts-1
        #       0        0.    0.   0.   0.   0.      0.
        #       1        0.5  0.45 0.8  0.8  0.7     0.6
        #       2        0.   0.35 0.45 0.2  0.45    0.1
        #       3        0.6  0.45 0.9  0.95 0.8     0.9
        #       4        0.   0.2  0.15 0.1  0.25    0.1
        #       5        0.6  0.1  0.2  0.4  0.5     0.
        #   n_anchors-1  0.   0.   0.   0.   0.      0.
        # low_thres = 0.4
        # high_thres = 0.5
        # first match from anchors' side
        anchors_to_gt = tf.argmax(overlap_matrix, axis=1)
        # anchors_to_gt = [0, 2, 2, 3, 4, 0, 0]
        # the matching degree
        match_values = tf.reduce_max(overlap_matrix, axis=1)
        # e.g., match_values.get_shape():  (8732,)
        # e.g., match_values.get_shape():  (8732, 1) if keepdims=True
        # match_values=[0., 0.8, 0.45, 0.95, 0.25, 0.6, 0.]

        # positive_mask = tf.greater(match_values, high_thres)
        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values,
                                              high_thres),
                                      tf.greater_equal(match_values,
                                                       low_thres))
        negative_mask = less_mask if ignore_between else between_mask
        ignore_mask = between_mask if ignore_between else less_mask
        # negative_mask = less_mask =
        # [True, False, False, False, True, False, True]
        # ignore_mask = between_mask =
        # [False, False, True, False, False, False, False]
        # comment following two lines
        # over_pos_mask = tf.greater(match_values, 0.7)
        # ignore_mask = tf.logical_or(ignore_mask, over_pos_mask)
        # fill all negative positions with -1, all ignore positions with -2
        match_indices = tf.where(negative_mask,
                                 -1 * tf.ones_like(anchors_to_gt),
                                 anchors_to_gt)
        match_indices = tf.where(ignore_mask,
                                 -2 * tf.ones_like(match_indices),
                                 match_indices)
        # e.g., match_indices.get_shape():  (8732,)
        # Each anchor matchs exactly to only one groundtruth object. The
        # luckily matched groundtruth object is the biggest one firstly
        # encountered in tf.argmax().
        # match_indices = [-1, 2, -2, 3, -1, 0, -1]

        anchors_to_gt_mask =\
            tf.one_hot(tf.clip_by_value(match_indices,
                                        -1,
                                        tf.cast(tf.shape(overlap_matrix)[1],
                                                tf.int64)),
                       tf.shape(overlap_matrix)[1],
                       on_value=1,  # present value
                       off_value=0,  # absence value
                       axis=1,
                       dtype=tf.int32)
        # Negtive values has no effect in tf.one_hot.
        # anchors_to_gt_mask =
        #           -1(-2)   0    1    2    3    4    5(=n_gts-1)
        # -1       [[  x     0,   0,   0,   0,   0,   0]       0
        #  2        [  x     0,   0,   1,   0,   0,   0]       1
        # -2        [  x     0,   0,   0,   0,   0,   0]       2
        #  3 =====> [  x     0,   0,   0,   1,   0,   0]       3
        # -1        [  x     0,   0,   0,   0,   0,   0]       4
        #  0        [  x     1,   0,   0,   0,   0,   0]       5
        # -1        [  x     0,   0,   0,   0,   0,   0]       6(=n_anchors-1)
        #            x means excluded by tf.one_hot
        # e.g., anchors_to_gt_mask.get_shape():  (8732, ?)

        # match from ground truth's side
        # gt_to_anchors = tf.argmax(overlap_matrix, axis=0)
        gt_to_anchors_overlap = tf.reduce_max(overlap_matrix,
                                              axis=0,
                                              keepdims=True)
        # e.g., gt_to_anchors_overlap.get_shape():  (1, ?) if keepdims=True
        # e.g., gt_to_anchors_overlap.get_shape():  (?,) if keepdims=False
        # overlap_matrix =
        #                0     1    2    3    4    n_gts-1
        #       0        0.    0.   0.   0.   0.      0.
        #       1        0.5  0.45 0.8  0.8  0.7     0.6
        #       2        0.   0.35 0.45 0.2  0.45    0.1
        #       3        0.6  0.45 0.9  0.95 0.8     0.9
        #       4        0.   0.2  0.15 0.1  0.25    0.1
        #       5        0.6  0.1  0.2  0.4  0.5     0.
        #   n_anchors-1  0.   0.   0.   0.   0.      0.
        # gt_to_anchors_overlap = [[3, 1, 3, 3, 3, 3]]

        # gt_to_anchors = tf.Print(gt_to_anchors,
        #                   [tf.equal(overlap_matrix, gt_to_anchors_overlap)],
        #                   message='gt_to_anchors_indices:', summarize=100)
        # the max match from ground truth's side has higher priority
        left_gt_to_anchors_mask =\
            tf.equal(overlap_matrix,
                     gt_to_anchors_overlap)
        # left_gt_to_anchors_mask =
        # [[False False False False False False]
        #  [False  True False False False False]
        #  [False False False False False False]
        #  [ True False  True  True  True  True]
        #  [False False False False False False]
        #  [False False False False False False]
        #  [False False False False False False]]
        # tf.one_hot(gt_to_anchors,
        #            tf.shape(overlap_matrix)[0],
        #            on_value=True,
        #            off_value=False,
        #            axis=0,
        #            dtype=tf.bool)
        # e.g., left_gt_to_anchors_mask.get_shape():  (8732, ?)
        if not gt_max_first:
            # the max match from anchors' side has higher priority
            # use match result from ground truth's side only when the the
            # matching degree from anchors' side is lower than position
            # threshold
            left_gt_to_anchors_mask =\
                tf.logical_and(tf.reduce_max(anchors_to_gt_mask,
                                             axis=0,
                                             keep_dims=True) < 1,
                               left_gt_to_anchors_mask)
        # can not use left_gt_to_anchors_mask here, because there are many
        # ground truthes match to one anchor, we should pick the highest one
        # even when we are merging matching from ground truth side
        left_gt_to_anchors_mask = tf.to_int64(left_gt_to_anchors_mask)
        # left_gt_to_anchors_mask =
        # [[ 0  0  0  0  0  0]
        #  [ 0  1  0  0  0  0]
        #  [ 0  0  0  0  0  0]
        #  [ 1  0  1  1  1  1]
        #  [ 0  0  0  0  0  0]
        #  [ 0  0  0  0  0  0]
        #  [ 0  0  0  0  0  0]]
        left_gt_to_anchors_scores =\
            overlap_matrix * tf.to_float(left_gt_to_anchors_mask)
        selected_scores =\
            tf.gather_nd(
                overlap_matrix,
                tf.stack([tf.range(tf.cast(tf.shape(overlap_matrix)[0],
                                           tf.int64)),
                          tf.where(tf.reduce_max(left_gt_to_anchors_mask,
                                                 axis=1) > 0,
                                   tf.argmax(left_gt_to_anchors_scores,
                                             axis=1),
                                   anchors_to_gt)],
                         axis=1))
        # tf.range = [0, 1, 2, ..., 6 (n_anchors-1)]
        # tf.reduce_max = [0, 1, 0, 1, 0, 0, 0]
        # left_gt_to_anchors_scores =
        #                0    1    2    3    4    n_gts-1
        #       0        0.   0.   0.   0.   0.      0.
        #       1        0.   0.45 0.   0.   0.      0.
        #       2        0.   0.   0.   0.   0.      0.
        #       3        0.6  0.   0.9  0.95 0.8     0.9
        #       4        0.   0.   0.   0.   0.      0.
        #       5        0.   0.   0.   0.   0.      0.
        #   n_anchors-1  0.   0.   0.   0.   0.      0.
        # tf.argmax = [0, 1, 0, 3, 0, 0, 0]
        # anchors_to_gt = [0, 2, 2, 3, 4, 0, 0]
        # tf.where = [0, 1, 2, 3, 4, 0, 0]
        # tf.stack = [[0, 0],
        #             [1, 1],
        #             [2, 2]
        #             [2, 2]
        #             [3, 3]
        #             [4, 4]
        #             [5, 0]
        #             [6, 0]]
        # tf.gather_nd = [0., 0.45, 0.45, 0.95, 0.25, 0.6, 0.]
        # overlap_matrix =
        #                0     1    2    3    4    n_gts-1
        #       0        0.    0.   0.   0.   0.      0.
        #       1        0.5  0.45 0.8  0.8  0.7     0.6
        #       2        0.   0.35 0.45 0.2  0.45    0.1
        #       3        0.6  0.45 0.9  0.95 0.8     0.9
        #       4        0.   0.2  0.15 0.1  0.25    0.1
        #       5        0.6  0.1  0.2  0.4  0.5     0.
        #   n_anchors-1  0.   0.   0.   0.   0.      0.

        # match_indices = [-1, 2, -2, 3, -1, 0, -1]
        # return:
        # - value1 = [-1, 1, -2, 3, -1, 0, -1]
        # - value2 = selected_scores = tf.gather_nd
        return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=1) > 0,
                        tf.argmax(left_gt_to_anchors_scores, axis=1),
                        match_indices), selected_scores


def intersection(gt_bboxes, default_bboxes):
    with tf.name_scope('bboxes_intersection',
                       values=[gt_bboxes, default_bboxes]):
        # num_anchors x 1
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        # 1 x num_anchors
        gt_ymin, gt_xmin, gt_ymax, gt_xmax =\
            [tf.transpose(b, perm=[1, 0])
             for b in tf.split(default_bboxes, 4, axis=1)]
        # broadcast here to generate the full matrix
        int_ymin = tf.maximum(ymin, gt_ymin)
        int_xmin = tf.maximum(xmin, gt_xmin)
        int_ymax = tf.minimum(ymax, gt_ymax)
        int_xmax = tf.minimum(xmax, gt_xmax)
        h = tf.maximum(int_ymax - int_ymin + 1., 0.)
        w = tf.maximum(int_xmax - int_xmin + 1., 0.)
        return h * w


def areas(gt_bboxes):
    with tf.name_scope('bboxes_areas', values=[gt_bboxes]):
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        return (xmax - xmin + 1.) * (ymax - ymin + 1.)
