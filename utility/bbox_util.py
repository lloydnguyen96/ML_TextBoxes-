from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def select_bboxes(scores_pred,
                  bboxes_pred,
                  quadrilaterals_pred,
                  num_classes,
                  select_threshold,
                  name=None):
    selected_bboxes = {}
    selected_quadrilaterals = {}
    selected_scores = {}
    with tf.name_scope(name, 'select_bboxes',
                       values=[scores_pred, bboxes_pred, quadrilaterals_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]

            select_mask = class_scores > select_threshold
            select_mask = tf.to_float(select_mask)

            selected_bboxes[class_ind] =\
                tf.multiply(bboxes_pred,
                            tf.expand_dims(select_mask,
                                           axis=-1))

            selected_quadrilaterals[class_ind] =\
                tf.multiply(quadrilaterals_pred,
                            tf.expand_dims(select_mask,
                                           axis=-1))

            selected_scores[class_ind] =\
                tf.multiply(class_scores,
                            select_mask)

    return selected_bboxes, selected_quadrilaterals, selected_scores


def clip_bboxes(ymin,
                xmin,
                ymax,
                xmax,
                height,
                width,
                name=None):
    with tf.name_scope(name,
                       'clip_bboxes',
                       values=[ymin, xmin, ymax, xmax]):
        ymin = tf.maximum(ymin, 0.)
        xmin = tf.maximum(xmin, 0.)
        ymax = tf.minimum(ymax, tf.to_float(height) - 1.)
        xmax = tf.minimum(xmax, tf.to_float(width) - 1.)

        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)

        return ymin, xmin, ymax, xmax


def clip_quadrilaterals(y1, x1, y2, x2, y3, x3, y4, x4,
                        height,
                        width,
                        name=None):
    with tf.name_scope(name, 'clip_quadrilaterals',
                       values=[y1, x1, y2, x2, y3, x3, y4, x4]):
        y1 = tf.maximum(y1, 0.)
        x1 = tf.maximum(x1, 0.)
        y2 = tf.maximum(y2, 0.)
        x2 = tf.minimum(x2, tf.to_float(width) - 1.)
        y3 = tf.minimum(y3, tf.to_float(height) - 1.)
        x3 = tf.minimum(x3, tf.to_float(width) - 1.)
        y4 = tf.minimum(y4, tf.to_float(height) - 1.)
        x4 = tf.maximum(x4, 0.)

        y1 = tf.minimum(y1, y4)
        y4 = tf.maximum(y1, y4)
        y2 = tf.minimum(y2, y3)
        y3 = tf.maximum(y2, y3)

        x1 = tf.minimum(x1, x2)
        x2 = tf.maximum(x1, x2)
        x4 = tf.minimum(x4, x3)
        x3 = tf.maximum(x4, x3)

        return y1, x1, y2, x2, y3, x3, y4, x4


def filter_bboxes_and_quadrilaterals(
        scores_pred,
        ymin, xmin, ymax, xmax,
        y1, x1, y2, x2, y3, x3, y4, x4,
        min_size, name=None):
    with tf.name_scope(name, 'filter_bboxes_and_quadrilaterals',
                       values=[scores_pred,
                               ymin, xmin, ymax, xmax,
                               y1, x1, y2, x2, y3, x3, y4, x4]):
        width = xmax - xmin + 1.
        height = ymax - ymin + 1.

        filter_mask =\
            tf.logical_and(width > min_size + 1., height > min_size + 1.)

        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(scores_pred, filter_mask),\
            tf.multiply(ymin, filter_mask),\
            tf.multiply(xmin, filter_mask),\
            tf.multiply(ymax, filter_mask),\
            tf.multiply(xmax, filter_mask),\
            tf.multiply(y1, filter_mask),\
            tf.multiply(x1, filter_mask),\
            tf.multiply(y2, filter_mask),\
            tf.multiply(x2, filter_mask),\
            tf.multiply(y3, filter_mask),\
            tf.multiply(x3, filter_mask),\
            tf.multiply(y4, filter_mask),\
            tf.multiply(x4, filter_mask)


def sort_bboxes_and_quadrilaterals(
        scores_pred,
        ymin, xmin, ymax, xmax,
        y1, x1, y2, x2, y3, x3, y4, x4,
        keep_topk,
        name=None):
    with tf.name_scope(name,
                       'sort_bboxes_and_quadrilaterals',
                       values=[scores_pred,
                               ymin, xmin, ymax, xmax,
                               y1, x1, y2, x2, y3, x3, y4, x4]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes =\
            tf.nn.top_k(scores_pred,
                        k=tf.minimum(keep_topk,
                                     cur_bboxes),
                        sorted=True)

        ymin = tf.gather(ymin, idxes)
        xmin = tf.gather(xmin, idxes)
        ymax = tf.gather(ymax, idxes)
        xmax = tf.gather(xmax, idxes)

        y1 = tf.gather(y1, idxes)
        x1 = tf.gather(x1, idxes)
        y2 = tf.gather(y2, idxes)
        x2 = tf.gather(x2, idxes)
        y3 = tf.gather(y3, idxes)
        x3 = tf.gather(x3, idxes)
        y4 = tf.gather(y4, idxes)
        x4 = tf.gather(x4, idxes)

        paddings =\
            tf.expand_dims(
                tf.stack([0,
                          tf.maximum(keep_topk-cur_bboxes, 0)],
                         axis=0),
                axis=0)

        return tf.pad(scores, paddings, "CONSTANT"),\
            tf.pad(ymin, paddings, "CONSTANT"),\
            tf.pad(xmin, paddings, "CONSTANT"),\
            tf.pad(ymax, paddings, "CONSTANT"),\
            tf.pad(xmax, paddings, "CONSTANT"),\
            tf.pad(y1, paddings, "CONSTANT"),\
            tf.pad(x1, paddings, "CONSTANT"),\
            tf.pad(y2, paddings, "CONSTANT"),\
            tf.pad(x2, paddings, "CONSTANT"),\
            tf.pad(y3, paddings, "CONSTANT"),\
            tf.pad(x3, paddings, "CONSTANT"),\
            tf.pad(y4, paddings, "CONSTANT"),\
            tf.pad(x4, paddings, "CONSTANT")


def nms_bboxes_with_padding(scores_pred,
                            bboxes_pred,
                            quadrilaterals_pred,
                            nms_topk,
                            nms_threshold,
                            name=None):
    with tf.name_scope(name, 'nms_bboxes_with_padding',
                       values=[scores_pred, bboxes_pred, quadrilaterals_pred]):
        idxes =\
            tf.image.non_max_suppression(bboxes_pred,
                                         scores_pred,
                                         nms_topk,
                                         nms_threshold)

        scores = tf.gather(scores_pred, idxes)
        bboxes = tf.gather(bboxes_pred, idxes)
        quadrilaterals = tf.gather(quadrilaterals_pred, idxes)

        nms_bboxes = tf.shape(idxes)[0]
        scores_paddings =\
            tf.expand_dims(tf.stack([0,
                                     tf.maximum(nms_topk - nms_bboxes, 0)],
                                    axis=0),
                           axis=0)
        bboxes_paddings =\
            tf.stack([[0, 0],
                      [tf.maximum(nms_topk - nms_bboxes, 0),
                       0]],
                     axis=1)
        quadrilaterals_paddings =\
            tf.stack([[0, 0],
                      [tf.maximum(nms_topk - nms_bboxes, 0),
                       0]],
                     axis=1)

        return tf.pad(scores, scores_paddings, "CONSTANT"),\
            tf.pad(bboxes, bboxes_paddings, "CONSTANT"),\
            tf.pad(quadrilaterals, quadrilaterals_paddings, "CONSTANT")


def parse_by_class(image_shape,
                   cls_pred,
                   bboxes_pred,
                   quadrilaterals_pred,
                   num_classes,
                   select_threshold,
                   min_size,
                   keep_topk,
                   nms_topk,
                   nms_threshold):
    with tf.name_scope('parse_by_class',
                       values=[cls_pred, bboxes_pred, quadrilaterals_pred]):
        # 1: num_classes & select_threshold
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes,\
            selected_quadrilaterals,\
            selected_scores =\
            select_bboxes(scores_pred,
                          bboxes_pred,
                          quadrilaterals_pred,
                          num_classes,
                          select_threshold)
        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax =\
                tf.unstack(selected_bboxes[class_ind],
                           4,
                           axis=-1)
            y1, x1, y2, x2, y3, x3, y4, x4 =\
                tf.unstack(selected_quadrilaterals[class_ind],
                           8,
                           axis=-1)
            ymin, xmin, ymax, xmax =\
                clip_bboxes(ymin,
                            xmin,
                            ymax,
                            xmax,
                            image_shape[0],
                            image_shape[1],
                            'clip_bboxes_{}'.format(class_ind))

            y1, x1, y2, x2, y3, x3, y4, x4 =\
                clip_quadrilaterals(
                    y1, x1, y2, x2, y3, x3, y4, x4,
                    image_shape[0],
                    image_shape[1],
                    'clip_quadrilaterals_{}'.format(class_ind))

            # 2: min_size
            selected_scores[class_ind],\
                ymin, xmin, ymax, xmax,\
                y1, x1, y2, x2, y3, x3, y4, x4 =\
                filter_bboxes_and_quadrilaterals(
                    selected_scores[class_ind],
                    ymin, xmin, ymax, xmax,
                    y1, x1, y2, x2, y3, x3, y4, x4,
                    min_size,
                    'filter_bboxes_and_quadrilaterals_{}'.format(class_ind))

            # 3: keep_topk
            selected_scores[class_ind],\
                ymin, xmin, ymax, xmax,\
                y1, x1, y2, x2, y3, x3, y4, x4 =\
                sort_bboxes_and_quadrilaterals(
                    selected_scores[class_ind],
                    ymin, xmin, ymax, xmax,
                    y1, x1, y2, x2, y3, x3, y4, x4,
                    keep_topk,
                    'sort_bboxes_{}'.format(class_ind))

            selected_bboxes[class_ind] =\
                tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            selected_quadrilaterals[class_ind] =\
                tf.stack([y1, x1, y2, x2, y3, x3, y4, x4], axis=-1)

            # 4: nms_topk & nms_threshold
            selected_scores[class_ind],\
                selected_bboxes[class_ind],\
                selected_quadrilaterals[class_ind] =\
                nms_bboxes_with_padding(
                    selected_scores[class_ind],
                    selected_bboxes[class_ind],
                    selected_quadrilaterals[class_ind],
                    nms_topk,
                    nms_threshold,
                    'nms_bboxes_{}'.format(class_ind))

        return selected_bboxes, selected_quadrilaterals, selected_scores
