from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import re
import cv2

import tensorflow as tf
# from matplotlib.pyplot import imread
import numpy as np

from net import textboxes_plusplus_net
from config import textboxes_plusplus_config as config

from preprocessing import textboxes_plusplus_preprocessing
from utility import anchor_manipulator
from utility import bbox_util
import time

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes',
    config.NUM_CLASSES,
    'Number of classes to use in the dataset')
# model related configuration
tf.app.flags.DEFINE_integer(
    'image_size',
    config.TRAIN_IMAGE_SIZE,
    'The size of the input image for the model to use')
tf.app.flags.DEFINE_string(
    'data_format',
    'channels_last',
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold',
    0.5,
    'Class-specific confidence score threshold for selecting a box')
tf.app.flags.DEFINE_float(
    'min_size',
    4.,
    'The min size of bboxes to keep')
tf.app.flags.DEFINE_float(
    'nms_threshold',
    0.45,
    'Matching threshold in NMS algorithm')
tf.app.flags.DEFINE_integer(
    'nms_topk',
    20,
    'Number of total objects to keep after NMS')
tf.app.flags.DEFINE_integer(
    'keep_topk',
    200,
    'Number of total objects to keep for each image before nms')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    './logs',
    'The path to a checkpoint from which to fine-tune')
tf.app.flags.DEFINE_string(
    'model_scope',
    'textboxes_plusplus',
    'Model scope name used to replace the name_scope in checkpoint')
tf.app.flags.DEFINE_string(
    'input_image_root',
    '/home/loinguyenvan/Projects/OneDriveHUST/Datasets/RCTW',
    'The path to a input_image_root from which to detect')
tf.app.flags.DEFINE_string(
    'input_image_stem_patterns',
    'icdar2017rctw_test/*.jpg',
    'comma-separated list of stem patterns of input images')
tf.app.flags.DEFINE_string(
    'output_directory',
    'evaluation/rctw_task1_evaluation/'
    'textboxes_plusplus_trained_rctw_p3_train',
    'The path to a output_directory from which to write')

FLAGS = tf.app.flags.FLAGS


def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    print('checkpoint_path', checkpoint_path)
    return checkpoint_path


def main(_):
    with tf.Graph().as_default():
        def split_image_into_overlapped_images(image, n, r):
            """TODO: Docstring for split_image_into_overlapped_images.

            :image: TODO
            :n: TODO
            :r: TODO
            :returns: TODO

            """
            IH, IW = tf.shape(image)[0], tf.shape(image)[1]
            ny, nx = n
            ry, rx = r
            SH = tf.cast(
                tf.floordiv(tf.cast(IH, tf.float32), (ny - ny * ry + ry)),
                tf.int32
            )
            SW = tf.cast(
                tf.floordiv(tf.cast(IW, tf.float32), (nx - nx * rx + rx)),
                tf.int32
            )
            OH = tf.cast(ry * tf.cast(SH, tf.float32), tf.int32)
            OW = tf.cast(rx * tf.cast(SW, tf.float32), tf.int32)
            images = []
            os = []
            for i in range(ny):
                oy = i * (SH - OH)
                for j in range(nx):
                    ox = j * (SW - OW)
                    os.append([oy, ox])
                    images.append(
                        image[oy:oy+SH, ox:ox+SW]
                    )
            return [[image, tf.shape(image), o]
                    for image, o in zip(images, os)]

        output_shape = [FLAGS.image_size] * 2

        input_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # nr1 = [(2, 0.7), (4, 0.6), (8, 0.5)]
        # nr2 = [(4, 0.4)]  # no1
        # nr3 = [(4, 0.2)]
        # nr4 = [(4, 0.3)]
        # nr5 = [(4, 0.6)]
        # nr6 = [(4, 0.5)]
        # nr7 = [(8, 0.2)]
        # nr8 = [(8, 0.8)]
        # nr9 = [(8, 0.4)]  # no1
        # nr10 = [(2, 0.8)]
        # nr11 = [(2, 0.2)]
        # nr12 = [(2, 0.4)]
        # nr13 = [(2, 0.6)]  # no1
        # nr14 = [(2, 0.5)]
        # nr15 = [(2, 0.6), (4, 0.4)]  # select_threshold = 0.5
        nr16 = [(2, 0.6), (4, 0.4)]  # select_threshold = 0.95
        images, shapes, os =\
            zip(*([[image, shape, o]
                   for n, r in nr16
                   for image, shape, o in split_image_into_overlapped_images(
                       input_image,
                       (n, n),
                       (r, r))] + [[input_image,
                                    tf.shape(input_image), [0, 0]]]))
        # images = [images[0], images[1]]
        # shapes = [shapes[0], shapes[1]]
        # os = [os[0], os[1]]

        oys, oxs = zip(*os)
        shapes = tf.stack(shapes)
        oys = tf.stack(oys)
        oxs = tf.stack(oxs)
        oys = tf.expand_dims(oys, -1)
        oxs = tf.expand_dims(oxs, -1)

        features = []
        for image in images:
            features.append(
                textboxes_plusplus_preprocessing.preprocess_for_eval(
                    image,
                    None, None,
                    output_shape,
                    data_format=FLAGS.data_format,
                    output_rgb=False)
            )
        features = tf.stack(features, axis=0)
        output_shape =\
            tf.expand_dims(
                tf.constant(output_shape,
                            dtype=tf.int32),
                axis=0)  # (1, 2)

        with tf.variable_scope(FLAGS.model_scope,
                               default_name=None,
                               values=[features],
                               reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                anchor_processor =\
                    anchor_manipulator.AnchorProcessor(
                        positive_threshold=None,
                        ignore_threshold=None,
                        prior_scaling=config.PRIOR_SCALING)

                anchor_heights_all_layers,\
                    anchor_widths_all_layers,\
                    num_anchors_per_location_all_layers =\
                    anchor_processor.get_anchors_size_all_layers(
                        config.ALL_ANCHOR_SCALES,
                        config.ALL_EXTRA_SCALES,
                        config.ALL_ANCHOR_RATIOS,
                        config.NUM_FEATURE_LAYERS)
                # anchor_heights_all_layers: [1d-tf.constant tf.float32,
                #                           1d-tf.constant tf.float32,
                #                           ...]
                # anchor_widths_all_layers: [1d-tf.constant tf.float32,
                #                           1d-tf.constant tf.float32,
                #                           ...]
                # num_anchors_per_location_all_layers:
                #   [Python int, Python int, ...]

                anchors_ymin,\
                    anchors_xmin,\
                    anchors_ymax,\
                    anchors_xmax, _ =\
                    anchor_processor.get_all_anchors_all_layers(
                        tf.squeeze(output_shape, axis=0),
                        anchor_heights_all_layers,
                        anchor_widths_all_layers,
                        num_anchors_per_location_all_layers,
                        config.ANCHOR_OFFSETS,
                        config.VERTICAL_OFFSETS,
                        config.ALL_LAYER_SHAPES,
                        config.ALL_LAYER_STRIDES,
                        [0.] * config.NUM_FEATURE_LAYERS,
                        [False] * config.NUM_FEATURE_LAYERS)
                # anchors_ymin: 1d-tf.Tensor(num_anchors_all_layers) tf.float32

                backbone =\
                    textboxes_plusplus_net.VGG16Backbone(FLAGS.data_format)
                feature_layers = backbone.forward(features, training=False)
                # shape = (num_feature_layers,
                #          BS,
                #          FH,
                #          FW,
                #          feature_depth)

                location_predictions, class_predictions =\
                    textboxes_plusplus_net.multibox_head(
                        feature_layers,
                        FLAGS.num_classes,
                        config.NUM_OFFSETS,
                        num_anchors_per_location_all_layers,
                        data_format=FLAGS.data_format)
                # shape = (num_feature_layers,
                #          bs,
                #          fh,
                #          fw,
                #          num_anchors_per_loc * 2 * num_offsets)

                if FLAGS.data_format == 'channels_first':
                    class_predictions =\
                        [tf.transpose(pred,
                                      [0, 2, 3, 1])
                         for pred in class_predictions]
                    location_predictions =\
                        [tf.transpose(pred,
                                      [0, 2, 3, 1])
                         for pred in location_predictions]
                class_predictions =\
                    [tf.reshape(pred,
                                [len(images), -1, FLAGS.num_classes])
                     for pred in class_predictions]
                location_predictions =\
                    [tf.reshape(pred, [len(images), -1, config.NUM_OFFSETS])
                     for pred in location_predictions]
                # shape = (num_feature_layers,
                #          bs,
                #          fh * fw * num_anchors_per_loc * 2,
                #          num_offsets)

                class_predictions = tf.concat(class_predictions, axis=1)
                location_predictions = tf.concat(location_predictions, axis=1)

                # total_parameters = 0
                # for variable in tf.trainable_variables():
                #     # shape is an array of tf.Dimension
                #     shape = variable.get_shape()
                #     print(shape)
                #     print(len(shape))
                #     variable_parameters = 1
                #     for dim in shape:
                #         print(dim)
                #         variable_parameters *= dim.value
                #     print(variable_parameters)
                #     total_parameters += variable_parameters
                # print(total_parameters)

        with tf.device('/cpu:0'):
            bboxes_pred, quadrilaterals_pred =\
                anchor_processor.batch_decode_anchors(
                    location_predictions,
                    anchors_ymin,
                    anchors_xmin,
                    anchors_ymax,
                    anchors_xmax)

            bboxes_ymin =\
                tf.cast(bboxes_pred[:, :, 0] * tf.expand_dims(tf.cast(
                    tf.truediv(shapes[:, 0],
                               output_shape[0, 0]),
                    tf.float32
                ), -1), tf.int32) + oys
            bboxes_xmin =\
                tf.cast(bboxes_pred[:, :, 1] * tf.expand_dims(tf.cast(
                    tf.truediv(shapes[:, 1],
                               output_shape[0, 1]),
                    tf.float32
                ), -1), tf.int32) + oxs
            bboxes_ymax =\
                tf.cast(bboxes_pred[:, :, 2] * tf.expand_dims(tf.cast(
                    tf.truediv(shapes[:, 0],
                               output_shape[0, 0]),
                    tf.float32), -1), tf.int32) + oys
            bboxes_xmax =\
                tf.cast(bboxes_pred[:, :, 3] * tf.expand_dims(tf.cast(
                    tf.truediv(shapes[:, 1],
                               output_shape[0, 1]),
                    tf.float32), -1), tf.int32) + oxs
            bboxes_pred =\
                tf.reshape(
                    tf.stack([bboxes_ymin, bboxes_xmin,
                              bboxes_ymax, bboxes_xmax], -1),
                    shape=[-1, 4])
            quadrilaterals_y1 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 0] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 0],
                                           output_shape[0, 0]),
                                tf.float32), -1), tf.int32) + oys
            quadrilaterals_x1 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 1] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 1],
                                           output_shape[0, 1]),
                                tf.float32), -1), tf.int32) + oxs
            quadrilaterals_y2 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 2] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 0],
                                           output_shape[0, 0]),
                                tf.float32), -1), tf.int32) + oys
            quadrilaterals_x2 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 3] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 1],
                                           output_shape[0, 1]),
                                tf.float32), -1), tf.int32) + oxs
            quadrilaterals_y3 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 4] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 0],
                                           output_shape[0, 0]),
                                tf.float32), -1), tf.int32) + oys
            quadrilaterals_x3 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 5] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 1],
                                           output_shape[0, 1]),
                                tf.float32), -1), tf.int32) + oxs
            quadrilaterals_y4 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 6] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 0],
                                           output_shape[0, 0]),
                                tf.float32), -1), tf.int32) + oys
            quadrilaterals_x4 =\
                tf.cast(
                    quadrilaterals_pred[:, :, 7] * tf.expand_dims(
                        tf.cast(tf.truediv(shapes[:, 1],
                                           output_shape[0, 1]),
                                tf.float32), -1), tf.int32) + oxs
            quadrilaterals_pred =\
                tf.reshape(
                    tf.stack([quadrilaterals_y1,
                              quadrilaterals_x1,
                              quadrilaterals_y2,
                              quadrilaterals_x2,
                              quadrilaterals_y3,
                              quadrilaterals_x3,
                              quadrilaterals_y4,
                              quadrilaterals_x4], -1),
                    shape=[-1, 8])
            class_predictions = tf.reshape(class_predictions,
                                           shape=[-1, FLAGS.num_classes])
            bboxes_pred = tf.cast(bboxes_pred, tf.float32)
            quadrilaterals_pred = tf.cast(quadrilaterals_pred, tf.float32)

            selected_bboxes,\
                selected_quadrilaterals,\
                selected_scores =\
                bbox_util.parse_by_class(
                    tf.shape(input_image)[:2],
                    class_predictions,
                    bboxes_pred,
                    quadrilaterals_pred,
                    FLAGS.num_classes,
                    FLAGS.select_threshold,
                    FLAGS.min_size,
                    FLAGS.keep_topk,
                    FLAGS.nms_topk,
                    FLAGS.nms_threshold)

            labels_list = []
            scores_list = []
            bboxes_list = []
            quadrilaterals_list = []
            for k, v in selected_scores.items():
                labels_list.append(tf.ones_like(v, tf.int32) * k)
                scores_list.append(v)
                bboxes_list.append(selected_bboxes[k])
                quadrilaterals_list.append(selected_quadrilaterals[k])
            all_labels = tf.concat(labels_list, axis=0)
            all_scores = tf.concat(scores_list, axis=0)
            all_bboxes = tf.concat(bboxes_list, axis=0)
            all_quadrilaterals = tf.concat(quadrilaterals_list, axis=0)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, get_checkpoint())

            image_paths =\
                sorted(
                    [path
                     for pattern in FLAGS.input_image_stem_patterns.split(',')
                     for path in Path(FLAGS.input_image_root).glob(pattern)],
                    key=lambda e: int(re.findall(r'(?<=_)\d+(?=.)',
                                                 e.name)[0]))
            for i, image_path in enumerate(image_paths):
                # image = imread(str(image_path))
                image =\
                    cv2.imread(
                        str(image_path),
                        cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR
                    )[:, :, ::-1]
                start_time = time.time()

                labels_,\
                    scores_,\
                    bboxes_,\
                    quadrilaterals_ =\
                    sess.run([all_labels,
                              all_scores,
                              all_bboxes,
                              all_quadrilaterals,
                              ],
                             feed_dict={input_image: image})

                elapsed_time = time.time() - start_time
                print('{}: elapsed_time = {}'.format(i + 1, elapsed_time))
                annotation_file_name =\
                    'task1_' + image_path.name.replace('.jpg', '.txt')
                with open(Path(FLAGS.output_directory).joinpath(
                        annotation_file_name), 'w') as f:
                    num_predicted_text_lines = np.shape(quadrilaterals_)[0]
                    for i in range(num_predicted_text_lines):
                        y1, x1, y2, x2,\
                            y3, x3, y4, x4 =\
                            [int(e) for e in quadrilaterals_[i, :]]
                        score = float(scores_[i])
                        if (y1 == 0 and x1 == 0 and
                                y2 == 0 and x2 == 0 and
                                y3 == 0 and x3 == 0 and
                                y4 == 0 and x4 == 0 and
                                score == 0.0):
                            continue
                        f.write('{},{},{},{},{},{},{},{},{}\n'.format(
                            x1, y1, x2, y2,
                            x3, y3, x4, y4, score))


# def main(_):
#     with tf.Graph().as_default():
#         output_shape = [FLAGS.image_size] * 2

#         input_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
#         input_shape = tf.placeholder(tf.int32, shape=(2,))

#         features =\
#             textboxes_plusplus_preprocessing.preprocess_for_eval(
#                 input_image,
#                 None, None,
#                 output_shape,
#                 data_format=FLAGS.data_format,
#                 output_rgb=False)
#         features = tf.expand_dims(features, axis=0)  # (1, IH, IW, 3)
#         output_shape =\
#             tf.expand_dims(
#                 tf.constant(output_shape,
#                             dtype=tf.int32),
#                 axis=0)  # (1, 2)

#         with tf.variable_scope(FLAGS.model_scope,
#                                default_name=None,
#                                values=[features],
#                                reuse=tf.AUTO_REUSE):
#             with tf.device('/cpu:0'):
#                 anchor_processor =\
#                     anchor_manipulator.AnchorProcessor(
#                         positive_threshold=None,
#                         ignore_threshold=None,
#                         prior_scaling=config.PRIOR_SCALING)

#                 anchor_heights_all_layers,\
#                     anchor_widths_all_layers,\
#                     num_anchors_per_location_all_layers =\
#                     anchor_processor.get_anchors_size_all_layers(
#                         config.ALL_ANCHOR_SCALES,
#                         config.ALL_EXTRA_SCALES,
#                         config.ALL_ANCHOR_RATIOS,
#                         config.NUM_FEATURE_LAYERS)
#                 # anchor_heights_all_layers: [1d-tf.constant tf.float32,
#                 #                           1d-tf.constant tf.float32,
#                 #                           ...]
#                 # anchor_widths_all_layers: [1d-tf.constant tf.float32,
#                 #                           1d-tf.constant tf.float32,
#                 #                           ...]
#                 # num_anchors_per_location_all_layers:
#                 #   [Python int, Python int, ...]

#                 anchors_ymin,\
#                     anchors_xmin,\
#                     anchors_ymax,\
#                     anchors_xmax, _ =\
#                     anchor_processor.get_all_anchors_all_layers(
#                         tf.squeeze(output_shape, axis=0),
#                         anchor_heights_all_layers,
#                         anchor_widths_all_layers,
#                         num_anchors_per_location_all_layers,
#                         config.ANCHOR_OFFSETS,
#                         config.VERTICAL_OFFSETS,
#                         config.ALL_LAYER_SHAPES,
#                         config.ALL_LAYER_STRIDES,
#                         [0.] * config.NUM_FEATURE_LAYERS,
#                         [False] * config.NUM_FEATURE_LAYERS)
#                 # anchors_ymin: 1d-tf.Tensor(num_anchors_all_layers) tf.float32

#                 backbone =\
#                     textboxes_plusplus_net.VGG16Backbone(FLAGS.data_format)
#                 feature_layers = backbone.forward(features, training=False)
#                 # shape = (num_feature_layers,
#                 #          BS,
#                 #          FH,
#                 #          FW,
#                 #          feature_depth)

#                 location_predictions, class_predictions =\
#                     textboxes_plusplus_net.multibox_head(
#                         feature_layers,
#                         FLAGS.num_classes,
#                         config.NUM_OFFSETS,
#                         num_anchors_per_location_all_layers,
#                         data_format=FLAGS.data_format)
#                 # shape = (num_feature_layers,
#                 #          bs,
#                 #          fh,
#                 #          fw,
#                 #          num_anchors_per_loc * 2 * num_offsets)

#                 if FLAGS.data_format == 'channels_first':
#                     class_predictions =\
#                         [tf.transpose(pred,
#                                       [0, 2, 3, 1])
#                          for pred in class_predictions]
#                     location_predictions =\
#                         [tf.transpose(pred,
#                                       [0, 2, 3, 1])
#                          for pred in location_predictions]
#                 class_predictions =\
#                     [tf.reshape(pred,
#                                 [-1, FLAGS.num_classes])
#                      for pred in class_predictions]
#                 location_predictions =\
#                     [tf.reshape(pred, [-1, config.NUM_OFFSETS])
#                      for pred in location_predictions]
#                 # shape = (num_feature_layers,
#                 #          bs * fh * fw * num_anchors_per_loc * 2,
#                 #          num_offsets)

#                 class_predictions = tf.concat(class_predictions, axis=0)
#                 location_predictions = tf.concat(location_predictions, axis=0)

#                 # total_parameters = 0
#                 # for variable in tf.trainable_variables():
#                 #     # shape is an array of tf.Dimension
#                 #     shape = variable.get_shape()
#                 #     print(shape)
#                 #     print(len(shape))
#                 #     variable_parameters = 1
#                 #     for dim in shape:
#                 #         print(dim)
#                 #         variable_parameters *= dim.value
#                 #     print(variable_parameters)
#                 #     total_parameters += variable_parameters
#                 # print(total_parameters)

#         with tf.device('/cpu:0'):
#             bboxes_pred, quadrilaterals_pred =\
#                 anchor_processor.decode_anchors(
#                     location_predictions,
#                     anchors_ymin,
#                     anchors_xmin,
#                     anchors_ymax,
#                     anchors_xmax)
#             selected_bboxes,\
#                 selected_quadrilaterals,\
#                 selected_scores =\
#                 bbox_util.parse_by_class(
#                     tf.squeeze(output_shape, axis=0),
#                     class_predictions,
#                     bboxes_pred,
#                     quadrilaterals_pred,
#                     FLAGS.num_classes,
#                     FLAGS.select_threshold,
#                     FLAGS.min_size,
#                     FLAGS.keep_topk,
#                     FLAGS.nms_topk,
#                     FLAGS.nms_threshold)

#             labels_list = []
#             scores_list = []
#             bboxes_list = []
#             quadrilaterals_list = []
#             for k, v in selected_scores.items():
#                 labels_list.append(tf.ones_like(v, tf.int32) * k)
#                 scores_list.append(v)
#                 bboxes_list.append(selected_bboxes[k])
#                 quadrilaterals_list.append(selected_quadrilaterals[k])
#             all_labels = tf.concat(labels_list, axis=0)
#             all_scores = tf.concat(scores_list, axis=0)
#             all_bboxes = tf.concat(bboxes_list, axis=0)
#             all_quadrilaterals = tf.concat(quadrilaterals_list, axis=0)

#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             init = tf.global_variables_initializer()
#             sess.run(init)

#             saver.restore(sess, get_checkpoint())

#             image_paths =\
#                 sorted(
#                     [path
#                      for pattern in FLAGS.input_image_stem_patterns.split(',')
#                      for path in Path(FLAGS.input_image_root).glob(pattern)],
#                     key=lambda e: int(re.findall(r'(?<=_)\d+(?=.)',
#                                                  e.name)[0]))
#             for i, image_path in enumerate(image_paths):
#                 image = imread(str(image_path))
#                 start_time = time.time()

#                 labels_,\
#                     scores_,\
#                     bboxes_,\
#                     quadrilaterals_,\
#                     output_shape_ =\
#                     sess.run([all_labels,
#                               all_scores,
#                               all_bboxes,
#                               all_quadrilaterals,
#                               output_shape],
#                              feed_dict={input_image: image,
#                                         input_shape: image.shape[:-1]})

#                 elapsed_time = time.time() - start_time
#                 print('{}: elapsed_time = {}'.format(i + 1, elapsed_time))

#                 # print('{}: elapsed_time = {}'.format(i + 1, elapsed_time))
#                 # bboxes_[:, 0] =\
#                 #     bboxes_[:, 0] * np_image.shape[0] / output_shape_[0, 0]
#                 # bboxes_[:, 1] =\
#                 #     bboxes_[:, 1] * np_image.shape[1] / output_shape_[0, 1]
#                 # bboxes_[:, 2] =\
#                 #     bboxes_[:, 2] * np_image.shape[0] / output_shape_[0, 0]
#                 # bboxes_[:, 3] =\
#                 quadrilaterals_[:, 0] =\
#                     quadrilaterals_[:, 0] *\
#                     image.shape[0] / output_shape_[0, 0]
#                 quadrilaterals_[:, 1] =\
#                     quadrilaterals_[:, 1] *\
#                     image.shape[1] / output_shape_[0, 1]
#                 quadrilaterals_[:, 2] =\
#                     quadrilaterals_[:, 2] *\
#                     image.shape[0] / output_shape_[0, 0]
#                 quadrilaterals_[:, 3] =\
#                     quadrilaterals_[:, 3] *\
#                     image.shape[1] / output_shape_[0, 1]
#                 quadrilaterals_[:, 4] =\
#                     quadrilaterals_[:, 4] *\
#                     image.shape[0] / output_shape_[0, 0]
#                 quadrilaterals_[:, 5] =\
#                     quadrilaterals_[:, 5] *\
#                     image.shape[1] / output_shape_[0, 1]
#                 quadrilaterals_[:, 6] =\
#                     quadrilaterals_[:, 6] *\
#                     image.shape[0] / output_shape_[0, 0]
#                 quadrilaterals_[:, 7] =\
#                     quadrilaterals_[:, 7] *\
#                     image.shape[1] / output_shape_[0, 1]

#                 annotation_file_name =\
#                     'task1_' + image_path.name.replace('.jpg', '.txt')
#                 with open(Path(FLAGS.output_directory).joinpath(
#                         annotation_file_name), 'w') as f:
#                     num_predicted_text_lines = np.shape(quadrilaterals_)[0]
#                     for i in range(num_predicted_text_lines):
#                         y1, x1, y2, x2,\
#                             y3, x3, y4, x4 =\
#                             [int(e) for e in quadrilaterals_[i, :]]
#                         score = float(scores_[i])
#                         if (y1 == 0 and x1 == 0 and
#                                 y2 == 0 and x2 == 0 and
#                                 y3 == 0 and x3 == 0 and
#                                 y4 == 0 and x4 == 0 and
#                                 score == 0.0):
#                             continue
#                         f.write('{},{},{},{},{},{},{},{},{}\n'.format(
#                             x1, y1, x2, y2,
#                             x3, y3, x4, y4, score))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
