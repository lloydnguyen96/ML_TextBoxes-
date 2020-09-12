from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from scipy.misc import imread, imsave
import numpy as np
import cv2

from net import textboxes_plusplus_net
from config import textboxes_plusplus_config as config

from dataset import dataset_common
from preprocessing import textboxes_plusplus_preprocessing
from utility import anchor_manipulator
from utility import drawing_toolbox
from utility import bbox_util
import time

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes',
    config.NUM_CLASSES,
    'Number of classes to use in the dataset.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size',
    config.TRAIN_IMAGE_SIZE,
    'The size of the input image for the model to use.')
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
    'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'min_size',
    4.,
    'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold',
    0.45,
    'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk',
    20,
    'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk',
    200,
    'Number of total object to keep for each image before nms.')
tf.app.flags.DEFINE_string(
    'image_file_name',
    'test.jpg',
    'Name of image file used to find text boxes.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    './models/textboxes_plusplus_trained_ctwd',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope',
    'textboxes_plusplus',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'storage_directory',
    './demo/demo_dest/',
    'The path to a storage_directory from which to write.')
tf.app.flags.DEFINE_string(
    'source_directory',
    './demo/demo_source/',
    'The path to a source_directory from which to detect.')

FLAGS=tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path=\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path=FLAGS.checkpoint_path
    print('checkpoint_path', checkpoint_path)
    return checkpoint_path

def main(_):
    with tf.Graph().as_default():
        out_shape=[FLAGS.train_image_size] * 2

        image_input=tf.placeholder(tf.uint8, shape=(None, None, 3))
        shape_input=tf.placeholder(tf.int32, shape=(2,))

        features, output_shape=\
            textboxes_plusplus_preprocessing.preprocess_for_eval(
                image_input,
                out_shape,
                data_format=FLAGS.data_format,
                output_rgb=False)
        features=tf.expand_dims(features, axis=0) # (1, ?, ?, 3)
        output_shape=tf.expand_dims(output_shape, axis=0) # (1, 2)

        with tf.variable_scope(FLAGS.model_scope,
                               default_name=None,
                               values=[features],
                               reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                anchor_processor=\
                    anchor_manipulator.AnchorProcessor(
                        positive_threshold=None,
                        ignore_threshold=None,
                        prior_scaling=config.PRIOR_SCALING)

                anchor_heights_all_layers,\
                anchor_widths_all_layers,\
                num_anchors_per_location_all_layers=\
                    anchor_processor.get_anchors_size_all_layers(
                        config.ALL_ANCHOR_SCALES,
                        config.ALL_EXTRA_SCALES,
                        config.ALL_ANCHOR_RATIOS,
                        config.NUM_FEATURE_LAYERS)

                # shape=(num_anchors_all_layers,).
                anchors_ymin,\
                anchors_xmin,\
                anchors_ymax,\
                anchors_xmax,\
                _=\
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

                backbone=textboxes_plusplus_net.VGG16Backbone(FLAGS.data_format)
                feature_layers=backbone.forward(features, training=False)
                # shape=(num_features,
                #        bs,
                #        fh,
                #        fw,
                #        num_anchors_per_locations * 2 * num_offsets)
                location_predictions, class_predictions=\
                    textboxes_plusplus_net.multibox_head(
                        feature_layers,
                        FLAGS.num_classes,
                        config.NUM_OFFSETS,
                        num_anchors_per_location_all_layers,
                        data_format=FLAGS.data_format)
                if FLAGS.data_format == 'channels_first':
                    class_predictions=\
                        [tf.transpose(pred,
                                      [0, 2, 3, 1])\
                         for pred in class_predictions]
                    location_predictions=\
                        [tf.transpose(pred,
                                      [0, 2, 3, 1])\
                         for pred in location_predictions]
                class_predictions=\
                    [tf.reshape(pred,
                                [-1, FLAGS.num_classes])\
                     for pred in class_predictions]
                location_predictions=\
                    [tf.reshape(pred, [-1, config.NUM_OFFSETS])\
                     for pred in location_predictions]

                class_predictions=tf.concat(class_predictions, axis=0)
                location_predictions=tf.concat(location_predictions, axis=0)

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
            bboxes_pred, quadrilaterals_pred=\
                anchor_processor.decode_anchors(
                    location_predictions,
                    anchors_ymin,
                    anchors_xmin,
                    anchors_ymax,
                    anchors_xmax)
            selected_bboxes,\
            selected_quadrilaterals,\
            selected_scores=\
                bbox_util.parse_by_class(
                    tf.squeeze(output_shape, axis=0),
                    class_predictions,
                    bboxes_pred,
                    quadrilaterals_pred,
                    FLAGS.num_classes,
                    FLAGS.select_threshold,
                    FLAGS.min_size,
                    FLAGS.keep_topk,
                    FLAGS.nms_topk,
                    FLAGS.nms_threshold)

            labels_list=[]
            scores_list=[]
            bboxes_list=[]
            quadrilaterals_list=[]
            for k, v in selected_scores.items():
                labels_list.append(tf.ones_like(v, tf.int32) * k)
                scores_list.append(v)
                bboxes_list.append(selected_bboxes[k])
                quadrilaterals_list.append(selected_quadrilaterals[k])
            all_labels=tf.concat(labels_list, axis=0)
            all_scores=tf.concat(scores_list, axis=0)
            all_bboxes=tf.concat(bboxes_list, axis=0)
            all_quadrilaterals=tf.concat(quadrilaterals_list, axis=0)

        saver=tf.train.Saver()
        with tf.Session() as sess:
            init=tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, get_checkpoint())

            total_time=0
            # np_image=imread('./demo/' + FLAGS.image_file_name)
            image_files_name=sorted(os.listdir(FLAGS.source_directory))
            for i, image_file_name in enumerate(image_files_name):
                np_image=imread(os.path.join(FLAGS.source_directory, image_file_name))
                start_time=time.time()

                labels_,\
                scores_,\
                bboxes_,\
                quadrilaterals_,\
                output_shape_=\
                    sess.run([all_labels,
                              all_scores,
                              all_bboxes,
                              all_quadrilaterals,
                              output_shape],
                             feed_dict={image_input : np_image,
                                        shape_input : np_image.shape[:-1]})

                elapsed_time=time.time() - start_time
                print('{}: elapsed_time = {}'.format(i + 1, elapsed_time))
                total_time+=elapsed_time

                bboxes_[:, 0]=bboxes_[:, 0] * np_image.shape[0] / output_shape_[0, 0]
                bboxes_[:, 1]=bboxes_[:, 1] * np_image.shape[1] / output_shape_[0, 1]
                bboxes_[:, 2]=bboxes_[:, 2] * np_image.shape[0] / output_shape_[0, 0]
                bboxes_[:, 3]=bboxes_[:, 3] * np_image.shape[1] / output_shape_[0, 1]
                quadrilaterals_[:, 0]=quadrilaterals_[:, 0] * np_image.shape[0] / output_shape_[0, 0]
                quadrilaterals_[:, 1]=quadrilaterals_[:, 1] * np_image.shape[1] / output_shape_[0, 1]
                quadrilaterals_[:, 2]=quadrilaterals_[:, 2] * np_image.shape[0] / output_shape_[0, 0]
                quadrilaterals_[:, 3]=quadrilaterals_[:, 3] * np_image.shape[1] / output_shape_[0, 1]
                quadrilaterals_[:, 4]=quadrilaterals_[:, 4] * np_image.shape[0] / output_shape_[0, 0]
                quadrilaterals_[:, 5]=quadrilaterals_[:, 5] * np_image.shape[1] / output_shape_[0, 1]
                quadrilaterals_[:, 6]=quadrilaterals_[:, 6] * np_image.shape[0] / output_shape_[0, 0]
                quadrilaterals_[:, 7]=quadrilaterals_[:, 7] * np_image.shape[1] / output_shape_[0, 1]

                # image_with_bboxes=\
                #     drawing_toolbox.draw_bboxes_on_image(
                #         np_image.copy(),
                #         labels_,
                #         scores_,
                #         bboxes_,
                #         thickness=2)
                # imsave('./demo/' + FLAGS.image_file_name[:-4] + '_bboxes' + '.jpg',
                #        image_with_bboxes)
                image_with_quadrilaterals=\
                    drawing_toolbox.draw_quadrilaterals_on_image(
                        np_image.copy(),
                        labels_,
                        scores_,
                        quadrilaterals_,
                        thickness=2)
                imsave(FLAGS.storage_directory + image_file_name[:-4] + '_quadrilaterals' + '.jpg', image_with_quadrilaterals)
                
                y1, x1, y2, x2,\
                y3, x3, y4, x4=[int(e) for e in quadrilaterals_[0, :]]

                topLeftVertex = [x1, y1]
                topRightVertex = [x2, y2]
                bottomLeftVertex = [x4, y4]
                bottomRightVertex = [x3, y3]

                ymin=int(round(bboxes_[0, 0]))
                xmin=int(round(bboxes_[0, 1]))
                ymax=int(round(bboxes_[0, 2]))
                xmax=int(round(bboxes_[0, 3]))

                PLATE_WIDTH = xmax - xmin
                PLATE_HEIGHT = ymax - ymin

                pts1 = np.float32([topLeftVertex, topRightVertex, bottomLeftVertex, bottomRightVertex])
                pts2 = np.float32([[0, 0], [PLATE_WIDTH, 0], [0, PLATE_HEIGHT], [PLATE_WIDTH, PLATE_HEIGHT]])
            
                M = cv2.getPerspectiveTransform(pts1, pts2)
                cropped_image = cv2.warpPerspective(np_image.copy(), M, (PLATE_WIDTH, PLATE_HEIGHT))
                imsave(FLAGS.storage_directory + image_file_name[:-4] + '_cropped' + '.jpg', cropped_image)
            
            print('total_time: ', total_time)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
