from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

import numpy as np
from scipy.misc import imread, imsave, imshow, imresize

from net import textboxes_plusplus_net
from config import textboxes_plusplus_config as config

from dataset import dataset_common
from preprocessing import textboxes_plusplus_preprocessing
from utility import anchor_manipulator
from utility import scaffolds
from utility import bbox_util
from utility import drawing_toolbox

# Hardware related configuration.
tf.app.flags.DEFINE_integer(
    'num_readers',
    8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads',
    24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads',
    0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction',
    1.,
    'GPU memory fraction to use.')
# scaffold related configuration.
tf.app.flags.DEFINE_string(
    'data_dir',
    './dataset/ctwd/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes',
    config.NUM_CLASSES,
    'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir',
    './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps',
    10,
    'The frequency with which logs are printed.')
# Model related configuration
tf.app.flags.DEFINE_integer(
    'batch_size',
    1,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_integer(
    'train_image_size',
    config.TRAIN_IMAGE_SIZE,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format',
    'channels_last', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold',
    0.01,
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
    200,
    'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk',
    400,
    'Number of total object to keep for each image before nms.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    './models',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope',
    'textboxes_plusplus',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS=tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.train.latest_checkpoint(FLAGS.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint '
                        'already exists in {}'.format(FLAGS.model_dir))
        return None
    # If full path is not provided.
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path=tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path=FLAGS.checkpoint_path
    return checkpoint_path

def save_image_with_labels(image,
                           labels_,
                           scores_,
                           bboxes_,
                           quadrilaterals_):
    if not hasattr(save_image_with_labels, 'counter'):
        # If counter doesn't exist yet, then initialize it.
        save_image_with_labels.counter=0
    save_image_with_labels.counter+=1

    bboxes_drawed_image=\
        drawing_toolbox.draw_bboxes_on_image(
            np.copy(image).astype(np.uint8),
            labels_,
            scores_,
            bboxes_,
            thickness=2)
    imsave(
        os.path.join('./debug/{}_bboxes.jpg').format(
            save_image_with_labels.counter),
        bboxes_drawed_image)
    quadrilaterals_drawed_image=\
        drawing_toolbox.draw_quadrilaterals_on_image(
            np.copy(image).astype(np.uint8),
            labels_,
            scores_,
            quadrilaterals_,
            thickness=2)
    imsave(
        os.path.join('./debug/{}_quadrilaterals.jpg').format(
            save_image_with_labels.counter),
        quadrilaterals_drawed_image)
    return save_image_with_labels.counter

# Couldn't find better way to pass params from input_fn to model_fn. Some
# tensors used by model_fn must be created in input_fn to ensure they are in
# the same graph. But when we put these tensors to labels's dict, the
# replicate_model_fn will split them into each GPU the problem is that they
# shouldn't be splited.
global_anchor_info={}

def input_pipeline(dataset_pattern='val-*',
                   is_training=False,
                   batch_size=FLAGS.batch_size):
    # is_training is overwriten when input_pipeline is called by predict.
    def input_fn():
        # We only support single batch when evaluation.
        assert(batch_size==1)
        target_shape=[FLAGS.train_image_size] * 2
        image_preprocessing_fn=\
            lambda image_, labels_, bboxes_, quadrilaterals_:\
            textboxes_plusplus_preprocessing.preprocess_image(
                image_,
                labels_,
                bboxes_,
                quadrilaterals_,
                target_shape,
                is_training=is_training,
                data_format=FLAGS.data_format,
                output_rgb=False)
        image, file_name, shape, output_shape=\
            dataset_common.slim_get_batch(
                FLAGS.num_classes,
                batch_size,
                ('train' if is_training else 'val'),
                os.path.join(FLAGS.data_dir, dataset_pattern),
                FLAGS.num_readers,
                FLAGS.num_preprocessing_threads,
                image_preprocessing_fn,
                anchor_encoding_fn=None,
                num_epochs=1,
                is_training=is_training)
        return {'image': image,
                'file_name': file_name,
                'shape': shape,
                'output_shape': output_shape}, None
    return input_fn

def model_fn(features, labels, mode, params):
    file_name=features['file_name']
    file_name=tf.identity(file_name, name='file_name')
    shape=features['shape']
    output_shape=features['output_shape']
    image=features['image']

    anchor_processor=anchor_manipulator.AnchorProcessor(
        positive_threshold=None,
        ignore_threshold=None,
        prior_scaling=config.PRIOR_SCALING)
    with tf.variable_scope(params['model_scope'],
                           default_name=None,
                           values=[image],
                           reuse=tf.AUTO_REUSE):
        with tf.device('/cpu:0'):
            anchor_heights_all_layers,\
            anchor_widths_all_layers,\
            num_anchors_per_location_all_layers=\
                anchor_processor.get_anchors_size_all_layers(
                    config.ALL_ANCHOR_SCALES,
                    config.ALL_EXTRA_SCALES,
                    config.ALL_ANCHOR_RATIOS,
                    config.NUM_FEATURE_LAYERS)

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

            backbone=\
                textboxes_plusplus_net.VGG16Backbone(params['data_format'])
            feature_layers=backbone.forward(
                image,
                training=(mode==tf.estimator.ModeKeys.TRAIN))
            location_predictions, class_predictions=\
                textboxes_plusplus_net.multibox_head(
                    feature_layers,
                    params['num_classes'],
                    config.NUM_OFFSETS,
                    num_anchors_per_location_all_layers,
                    data_format=params['data_format'])
            if params['data_format'] == 'channels_first':
                location_predictions=\
                    [tf.transpose(pred, [0, 2, 3, 1])\
                    for pred in location_predictions]
                class_predictions=\
                    [tf.transpose(pred, [0, 2, 3, 1])\
                     for pred in class_predictions]

            location_predictions=\
                [tf.reshape(pred,
                            [tf.shape(image)[0],
                             -1,
                             config.NUM_OFFSETS])\
                 for pred in location_predictions]
            class_predictions=\
                [tf.reshape(pred,
                            [tf.shape(image)[0],
                            -1,
                            params['num_classes']])\
                for pred in class_predictions]

            location_predictions=tf.concat(location_predictions, axis=1)
            class_predictions=tf.concat(class_predictions, axis=1)

            location_predictions=tf.reshape(location_predictions,
                                            [-1, config.NUM_OFFSETS])
            class_predictions=tf.reshape(class_predictions,
                                         [-1, params['num_classes']])
    with tf.device('/cpu:0'):
        bboxes_pred,\
        quadrilaterals_pred=\
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
                params['num_classes'],
                params['select_threshold'],
                params['min_size'],
                params['keep_topk'],
                params['nms_topk'],
                params['nms_threshold'])

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

    save_image_op=\
        tf.py_func(save_image_with_labels,
                   [textboxes_plusplus_preprocessing.unwhiten_image(
                       tf.squeeze(image, axis=0),
                       output_rgb=False),
                    all_labels * tf.to_int32(all_scores > 0.3),
                    all_scores,
                    all_bboxes,
                    all_quadrilaterals],
                   tf.int64,
                   stateful=True)
    tf.identity(save_image_op, name='save_image_op')
    predictions=\
        {'file_name': file_name,
         'shape': shape,
         'output_shape': output_shape}
    for class_ind in range(1, params['num_classes']):
        predictions['scores_{}'.format(class_ind)]=\
            tf.expand_dims(selected_scores[class_ind], axis=0)
        predictions['bboxes_{}'.format(class_ind)]=\
            tf.expand_dims(selected_bboxes[class_ind], axis=0)
        predictions['quadrilaterals_{}'.format(class_ind)]=\
            tf.expand_dims(selected_quadrilaterals[class_ind], axis=0)

    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            prediction_hooks=None,
            loss=None,
            train_op=None)
    else:
        raise ValueError('This script only support "PREDICT" mode!')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance
    # boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED']='1'

    gpu_options=\
        tf.GPUOptions(
            per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=FLAGS.num_cpu_threads,
        inter_op_parallelism_threads=FLAGS.num_cpu_threads,
        gpu_options=gpu_options)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config=tf.estimator.RunConfig().replace(
        save_checkpoints_secs=None).replace(
            save_checkpoints_steps=None).replace(
                save_summary_steps=None).replace(
                    keep_checkpoint_max=5).replace(
                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                            session_config=config)

    summary_dir=os.path.join(FLAGS.model_dir, 'predict')
    tf.gfile.MakeDirs(summary_dir)
    detector=tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config,
        params={
            'select_threshold': FLAGS.select_threshold,
            'min_size': FLAGS.min_size,
            'nms_threshold': FLAGS.nms_threshold,
            'nms_topk': FLAGS.nms_topk,
            'keep_topk': FLAGS.keep_topk,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
        })
    tensors_to_log={
        'cur_image': 'file_name',
        'cur_ind': 'save_image_op'
    }
    logging_hook=tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=FLAGS.log_every_n_steps)

    print('Starting a predict cycle.')
    pred_results=detector.predict(
        input_fn=input_pipeline(dataset_pattern='val-*',
                                is_training=False,
                                batch_size=FLAGS.batch_size),
        hooks=[logging_hook],
        checkpoint_path=get_checkpoint())#, yield_single_examples=False)

    det_results=list(pred_results)
    #print(list(det_results))

    for class_ind in range(1, FLAGS.num_classes):
        with open(os.path.join(summary_dir, 'results_{}.txt'.format(class_ind)), 'wt') as f:
            for image_ind, pred in enumerate(det_results):
                file_name=pred['file_name']
                shape=pred['shape']
                output_shape=pred['output_shape']
                scores=pred['scores_{}'.format(class_ind)]
                bboxes=pred['bboxes_{}'.format(class_ind)]
                bboxes[:, 0]=bboxes[:, 0] * shape[0] / output_shape[0]
                bboxes[:, 1]=bboxes[:, 1] * shape[1] / output_shape[1]
                bboxes[:, 2]=bboxes[:, 2] * shape[0] / output_shape[0]
                bboxes[:, 3]=bboxes[:, 3] * shape[1] / output_shape[1]

                valid_mask=np.logical_and(
                    (bboxes[:, 2] - bboxes[:, 0] > 1.),
                    (bboxes[:, 3] - bboxes[:, 1] > 1.))

                for det_ind in range(valid_mask.shape[0]):
                    if not valid_mask[det_ind]:
                        continue
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(file_name.decode('utf8')[:-4],
                                   scores[det_ind], # pred=softmax()
                                   bboxes[det_ind, 1], # xmin
                                   bboxes[det_ind, 0], # ymin
                                   bboxes[det_ind, 3], # xmax
                                   bboxes[det_ind, 2])) # ymax

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs('./debug')
    tf.app.run()
