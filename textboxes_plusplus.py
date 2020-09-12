from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import sys

import tensorflow as tf

from net import textboxes_plusplus_net
from config import textboxes_plusplus_config as config

from dataset import dataset_common
from preprocessing import textboxes_plusplus_preprocessing
from utility import anchor_manipulator
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers',
    8,
    'The number of parallel (.tfrecord) readers that read data from the '
    'dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads',
    24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads',
    0,
    'The number of cpu threads used to run TensorFlow graph. 0 means the '
    'system picks an appropriate number.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction',
    1.,
    'GPU memory fraction to use.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu',
    False,
    'Whether there is GPU to use for training.')
# scaffold related configuration
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
    'The directory where the model will be stored and where to fine-tune '
    'model if checkpoint file existed. It is the first location to look for '
    'model and no preprocessing model file is required.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps',
    10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps',
    500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'keep_checkpoint_max',
    5,
    'The maximum number of checkpoints to keep.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_steps',
    10000,
    'The frequency with which the model is saved, in steps.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size',
    config.TRAIN_IMAGE_SIZE,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs',
    None,  # (not used, use max_number_of_steps instead)
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps',
    120000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size',
    32,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format',
    'channels_last',
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible with '
    'CPU. If left unspecified, the data format will be chosen automatically '
    'based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio',
    3.,
    'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold',  # high_thres
    0.5,
    'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold',  # low_thres
    0.5,
    'Matching threshold for the negative examples in the loss function.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed',
    20190512,
    'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay',
    5e-4,
    'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum',
    0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float(
    'learning_rate',
    1e-3,
    'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate',
    0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries',
    '1000, 80000, 100000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors',
    '0.1, 1, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between '
    'boundaries (comma-separated list).')
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    './models',
    'The path to a checkpoint from which to fine-tune. It is the second '
    'location to look for model and the preprocessing for model is required '
    '(see function get_init_fn_for_scaffold. E.g., The process of taking '
    'trained weights from VGG16 to finetune another model in which VGG16 is '
    'the base network need several preprocessing steps like change name '
    'scope ).')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope',
    'textboxes_plusplus',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope',
    'textboxes_plusplus',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes',
    'textboxes_plusplus/multibox_head_with_inception_module',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars',
    True,
    'When restoring a checkpoint would ignore missing variables.')
# process related configuration
tf.app.flags.DEFINE_string(
    'mode',
    'training',
    'Whether this process is used for training, evaluation or prediction.')
tf.app.flags.DEFINE_string(
    'dataset_pattern',
    'train-*',
    'regexp of filename of .tfrecord files')

FLAGS = tf.app.flags.FLAGS


def get_checkpoint():
    if tf.train.latest_checkpoint(FLAGS.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint '
                        'already exists in {}'.format(FLAGS.model_dir))
        return None
    # If full path is not provided.
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    return checkpoint_path


# Not used by now
def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of available
    GPUs.

    Note that this should eventually be handled by replicate_model_fn directly
    Multi-GPU support is currently experimental, however, so doing the work
    here until that feature is in place.
    """
    if FLAGS.multi_gpu:
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        num_gpus =\
            sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs were '
                             'found. To use CPU, run --multi_gpu=False.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size must be a'
                   'multiple of the number of available GPUs. Found {} GPUs'
                   'with a batch size of {}; try --batch_size={} instead.'
                   ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)
        return num_gpus
    return 0


def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(
        FLAGS.model_dir,  # First
        FLAGS.checkpoint_path,  # Second
        FLAGS.model_scope,
        FLAGS.checkpoint_model_scope,
        FLAGS.checkpoint_exclude_scopes,
        FLAGS.ignore_missing_vars,
        name_remap=None)


# I couldn't find a better way to pass params from input_fn to model_fn. Some
# tensors used by model_fn must be created in input_fn to ensure they are in
# the same graph. But when we put these tensors to labels's dict, the
# replicate_model_fn will split them into each GPU the problem is that they
# shouldn't be splited.
global_anchor_info = {}


def input_pipeline(dataset_pattern='train-*',
                   is_training=True,
                   batch_size=FLAGS.batch_size):
    def input_fn():
        target_shape = [FLAGS.train_image_size] * 2

        anchor_processor =\
            anchor_manipulator.AnchorProcessor(
                positive_threshold=FLAGS.match_threshold,
                ignore_threshold=FLAGS.neg_threshold,
                prior_scaling=config.PRIOR_SCALING)
        # anchor_processor: Python object

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
        # num_anchors_per_location_all_layers: [Python int, Python int, ...]

        anchors_ymin,\
            anchors_xmin,\
            anchors_ymax,\
            anchors_xmax,\
            inside_mask =\
            anchor_processor.get_all_anchors_all_layers(
                target_shape,
                anchor_heights_all_layers,
                anchor_widths_all_layers,
                num_anchors_per_location_all_layers,
                config.ANCHOR_OFFSETS,
                config.VERTICAL_OFFSETS,
                config.ALL_LAYER_SHAPES,
                config.ALL_LAYER_STRIDES,
                [FLAGS.train_image_size * 1.] * config.NUM_FEATURE_LAYERS,
                [False] * config.NUM_FEATURE_LAYERS)
        # anchors_ymin: 1d-tf.Tensor(num_anchors_all_layers) tf.float32
        # inside_mask: 1d-tf.Tensor(num_anchors_all_layers) tf.bool

        num_anchors_per_layer = []
        for ind, layer_shape in enumerate(config.ALL_LAYER_SHAPES):
            _, _num_anchors_per_layer =\
                anchor_processor.count_num_anchors_per_layer(
                    num_anchors_per_location_all_layers[ind],
                    layer_shape,
                    name='count_num_anchors_per_layer_{}'.format(ind))
            num_anchors_per_layer.append(_num_anchors_per_layer)
        # num_anchors_per_layer = [num_anchors_layer1, num_anchors_layer2, ...]
        # e.g., num_anchors_per_layer = [48 x 48 x 2 x 10, ...]

        def image_preprocessing_fn(image_, labels_, bboxes_, quadrilaterals_):
            return textboxes_plusplus_preprocessing.preprocess_image(
                image_,
                labels_,
                bboxes_,
                quadrilaterals_,
                target_shape,
                is_training=is_training,
                data_format=FLAGS.data_format,
                output_rgb=False)

        def anchor_encoder_fn(glabels_, gbboxes_, gquadrilaterals_):
            return anchor_processor.encode_anchors(
                glabels_,
                gbboxes_,
                gquadrilaterals_,
                anchors_ymin,
                anchors_xmin,
                anchors_ymax,
                anchors_xmax,
                inside_mask)
        image, _, shape, loc_targets, cls_targets, match_scores =\
            dataset_common.slim_get_batch(
                FLAGS.num_classes,
                batch_size,
                FLAGS.dataset_pattern[:-2],
                os.path.join(FLAGS.data_dir, dataset_pattern),
                FLAGS.num_readers,
                FLAGS.num_preprocessing_threads,
                image_preprocessing_fn,
                anchor_encoder_fn,
                num_epochs=FLAGS.train_epochs,
                is_training=is_training)

        global global_anchor_info
        global_anchor_info =\
            {'decode_fn':
             lambda pred: anchor_processor.batch_decode_anchors(
                pred,
                anchors_ymin,
                anchors_xmin,
                anchors_ymax,
                anchors_xmax),
             'num_anchors_per_layer': num_anchors_per_layer,
             'num_anchors_per_location_all_layers':
                num_anchors_per_location_all_layers}

        return image,\
            {'shape': shape,  # original shape from .tfrecord files
             'loc_targets': loc_targets,  # [bs, n_anchors, 12]
             'cls_targets': cls_targets,  # [bs, n_anchors]
             'match_scores': match_scores  # [bs, n_anchors]
             }
    return input_fn


def modified_smooth_l1(loc_pred,
                       loc_targets,
                       bbox_inside_weights=1.,
                       bbox_outside_weights=1.,
                       sigma=1.):
    """
    ResultLoss = outside_weights *
                SmoothL1(inside_weights * (loc_pred - loc_targets))
    SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                  |x| - 0.5 / sigma^2,    otherwise
    """
    with tf.name_scope('smooth_l1', values=[loc_pred, loc_targets]):
        sigma2 = sigma * sigma

        # x = i * (p - t)
        inside_mul =\
            tf.multiply(bbox_inside_weights,
                        tf.subtract(loc_pred,
                                    loc_targets))

        # l1_sign = |x| < 1.0/sigma2
        smooth_l1_sign =\
            tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        # l1_option1 = x^2 * 0.5 * sigma2 = 0.5 * (x * sigma)^2
        smooth_l1_option1 =\
            tf.multiply(tf.multiply(inside_mul,
                                    inside_mul),
                        0.5 * sigma2)
        # l1_option2 = |x| - 0.5/sigma2
        smooth_l1_option2 =\
            tf.subtract(tf.abs(inside_mul),
                        0.5 / sigma2)
        smooth_l1_result =\
            tf.add(tf.multiply(smooth_l1_option1,
                               smooth_l1_sign),
                   tf.multiply(smooth_l1_option2,
                               tf.abs(tf.subtract(smooth_l1_sign,
                                                  1.0))))
        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


def model_fn(features, labels, mode, params):
    # shape = labels['shape']
    loc_targets = labels['loc_targets']  # (bs, n_anchors_all_layers, n_ofsets)
    cls_targets = labels['cls_targets']  # (bs, n_anchors_all_layers)
    # match_scores = labels['match_scores']  # (bs, n_anchors_all_layers)

    global global_anchor_info
    # decode_fn = global_anchor_info['decode_fn']
    # num_anchors_per_layer = global_anchor_info['num_anchors_per_layer']
    num_anchors_per_location_all_layers =\
        global_anchor_info['num_anchors_per_location_all_layers']

    with tf.variable_scope(params['model_scope'],
                           default_name=None,
                           values=[features],
                           reuse=tf.AUTO_REUSE):

        # features: 4d-tf.Tensor-(bs, n_channels, fH, fW)-tf.float32
        backbone = textboxes_plusplus_net.VGG16Backbone(params['data_format'])
        feature_layers =\
            backbone.forward(features,
                             training=(mode == tf.estimator.ModeKeys.TRAIN))

        # shape = (num_feature_layers,
        #          bs,
        #          num_anchors_per_loc * 2 * num_offsets,
        #          fh,
        #          fw)
        location_predictions, class_predictions =\
            textboxes_plusplus_net.multibox_head(
                feature_layers,
                params['num_classes'],
                config.NUM_OFFSETS,
                num_anchors_per_location_all_layers,
                data_format=params['data_format'])

        # shape = (num_feature_layers,
        #          bs,
        #          fh,
        #          fw,
        #          num_anchors_per_loc * 2 * num_offsets)
        if params['data_format'] == 'channels_first':
            location_predictions =\
                [tf.transpose(pred,
                              [0, 2, 3, 1]) for pred in location_predictions]
            class_predictions =\
                [tf.transpose(pred,
                              [0, 2, 3, 1]) for pred in class_predictions]
            # if channels_first ==> move channel to last

        # shape = (num_feature_layers,
        #          bs,
        #          num_anchors_per_layer=fh * fw * num_anchors_per_loc * 2,
        #          num_offsets)
        location_predictions = [tf.reshape(pred,
                                           [tf.shape(features)[0],
                                            -1,
                                            config.NUM_OFFSETS])
                                for pred in location_predictions]
        class_predictions = [tf.reshape(pred,
                                        [tf.shape(features)[0],
                                         -1,
                                         params['num_classes']])
                             for pred in class_predictions]

        # shape = (bs,
        #          num_anchors_all_layers,
        #          num_offsets)
        location_predictions = tf.concat(location_predictions, axis=1)
        class_predictions = tf.concat(class_predictions, axis=1)

        # shape = (num_anchors_per_batch,
        #          num_offsets)
        location_predictions = tf.reshape(location_predictions,
                                          [-1, config.NUM_OFFSETS])
        class_predictions = tf.reshape(class_predictions,
                                       [-1, params['num_classes']])

    with tf.device('/cpu:0'):
        with tf.control_dependencies([class_predictions,
                                      location_predictions]):
            with tf.name_scope('post_forward'):
                # decoded_location_predictions =\
                #     decode_fn(tf.reshape(location_predictions,
                #                          [tf.shape(features)[0],
                #                           -1,
                #                           config.NUM_OFFSETS]))
                # decoded_location_predictions =\
                #     tf.reshape(decoded_location_predictions,
                #                [-1, config.NUM_OFFSETS])

                # - location_predictions[i, :] contains:
                # + bebore decode_fn:
                # [pred_cy, pred_cx, pred_h, pred_w, pred_y1, pred_x1, ...]
                # + after decode_fn:
                # [pred_ymin*, pred_xmin*, pred_ymax*, pred_xmax*, pred_y1*,
                # ...]
                # in which * means decoded value

                # e.g., cls_targets.get_shape():  (bs, n_anchors)
                # e.g., loc_targets.get_shape():  (bs, n_anchors, n_offsets)
                flaten_cls_targets = tf.reshape(cls_targets, [-1])
                # flaten_match_scores = tf.reshape(match_scores, [-1])
                flaten_loc_targets = tf.reshape(loc_targets,
                                                [-1, config.NUM_OFFSETS])
                # - loc_targets:
                # + gt_target 0 for negatives and ignores
                # + gt_target otherwise for object (positives and labeled bg)
                # - cls_targets:
                # + gt_label -1 for ignores
                # + gt_label 0 for labeled background (usually empty) and
                # negatives considered as background
                # + gt_label > 0 for detection object (positives)
                # - match_scores:
                # + gt_score >= 0

                # Each positive example has one label.
                # shape = (num_anchors_per_batch, )
                positive_mask = flaten_cls_targets > 0
                # shape = ()
                # n_positives = tf.count_nonzero(positive_mask)

                # shape = (bs, )
                # batch_n_positives = [n_positives_0, ..., n_positives_bs-1]
                batch_n_positives = tf.count_nonzero(cls_targets > 0, -1)

                # shape = (bs, num_anchors_all_layers)
                batch_negtive_mask = tf.equal(cls_targets, 0)
                # shape = (bs, )
                batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

                # shape = (bs, )
                batch_n_neg_select =\
                    tf.to_int32(params['negative_ratio'] *
                                tf.to_float(batch_n_positives))
                # shape = (bs, )
                batch_n_neg_select =\
                    tf.minimum(batch_n_neg_select,
                               tf.to_int32(batch_n_negtives))

                # hard negative mining for classification
                # class_predictions.get_shape(): (num_anchors_per_batch,
                #                                 num_classes)
                # shape = (bs, num_anchors_all_layers)
                predictions_for_bg =\
                    tf.nn.softmax(tf.reshape(class_predictions,
                                             [tf.shape(features)[0],
                                              -1,
                                              params['num_classes']]))[:, :, 0]
                # shape = (bs, num_anchors_all_layers)
                prob_for_negtives =\
                    tf.where(batch_negtive_mask,
                             0. - predictions_for_bg,
                             # ignore all the positives
                             0. - tf.ones_like(predictions_for_bg))
                # shape = (bs, num_anchors_all_layers)
                # rearrange the anchors according to the prob for bg.
                topk_prob_for_bg, _ =\
                    tf.nn.top_k(prob_for_negtives,
                                k=tf.shape(prob_for_negtives)[1])
                # shape = (bs, )
                score_at_k =\
                    tf.gather_nd(topk_prob_for_bg,
                                 tf.stack([tf.range(tf.shape(features)[0]),
                                           batch_n_neg_select - 1],
                                          axis=-1))
                # tf.stack =
                # [
                #   [0, n_negatives_0 - 1],
                #   [1, n_negatives_1 - 1],
                #   ...
                #   [bs - 1, n_negatives_bs-1 - 1],
                # ]
                # topk_prob_for_bg =
                #            n_negatives_0 - 1
                #                    | n_negatives_1 - 1
                #                    |       |  n_negatives_bs-1 - 1
                #                    |       |         |
                #                   \/      \/        \/
                # [        0        1       2         x     y   n_anchors-1
                #   0   [-0.001, -0.002, -0.01, ..., -1,   -1,   -1]
                #   1   [-0.002, -0.008, -0.05, ..., -0.7, -1,   -1]
                #       ...
                #  bs-1 [-0.05,  -0.09,  -0.1, ...,  -0.9, -1,   -1]
                # ]
                # NOTE: n_negatives_i never points to -1 because
                # batch_n_neg_select = tf.minimum(batch_n_neg_select,
                # batch_n_negtives)
                # score_at_k =
                # [    0       1         bs-1
                #   -0.002, -0.05, ..., -0.9
                # ]

                # shape = (bs, num_anchors_all_layers)
                selected_neg_mask =\
                    prob_for_negtives >= tf.expand_dims(score_at_k,
                                                        axis=-1)
                # selected_neg_mask =
                # [
                #  original_order[True, True, False, ..., False, False, False]
                #  original_order[True, True, True, False, ...,  False, False]
                #                 ...
                #  original_order[True, True, True, ...,  True,  False, False]
                # ]

                # include both selected negtive and all positive examples
                # Training is not allowed to change value of mask each time a
                # new batch is fetched. Model depends on mask to change
                # weights, the opposite is wrong.
                final_mask =\
                    tf.stop_gradient(
                        tf.logical_or(
                            tf.reshape(
                                tf.logical_and(
                                    batch_negtive_mask,
                                    selected_neg_mask),
                                [-1]),
                            positive_mask))

                # shape = (n_positive_anchors_per_batch +
                #          n_chosen_negative_anchors_per_batch, num_classes)
                class_predictions = tf.boolean_mask(class_predictions,
                                                    final_mask)
                # class_predictions[i, :] != 0 if anchor_i is positive anchor
                # or selected negative anchor else = 0

                # shape = (n_positive_anchors_per_batch, num_offsets)
                location_predictions =\
                    tf.boolean_mask(location_predictions,
                                    tf.stop_gradient(positive_mask))
                # shape = (n_positive_anchors_per_batch +
                #          n_chosen_negative_anchors_per_batch, )
                flaten_cls_targets =\
                    tf.boolean_mask(  # filter out unused negatives
                        tf.clip_by_value(  # consider ignores as background
                            flaten_cls_targets,
                            0,
                            params['num_classes']),
                        final_mask)
                # shape = (n_positive_anchors_per_batch, num_offsets)
                flaten_loc_targets =\
                    tf.stop_gradient(
                        tf.boolean_mask(
                            flaten_loc_targets,
                            positive_mask))

                # location_predictions is from model, flaten_loc_targets is
                # from data
                predictions = {
                    'classes': tf.argmax(class_predictions, axis=-1),
                    'probabilities': tf.reduce_max(
                        tf.nn.softmax(
                            class_predictions,
                            name='softmax_tensor'),
                        axis=-1),
                    # 'loc_predict': decoded_location_predictions
                }

    with tf.device('/cpu:0'):
        with tf.control_dependencies([class_predictions,
                                      location_predictions]):
            cls_accuracy =\
                tf.metrics.accuracy(flaten_cls_targets,
                                    predictions['classes'])
            cls_precision =\
                tf.metrics.precision(flaten_cls_targets,
                                     predictions['classes'])
            cls_recall =\
                tf.metrics.recall(flaten_cls_targets,
                                  predictions['classes'])
            metrics = {'cls_accuracy': cls_accuracy,
                       'cls_precision': cls_precision,
                       'cls_recall': cls_recall}

    # for logging purposes
    tf.identity(cls_accuracy[1], name='cls_accuracy')
    tf.summary.scalar('cls_accuracy', cls_accuracy[1])
    tf.identity(cls_precision[1], name='cls_precision')
    tf.summary.scalar('cls_precision', cls_precision[1])
    tf.identity(cls_recall[1], name='cls_recall')
    tf.summary.scalar('cls_recall', cls_recall[1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # - function name differences:
    # + without 'sparse_': labels is one-hot encoded tensor (n_examples,
    # n_classes)
    # + with 'sparse_': labels is tensor (n_examples, )
    # + without '_with_logits': return mean([loss(example) for example in
    # examples])
    # + with '_with_logits': return [loss(example) for example in examples]
    # NOTE: cross_entropy function calculates softmax(logits) inside.
    # shape = ()
    # average class loss of all examples
    cross_entropy =\
        tf.losses.sparse_softmax_cross_entropy(
            labels=flaten_cls_targets,
            logits=class_predictions) *\
        (params['negative_ratio'] + 1.)
    # create a tensor named cross_entropy for logging purposes
    tf.identity(cross_entropy,
                name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss',
                      cross_entropy)

    loc_loss =\
        modified_smooth_l1(location_predictions,
                           flaten_loc_targets,
                           sigma=1.)

    # average location loss of all positive anchors
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss,
                                            axis=-1),
                              name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if '_bn' not in trainable_var.name:
            if 'conv4_3_scale' not in trainable_var.name:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
    # add weight decay to the loss
    # We exclude the batch norm variables because doing so leads to a small
    # improvement in accuracy.
    total_loss =\
        tf.add(cross_entropy + loc_loss,
               tf.multiply(params['weight_decay'],
                           tf.add_n(l2_loss_vars),
                           name='l2_loss'),
               name='total_loss')

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                                mode=mode,
                                predictions=None,
                                loss=total_loss,
                                train_op=None,
                                eval_metric_ops=metrics,
                                scaffold=None)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay
                     for decay in params['lr_decay_factors']]
        learning_rate =\
            tf.train.piecewise_constant(
                tf.cast(global_step, tf.int32),
                [int(_) for _ in params['decay_boundaries']],
                lr_values)
        truncated_learning_rate =\
            tf.maximum(learning_rate,
                       tf.constant(params['end_learning_rate'],
                                   dtype=learning_rate.dtype),
                       name='learning_rate')
        # create a tensor named learning_rate for logging purposes
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=truncated_learning_rate,
            momentum=params['momentum'])
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None

    # used to train from scratch or finetune from its checkpoint file (it
    # means there are no differences in terms of network architecture between
    # the network we build now and the network stored in checkpoint file)
    return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions=predictions,
                            loss=total_loss,
                            train_op=train_op,
                            eval_metric_ops=metrics,
                            scaffold=tf.train.Scaffold(None))
    # used to finetune from other models
    # return tf.estimator.EstimatorSpec(
    #                         mode=mode,
    #                         predictions=predictions,
    #                         loss=total_loss,
    #                         train_op=train_op,
    #                         eval_metric_ops=metrics,
    #                         scaffold=tf.train.Scaffold(init_fn=get_init_fn()))


def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]


def main(_):
    # Using the Winograd non-fused algorithms provides a small performance
    # boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # per_process_gpu_memory_fraction:
    # The per_process_gpu_memory_fraction acts as a hard upper bound on the
    # amount of GPU memory that will be used by the process on each GPU on
    # the same machine. Currently, this fraction is applied uniformly to all
    # of the GPUs on the same machine; there is no way to set this on a
    # per-GPU basis.
    # Related note: setting CUDA_VISIBLE_DEVICES to limit TensorFlow to a
    # single GPU
    gpu_options =\
        tf.GPUOptions(
            per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    # allow_soft_placement:
    # If you would like TensorFlow to automatically choose an existing and
    # supported device to run the operations in case the specified one doesn't
    # exist, you can set allow_soft_placement to True in the configuration
    # option when creating the session. Which will help you if you
    # accidentally manually specified the wrong device or a device which does
    # not support a particular op. This is useful if you write a code which
    # can be executed in environments you do not know. You still can provide
    # useful defaults, but in the case of failure a graceful fallback.

    # log_device_placement:
    # To find out which devices (/cpu:0, /gpu:0, ...) your operations and
    # tensors are assigned to, create the session with log_device_placement
    # configuration option set to True. Which is helpful for debugging. For
    # each of the nodes of your graph, you will see the device it was assigned
    # to.

    # There are several possible forms of parallelism when running a
    # TensorFlow graph, and these options provide some control multi-core CPU
    # parallelism:
    # 1 -> intra_op_parallelism_threads:
    # Create a thread pool with intra_op_parallelism_threads threads.
    # If you have an operation that can be parallelized internally, such as
    # matrix multiplication (tf.matmul()) or a reduction
    # (e.g. tf.reduce_sum()), TensorFlow will execute it by scheduling tasks
    # in a thread pool with intra_op_parallelism_threads threads. This
    # configuration option therefore controls the maximum parallel speedup for
    # a single operation. Note that if you run multiple operations in
    # parallel, these operations will share this thread pool.
    # 2 -> inter_op_parallelism_threads:
    # Create a thread pool with inter_op_parallelism_threads threads.
    # If you have many operations that are independent in your TensorFlow
    # graph—because there is no directed path between them in the dataflow
    # graph—TensorFlow will attempt to run them concurrently, using a thread
    # pool with inter_op_parallelism_threads threads. If those operations have
    # a multithreaded implementation, they will (in most cases) share the same
    # thread pool for intra-op parallelism.
    # !!!Finally, both configuration options take a default value of 0, which
    # means "The system picks an appropriate number."

    # log_device_placement=False is helpful for debugging if set to True
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=FLAGS.num_cpu_threads,
        inter_op_parallelism_threads=FLAGS.num_cpu_threads,
        gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    # num_gpus = validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    # set up a RunConfig to only save checkpoints once per training cycle
    # save_checkpoints_secs: the frequency with which the model is saved, in
    # seconds (not used, use save_checkpoints_steps instead)
    if FLAGS.mode == 'training':
        run_config = tf.estimator.RunConfig().replace(
            save_checkpoints_secs=None).replace(
            save_checkpoints_steps=FLAGS.save_checkpoints_steps).replace(
            save_summary_steps=FLAGS.save_summary_steps).replace(
            keep_checkpoint_max=FLAGS.keep_checkpoint_max).replace(
            log_step_count_steps=FLAGS.log_every_n_steps).replace(
            tf_random_seed=FLAGS.tf_random_seed).replace(
            session_config=config)
        replicate_model_fn =\
            tf.contrib.estimator.replicate_model_fn(
                model_fn,
                loss_reduction=tf.losses.Reduction.MEAN)
    if FLAGS.mode == 'evaluation':
        run_config = tf.estimator.RunConfig().replace(
            save_checkpoints_secs=None).replace(
            save_checkpoints_steps=None).replace(
            save_summary_steps=None).replace(
            keep_checkpoint_max=FLAGS.keep_checkpoint_max).replace(
            log_step_count_steps=FLAGS.log_every_n_steps).replace(
            session_config=config)

    # num_gpus: can be removed from params
    # data_format: should also be passed to input_pipeline as a parameter
    # batch_size: can be removed from params
    # match_threshold: can be removed from params
    # neg_threshold: can be removed from params
    params = {
        # 'num_gpus': num_gpus,
        'data_format': FLAGS.data_format,
        # 'batch_size': FLAGS.batch_size,
        'model_scope': FLAGS.model_scope,
        'num_classes': FLAGS.num_classes,
        'negative_ratio': FLAGS.negative_ratio,
        # 'match_threshold': FLAGS.match_threshold,
        # 'neg_threshold': FLAGS.neg_threshold,
        'weight_decay': FLAGS.weight_decay,
    }
    if FLAGS.mode == 'training':
        params.update({
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })
    detector = tf.estimator.Estimator(
        model_fn=replicate_model_fn if FLAGS.mode == 'training' else model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config,
        params=params
    )

    tensors_to_log = {}
    if FLAGS.mode == 'training':
        tensors_to_log.update({
            'lr': 'learning_rate',
        })
    tensors_to_log.update({
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'cls_accuracy',
        'p': 'cls_precision',
        'r': 'cls_recall',
    })

    logging_hook =\
        tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=FLAGS.log_every_n_steps,
            formatter=lambda dicts: (', '.join(['%s=%.6f' % (k, v)
                                                for k, v in dicts.items()])))

    # hook = tf.train.ProfilerHook(save_steps=50,
    #                              output_dir='.',
    #                              show_memory=True)
    # hooks=[logging_hook]: list of tf.train.SessionRunHook subclass instances
    if FLAGS.mode == 'training':
        print('Starting a training cycle.')
        detector.train(
            input_fn=input_pipeline(dataset_pattern=FLAGS.dataset_pattern,
                                    is_training=FLAGS.mode == 'training',
                                    batch_size=FLAGS.batch_size),
            hooks=[logging_hook],
            max_steps=FLAGS.max_number_of_steps)
    elif FLAGS.mode == 'evaluation':
        print('Starting a evaluation cycle.')
        results = detector.evaluate(
            input_fn=input_pipeline(dataset_pattern=FLAGS.dataset_pattern,
                                    is_training=FLAGS.mode == 'training',
                                    batch_size=FLAGS.batch_size),
            hooks=[logging_hook],
            steps=None,
            checkpoint_path=get_checkpoint())
        print('results: ', results)
    else:  # prediction mode
        return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
