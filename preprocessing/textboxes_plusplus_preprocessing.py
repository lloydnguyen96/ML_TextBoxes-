from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _ImageDimensions(image, rank=3):
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def preprocess_image(image,
                     labels,
                     bboxes,
                     quadrilaterals,
                     out_shape,
                     is_training=False,
                     data_format='channels_first',
                     output_rgb=True):
    if is_training:
        return preprocess_for_train(image,
                                    labels,
                                    bboxes,
                                    quadrilaterals,
                                    out_shape,
                                    data_format=data_format,
                                    output_rgb=output_rgb)
    else:
        return preprocess_for_eval(image,
                                   bboxes,
                                   quadrilaterals,
                                   out_shape,
                                   data_format=data_format,
                                   output_rgb=output_rgb)


# def preprocess_for_test(image,
#                         out_shape,
#                         data_format='channels_first',
#                         scope='preprocess_for_test',
#                         output_rgb=True):
#     with tf.name_scope(scope, 'preprocess_for_test', [image]):
#         image = tf.to_float(image)

#         if out_shape is not None:
#             image =\
#                 tf.image.resize_images(
#                     image,
#                     out_shape,
#                     method=tf.image.ResizeMethod.BILINEAR,
#                     align_corners=False)
#             image.set_shape(out_shape + [3])

#         height, width, _ = _ImageDimensions(image, rank=3)
#         output_shape = tf.stack([height, width])

#         image =\
#             _mean_image_subtraction(
#                 image,
#                 [_R_MEAN, _G_MEAN, _B_MEAN])

#         if not output_rgb:
#             image_channels =\
#                 tf.unstack(image, axis=-1, name='split_rgb')
#             image =\
#                 tf.stack([image_channels[2],
#                           image_channels[1],
#                           image_channels[0]],
#                          axis=-1,
#                          name='merge_bgr')

#         if data_format == 'channels_first':
#             image = tf.transpose(image, perm=(2, 0, 1))
#         return image, output_shape


def preprocess_for_eval(image,
                        bboxes,
                        quadrilaterals,
                        out_shape,
                        data_format='channels_first',
                        scope='preprocess_for_eval',
                        output_rgb=True):
    with tf.name_scope(scope, 'preprocess_for_eval', [image]):
        image = tf.to_float(image)
        height, width, _ = _ImageDimensions(image, rank=3)

        # We need to convert image to float before interpolating
        # 1:
        if out_shape is not None:
            image =\
                tf.image.resize_images(
                    image,
                    out_shape,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False)
            image.set_shape(out_shape + [3])

        # output_shape = tf.stack([height, width])

        # 2:
        image = _mean_image_subtraction(
            image,
            [_R_MEAN, _G_MEAN, _B_MEAN])

        # 3:
        if not output_rgb:
            image_channels =\
                tf.unstack(image, axis=-1, name='split_rgb')
            image =\
                tf.stack([image_channels[2],
                          image_channels[1],
                          image_channels[0]],
                         axis=-1,
                         name='merge_bgr')

        # 4:
        if data_format == 'channels_first':
            image = tf.transpose(image, perm=(2, 0, 1))

        # 5: resize bboxes, quadrilaterals such that their sizes are compatible
        # with image
        if bboxes is not None and quadrilaterals is not None:
            float_height, float_width = tf.to_float(height), tf.to_float(width)
            ymin, xmin, ymax, xmax = tf.unstack(bboxes, 4, axis=-1)
            target_height, target_width =\
                tf.to_float(out_shape[0]), tf.to_float(out_shape[1])
            ymin, ymax =\
                ymin * target_height / float_height,\
                ymax * target_height / float_height
            xmin, xmax =\
                xmin * target_width / float_width,\
                xmax * target_width / float_width
            bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            y1, x1, y2, x2, y3, x3, y4, x4 =\
                tf.unstack(quadrilaterals, 8, axis=-1)
            y1, y2, y3, y4 =\
                y1 * target_height / float_height,\
                y2 * target_height / float_height,\
                y3 * target_height / float_height,\
                y4 * target_height / float_height
            x1, x2, x3, x4 =\
                x1 * target_width / float_width,\
                x2 * target_width / float_width,\
                x3 * target_width / float_width,\
                x4 * target_width / float_width
            quadrilaterals =\
                tf.stack([y1, x1, y2, x2, y3, x3, y4, x4], axis=-1)

            cliped_ymin = tf.maximum(0., bboxes[:, 0])
            cliped_xmin = tf.maximum(0., bboxes[:, 1])
            cliped_ymax = tf.minimum(target_height - 1.,
                                     bboxes[:, 2])
            cliped_xmax = tf.minimum(target_width - 1.,
                                     bboxes[:, 3])

            mask_bboxes = tf.stack([cliped_ymin,
                                    cliped_xmin,
                                    cliped_ymax,
                                    cliped_xmax], axis=-1)

            cliped_y1 = tf.maximum(0., quadrilaterals[:, 0])
            cliped_x1 = tf.maximum(0., quadrilaterals[:, 1])
            cliped_y2 = tf.maximum(0., quadrilaterals[:, 2])
            cliped_x2 = tf.minimum(target_width - 1.,
                                   quadrilaterals[:, 3])
            cliped_y3 = tf.minimum(target_height - 1.,
                                   quadrilaterals[:, 4])
            cliped_x3 = tf.minimum(target_width - 1.,
                                   quadrilaterals[:, 5])
            cliped_y4 = tf.minimum(target_height - 1.,
                                   quadrilaterals[:, 6])
            cliped_x4 = tf.maximum(0., quadrilaterals[:, 7])

            mask_quadrilaterals = tf.stack([cliped_y1,
                                            cliped_x1,
                                            cliped_y2,
                                            cliped_x2,
                                            cliped_y3,
                                            cliped_x3,
                                            cliped_y4,
                                            cliped_x4], axis=-1)
            return image, mask_bboxes, mask_quadrilaterals
        return image


def preprocess_for_train(image,
                         labels,
                         bboxes,
                         quadrilaterals,
                         out_shape,
                         data_format='channels_first',
                         scope='preprocess_for_train',
                         output_rgb=True):
    with tf.name_scope(
            # the name argument that is passed to the op function
            scope,
            # the default name to use if the `name` argument is `None`
            'preprocess_for_train',
            # the list of `Tensor` arguments that are passed to the op function
            [image, labels, bboxes, quadrilaterals]):
        # 1: filter out as many difficult objects as possible in records (read
        # dataset_common.py)
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0].')
        orig_dtype = image.dtype  # uint8
        # 2: convert from [0, 256) => [0., 1.], convert from uint8 to
        # tf.float32
        if orig_dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # from image pixel value range [0, 256) to range [0., 1.]

        # 3: make some image changes like: change brightness, saturation, ...
        distort_image =\
            apply_with_random_selector(
                image,
                lambda x, ordering: distort_color(x, ordering, True),
                num_cases=4)

        # 4:
        # I: random_sample_patch_wrapper:
        # I1: index=0, max_attempt=3*
        # I2: while(index < 1 or (index < 3 and doesn't exist at least one bbox
        # in bboxes satisfying bbox_width > 5.0* and bbox_height > 5.0* and 64*
        # < bbox_area < image_area * 0.96*))
        #   J1: 50%* to run random_expand to create larger image
        #   J2: random_sample_patch:
        #       J21: sample_patch
        #           J211: check_roi_overlap
        #               J2111: jaccard_with_anchors
        #               J2112: check_roi_center
        #                   J21121: sample_width_height
        # I3: filter out those objects that still don't satisfy dimension
        # conditions (see check_bboxes) in the while loop above
        # I4: if run out of max_attempt, then return orginal image, labels, ...
        # I4: if not, return preprocessed image, labels, ...
        # J1: random_expand: expand in four directions (up, down, left, right)
        #   - new_image_height = x * old_image_height
        #   - new_image_width = x * old_image_width
        #   with 1.1* <= x <= 1.5*
        #   - padding value: not zero but [_R_MEAN*/255., _G_MEAN*/255.,
        #   _B_MEAN*/255.]
        #   - move Oxy
        # J2: random_sample_patch: sample (crop) image
        #   J20:
        #   - choose ratio from ratio_list = [0.8*, 0.9*, 1.0*] with equal prob
        #   + if ratio = 1.0 (nothing changed), return input itself
        #   + else: return output of J21 (actually do sampling or cropping)
        #   J21:
        #   - patch, labels, ... = output of J211
        #   - new coordinates for bboxes, quadrilaterals after sampling:
        #   + move Oxy
        #   + crop out-of-patch dimension values of y, x
        #   => new_bboxes, new_quadrilaterals
        #   - if patch is zero (patch_height = 0 or patch_width = 0), then
        #   return input itself
        #   - else: return cropped_image = tf.slice(image, patch), labels,
        #   new_bboxes, new_quadrilaterals
        #       J211:
        #       index = 0, max_attempt = 50*
        #       while(no object present in current patch or index = 0 (not try
        #       anytime) or (index < max_attempt and exists at least one object
        #       with iou(bbox, patch) < ratio)). This means we want at least
        #       one object present in patch and all patch objects must have
        #       iou(object_bbox, patch) >= ratio. Then
        #           call J2112
        #       if exists at least one object in patch, return output of J2112
        #       else return input itself
        #           J2112:
        #           index = 0, max_attempt = 20*
        #           while(index = 0 (not try anytime) or (index < max_attempt
        #           and there exists no object with CENTER in patch))
        #               choose patch_width, patch_height from J21121
        #               choose patch_xmin, patch_ymin
        #                   J21121:
        #                   index = 0, max_attempt = 15*
        #                   while (index = 0 or (index < max_attempt and
        #                   sampled_width > sampled_height * 1.625* or
        #                   sampled_height > sampled_width * 1.625 *))
        #                       patch_width = x * image_width
        #                       patch_height = y * image_height
        #                       0.4* <= x, y <= 0.999*
        #                   return (sampled_width, sampled_height)
        #               create patch using patch_width, patch_height,
        #               patch_xmin, patch_ymin
        #           filter out those objects whose bbox center lies outside of
        #           the patch (including those with center in roi border)
        #           ==> filtered objects (labels, bboxes, quadrilaterals)
        # NOTE: those values with * beside means that those values are
        # customizable for preprocessing step
        random_sample_image, labels, bboxes, quadrilaterals =\
            random_sample_patch_wrapper(distort_image,
                                        labels,
                                        bboxes,
                                        quadrilaterals)

        height, width, _ = _ImageDimensions(random_sample_image, rank=3)
        float_height, float_width = tf.to_float(height), tf.to_float(width)

        ymin, xmin, ymax, xmax = tf.unstack(bboxes, 4, axis=-1)
        target_height, target_width =\
            tf.to_float(out_shape[0]), tf.to_float(out_shape[1])
        ymin, ymax =\
            ymin * target_height / float_height,\
            ymax * target_height / float_height
        xmin, xmax =\
            xmin * target_width / float_width,\
            xmax * target_width / float_width
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        y1, x1, y2, x2, y3, x3, y4, x4 = tf.unstack(quadrilaterals, 8, axis=-1)
        y1, y2, y3, y4 =\
            y1 * target_height / float_height,\
            y2 * target_height / float_height,\
            y3 * target_height / float_height,\
            y4 * target_height / float_height

        x1, x2, x3, x4 =\
            x1 * target_width / float_width,\
            x2 * target_width / float_width,\
            x3 * target_width / float_width,\
            x4 * target_width / float_width
        quadrilaterals = tf.stack([y1, x1, y2, x2, y3, x3, y4, x4], axis=-1)

        # 5: resize image
        # TF converts to float before interpolating
        random_sample_resized_image =\
            tf.image.resize_images(random_sample_image,
                                   out_shape,
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   align_corners=False)
        random_sample_resized_image.set_shape([None, None, 3])

        # 6: convert back to 3d-tf.Tensor(H, W, RGB)-uint8
        # Note that converting from floating point inputs to integer types may
        # lead to over/underflow problems. Set saturate to `True` to avoid such
        # problem in problematic conversions.
        final_image =\
            tf.to_float(
                tf.image.convert_image_dtype(
                    random_sample_resized_image,
                    orig_dtype,
                    saturate=True))

        # 7:
        final_image =\
            _mean_image_subtraction(final_image,
                                    [_R_MEAN, _G_MEAN, _B_MEAN])

        final_image.set_shape(out_shape + [3])

        # 8: rgb -> bgr
        if not output_rgb:
            image_channels =\
                tf.unstack(final_image, axis=-1, name='split_rgb')
            final_image =\
                tf.stack([image_channels[2],
                          image_channels[1],
                          image_channels[0]],
                         axis=-1,
                         name='merge_bgr')

        # 9: (H, W, C) -> (C, H, W)
        #     0  1  2  ->  2  0  1
        if data_format == 'channels_first':
            final_image = tf.transpose(final_image, perm=(2, 0, 1))
        return final_image, labels, bboxes, quadrilaterals


def _mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def apply_with_random_selector(x, func, num_cases):

    # sel value is uniformly chosen from [0, 1, 2, num_cases-1]
    sel = tf.random_uniform([],
                            maxval=num_cases,
                            dtype=tf.int32)
    # control flow in Python is for normal Python variables and values
    # control flow in TensorFlow is for tf.Variables, Tensors, tf.constant
    # control flow in TensorFlow is represented by control_flow_ops operations
    # in graph
    # NOTE: Why do we use random number op of TF, not plain Python???
    # ==> sel value will be different from session to session, so it will keeps
    # changing between records. We don't want to use same sel value for all
    # records because it is simply a selection value.
    # Why can this be different from session to session???
    # ==> random number op is built on TF graph, which will be run and rerun
    # each session. If sel is plain Python random variable from this function,
    # it means it doesn't lie on any sessions, but graph creation process. It
    # also not an object on graph, so session can't flow tensors through that
    # op and run that op ==> not generating new random value

    # for example:
    # [
    #    func(control_flow_ops.switch(x, tf.equal(sel, 0))[1], 0),
    #    func(control_flow_ops.switch(x, tf.equal(sel, 1))[1], 1),
    #    func(control_flow_ops.switch(x, tf.equal(sel, 2))[1], 2),
    #    func(control_flow_ops.switch(x, tf.equal(sel, 3))[1], 3),
    # ]
    # [
    #    func(control_flow_ops.switch(x, tf.equal(1, 0))[1], 0),
    #    func(control_flow_ops.switch(x, tf.equal(1, 1))[1], 1),
    #    func(control_flow_ops.switch(x, tf.equal(1, 2))[1], 2),
    #    func(control_flow_ops.switch(x, tf.equal(1, 3))[1], 3),
    # ]
    # [
    #    func(control_flow_ops.switch(x, False)[1], 0),
    #    func(control_flow_ops.switch(x, True)[1], 1),
    #    func(control_flow_ops.switch(x, False)[1], 2),
    #    func(control_flow_ops.switch(x, False)[1], 3),
    # ]
    # [
    #    func(Tensor with no value, 0),
    #    func(Tensor with value x, 1),
    #    func(Tensor with no value, 2),
    #    func(Tensor with no value, 3),
    # ]
    # ==> merge(
    # [
    #    func(Tensor with no value, 0),
    #    func(Tensor with value x, 1),
    #    func(Tensor with no value, 2),
    #    func(Tensor with no value, 3),
    # ]
    # )
    # = (result of func(Tensor with value x, 1), index=1)

    # control_flow_ops.switch(data, pred) --> output_false
    #                                     `--> output_true
    # tensor (containing data) flows in TF graph once encountered switch op
    # will flow through first direction (output_false) if pred is false or
    # second direction (output_true) otherwise

    # image---switch_op_case0---output_false (x)
    #  \ \ \                 \--output_true---brightness_op---saturation_op--
    #   \ \ \-switch_op_case1---output_false (x)                            |
    #    \ \                 \--output_true---saturation_op---brightness_op-+
    #     \ \-switch_op_case2---output_false (x)                            |
    #      \                 \--output_true---saturation_op---brightness_op-+
    #       \-switch_op_case3---output_false (x)                            |
    #                        \--output_true---saturation_op---brightness_op-+
    #                                 this function output <----merge_op---/
    # NOTE:
    # - (x) means excluded: the directions that we don't use
    # - depends on sel tensor value, tensor will flow through switch_op_case0
    # output_true (or case1, case2, case3 exclusively. The tensor will flow
    # through output_false of the other casei but those outputs are not
    # connected to anything), random_brightness op, saturation_op and merge_op
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x,
                                     tf.equal(sel, case))[1],
             case)
        for case in range(num_cases)])[0]


def distort_color(image,
                  color_ordering=0,
                  fast_mode=True,
                  scope=None):

    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # error: tf.clip_by_value will allow value 1.0, invalid pixel value of
        # image!!!
        return tf.clip_by_value(image, 0.0, 1.0)


def random_sample_patch_wrapper(image,
                                labels,
                                bboxes,
                                quadrilaterals):
    with tf.name_scope('random_sample_patch_wrapper'):
        orgi_image, orgi_labels, orgi_bboxes, orgi_quadrilaterals =\
            image, labels, bboxes, quadrilaterals

        def check_bboxes(image, bboxes):
            img_shape = tf.shape(image)
            areas = (bboxes[:, 3] - bboxes[:, 1] + 1.) *\
                    (bboxes[:, 2] - bboxes[:, 0] + 1.)
            return tf.logical_and(
                tf.logical_and(
                    areas > 64.,
                    areas < tf.to_float(img_shape[0] * img_shape[1]) * 0.96),
                tf.logical_and((bboxes[:, 3] - bboxes[:, 1] + 1.) > 5.,
                               (bboxes[:, 2] - bboxes[:, 0] + 1.) > 5.))
        index = 0
        max_attempt = 3

        def condition(index,
                      image,
                      labels,
                      bboxes,
                      quadrilaterals,
                      orgi_image,
                      orgi_labels,
                      orgi_bboxes,
                      orgi_quadrilaterals):
            return tf.logical_or(
                tf.logical_and(
                    tf.reduce_sum(
                        tf.cast(check_bboxes(image, bboxes), tf.int64)) < 1,
                        # tf.cast(
                        #     check_bboxes(image,
                        #                  bboxes),
                        #     tf.int64)) < tf.shape(bboxes,
                        #                           out_type=tf.int64)[0],
                    tf.less(index, max_attempt)),
                tf.less(index, 1))

        def body(index,
                 image,
                 labels,
                 bboxes,
                 quadrilaterals,
                 orgi_image,
                 orgi_labels,
                 orgi_bboxes,
                 orgi_quadrilaterals):
            image, bboxes, quadrilaterals =\
                tf.cond(
                    tf.random_uniform([],
                                      minval=0.,
                                      maxval=1.,
                                      dtype=tf.float32) < 0.5,
                    lambda: (orgi_image, orgi_bboxes, orgi_quadrilaterals),
                    lambda: random_expand(
                        orgi_image,
                        orgi_bboxes,
                        orgi_quadrilaterals,
                        tf.random_uniform([1],
                                          minval=1.1,
                                          # maxval=1.4,
                                          maxval=1.5,
                                          dtype=tf.float32)[0]))
            # Distort image and bounding boxes.
            random_sample_image, labels, bboxes, quadrilaterals =\
                random_sample_patch(image,
                                    orgi_labels,
                                    bboxes,
                                    quadrilaterals,
                                    ratio_list=[0.8, 0.9, 1.]
                                    # ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.]
                                    )
            random_sample_image.set_shape([None, None, 3])
            return index+1,\
                random_sample_image,\
                labels,\
                bboxes,\
                quadrilaterals,\
                orgi_image,\
                orgi_labels,\
                orgi_bboxes,\
                orgi_quadrilaterals

        # tf.while_loop is while()_do(), not do()_while()
        # When the flag swap_memory is true, we swap out these tensors from GPU
        # to CPU.
        [index, image, labels, bboxes, quadrilaterals,
         orgi_image, orgi_labels, orgi_bboxes, orgi_quadrilaterals] =\
            tf.while_loop(condition,
                          body,
                          [index, image,
                           labels, bboxes, quadrilaterals,
                           orgi_image,
                           orgi_labels, orgi_bboxes, orgi_quadrilaterals],
                          parallel_iterations=4,
                          back_prop=False,
                          swap_memory=True)

        valid_mask = check_bboxes(image, bboxes)
        labels, bboxes, quadrilaterals =\
            tf.boolean_mask(labels, valid_mask),\
            tf.boolean_mask(bboxes, valid_mask),\
            tf.boolean_mask(quadrilaterals, valid_mask)
        return tf.cond(tf.less(index, max_attempt),
                       lambda: (image,
                                labels, bboxes, quadrilaterals),
                       lambda: (orgi_image,
                                orgi_labels, orgi_bboxes, orgi_quadrilaterals))


def random_expand(image,
                  bboxes,
                  quadrilaterals,
                  ratio=1.2,
                  name=None):
    with tf.name_scope('random_expand'):
        image = tf.convert_to_tensor(image, name='image')
        if image.get_shape().ndims != 3:
            raise ValueError('\'image\' must have 3 dimensions.')
        height, width, depth = _ImageDimensions(image, rank=3)
        float_height, float_width = tf.to_float(height), tf.to_float(width)

        canvas_width, canvas_height =\
            float_width * ratio,\
            float_height * ratio
        mean_color_of_image = [_R_MEAN/255., _G_MEAN/255., _B_MEAN/255.]
        # with tf.control_dependencies(
        #         [tf.debugging.assert_greater(canvas_width, width),
        #          tf.debugging.assert_greater(canvas_height, height)]):
        # x = tf.random_uniform([],
        #                       minval=0,
        #                       maxval=canvas_width - width,
        #                       dtype=tf.int32)
        # y = tf.random_uniform([],
        #                       minval=0,
        #                       maxval=canvas_height - height,
        #                       dtype=tf.int32)
        x = tf.to_int32(
            tf.random_uniform([],
                              minval=0,
                              maxval=canvas_width - float_width,
                              dtype=tf.float32))
        y = tf.to_int32(
            tf.random_uniform([],
                              minval=0,
                              maxval=canvas_height - float_height,
                              dtype=tf.float32))
        paddings = tf.convert_to_tensor(
            [[y, tf.to_int32(canvas_height) - height - y],
             [x, tf.to_int32(canvas_width) - width - x]])

        big_canvas = tf.stack([tf.pad(image[:, :, 0],
                                      paddings,
                                      "CONSTANT",
                                      constant_values=mean_color_of_image[0]),
                               tf.pad(image[:, :, 1],
                                      paddings,
                                      "CONSTANT",
                                      constant_values=mean_color_of_image[1]),
                               tf.pad(image[:, :, 2],
                                      paddings,
                                      "CONSTANT",
                                      constant_values=mean_color_of_image[2])],
                              axis=-1)
        return big_canvas,\
            bboxes + tf.cast(tf.stack([y, x, y, x]), bboxes.dtype),\
            quadrilaterals + tf.cast(tf.stack([y, x, y, x, y, x, y, x]),
                                     quadrilaterals.dtype)


def random_sample_patch(image,
                        labels,
                        bboxes,
                        quadrilaterals,
                        ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.],
                        name=None):
    def jaccard_with_anchors(roi, bboxes):
        with tf.name_scope('jaccard_with_anchors'):
            int_ymin = tf.maximum(roi[0], bboxes[:, 0])
            int_xmin = tf.maximum(roi[1], bboxes[:, 1])
            int_ymax = tf.minimum(roi[2], bboxes[:, 2])
            int_xmax = tf.minimum(roi[3], bboxes[:, 3])
            h = tf.maximum(int_ymax - int_ymin + 1., 0.)
            w = tf.maximum(int_xmax - int_xmin + 1., 0.)
            inter_vol = h * w
            union_vol = (roi[3] - roi[1] + 1.) *\
                (roi[2] - roi[0] + 1.) +\
                ((bboxes[:, 2] - bboxes[:, 0] + 1.) *
                 (bboxes[:, 3] - bboxes[:, 1] + 1.) -
                 inter_vol)
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard

    def sample_width_height(width, height):
        with tf.name_scope('sample_width_height'):
            index = 0
            max_attempt = 10
            sampled_width, sampled_height = width, height

            def condition(index,
                          sampled_width,
                          sampled_height,
                          width,
                          height):
                return tf.logical_or(
                    tf.logical_and(
                        tf.logical_or(
                            tf.greater(sampled_width,
                                       sampled_height * 1.625),
                            tf.greater(sampled_height,
                                       sampled_width * 1.625)),
                        tf.less(index, max_attempt)),
                    tf.less(index, 1))

            def body(index,
                     sampled_width,
                     sampled_height,
                     width,
                     height):
                sampled_width =\
                    tf.random_uniform([1],
                                      minval=0.4,
                                      # minval=0.05,
                                      maxval=0.999,
                                      dtype=tf.float32)[0] * width
                sampled_height =\
                    tf.random_uniform([1],
                                      minval=0.4,
                                      # minval=0.05,
                                      maxval=0.999,
                                      dtype=tf.float32)[0] * height

                return index+1, sampled_width, sampled_height, width, height

            [index, sampled_width, sampled_height, _, _] =\
                tf.while_loop(condition,
                              body,
                              [index,
                               sampled_width,
                               sampled_height,
                               width,
                               height],
                              parallel_iterations=4,
                              back_prop=False,
                              swap_memory=True)

            return tf.cast(sampled_width, tf.int32),\
                tf.cast(sampled_height, tf.int32)

    def check_roi_center(width,
                         height,
                         labels,
                         bboxes,
                         quadrilaterals):
        with tf.name_scope('check_roi_center'):
            index = 0
            max_attempt = 20
            float_width = tf.to_float(width)
            float_height = tf.to_float(height)
            roi = [0., 0., float_height - 1., float_width - 1.]

            mask = tf.cast(tf.zeros_like(labels, dtype=tf.uint8), tf.bool)
            center_x, center_y =\
                (bboxes[:, 1] + bboxes[:, 3]) / 2,\
                (bboxes[:, 0] + bboxes[:, 2]) / 2

            def condition(index, roi, mask):
                return tf.logical_or(
                    tf.logical_and(
                        tf.reduce_sum(
                            tf.to_int32(mask)) < 1,
                        tf.less(index, max_attempt)),
                    tf.less(index, 1))

            def body(index, roi, mask):
                sampled_width, sampled_height =\
                    sample_width_height(float_width, float_height)

                x = tf.random_uniform([],
                                      minval=0,
                                      maxval=width - sampled_width,
                                      dtype=tf.int32)
                y = tf.random_uniform([],
                                      minval=0,
                                      maxval=height - sampled_height,
                                      dtype=tf.int32)

                roi = [tf.to_float(y),
                       tf.to_float(x),
                       tf.to_float(y + sampled_height - 1),
                       tf.to_float(x + sampled_width - 1)]

                mask_min = tf.logical_and(tf.greater(center_y, roi[0]),
                                          tf.greater(center_x, roi[1]))
                mask_max = tf.logical_and(tf.less(center_y,
                                                  roi[2]),
                                          tf.less(center_x,
                                                  roi[3]))
                mask = tf.logical_and(mask_min, mask_max)

                return index + 1, roi, mask

            [index, roi, mask] =\
                tf.while_loop(condition,
                              body,
                              [index, roi, mask],
                              parallel_iterations=10,
                              back_prop=False,
                              swap_memory=True)

            mask_labels = tf.boolean_mask(labels, mask)
            mask_bboxes = tf.boolean_mask(bboxes, mask)
            mask_quadrilaterals = tf.boolean_mask(quadrilaterals, mask)

            return roi, mask_labels, mask_bboxes, mask_quadrilaterals

    def check_roi_overlap(width,
                          height,
                          labels,
                          bboxes,
                          quadrilaterals,
                          min_iou):
        with tf.name_scope('check_roi_overlap'):
            index = 0
            max_attempt = 50
            float_width = tf.to_float(width)
            float_height = tf.to_float(height)
            roi = [0., 0., float_height - 1., float_width - 1.]

            mask_labels = labels
            mask_bboxes = bboxes
            mask_quadrilaterals = quadrilaterals

            def condition(index,
                          roi,
                          mask_labels,
                          mask_bboxes,
                          mask_quadrilaterals):
                return tf.logical_or(
                    tf.logical_or(
                        tf.logical_and(
                            tf.reduce_sum(
                                tf.to_int32(
                                    jaccard_with_anchors(
                                        roi,
                                        mask_bboxes) < min_iou)) > 0,
                            tf.less(index, max_attempt)),
                        tf.less(index, 1)),
                    tf.less(tf.shape(mask_labels)[0], 1))

            def body(index,
                     roi,
                     mask_labels,
                     mask_bboxes,
                     mask_quadrilaterals):
                roi, mask_labels, mask_bboxes, mask_quadrilaterals =\
                    check_roi_center(width,
                                     height,
                                     labels,
                                     bboxes,
                                     quadrilaterals)
                return index+1,\
                    roi,\
                    mask_labels,\
                    mask_bboxes,\
                    mask_quadrilaterals

            [index, roi, mask_labels,
             mask_bboxes, mask_quadrilaterals] =\
                tf.while_loop(condition,
                              body,
                              [index, roi, mask_labels,
                               mask_bboxes, mask_quadrilaterals],
                              parallel_iterations=16,
                              back_prop=False,
                              swap_memory=True)

            return tf.cond(
                tf.greater(
                    tf.shape(mask_labels)[0], 0),
                lambda: (tf.to_int32(
                    [roi[0],
                     roi[1],
                     roi[2] - roi[0] + 1.,
                     roi[3] - roi[1] + 1.]),
                          mask_labels,
                          mask_bboxes, mask_quadrilaterals),
                lambda: (tf.to_int32([0.,
                                      0.,
                                      float_height,
                                      float_width]),
                         labels,
                         bboxes, quadrilaterals))

    def sample_patch(image,
                     labels,
                     bboxes,
                     quadrilaterals,
                     min_iou):
        with tf.name_scope('sample_patch'):
            height, width, depth = _ImageDimensions(image, rank=3)

            roi_slice_range, mask_labels, mask_bboxes, mask_quadrilaterals =\
                check_roi_overlap(width,
                                  height,
                                  labels,
                                  bboxes,
                                  quadrilaterals,
                                  min_iou)
            # roi_slice_range=[y, x, h, w] of sampled patch

            # Add offset.
            offset = tf.cast(tf.stack([roi_slice_range[0],
                                       roi_slice_range[1],
                                       roi_slice_range[0],
                                       roi_slice_range[1]]),
                             mask_bboxes.dtype)
            mask_bboxes = mask_bboxes - offset
            qoffset = tf.cast(tf.stack([roi_slice_range[0],
                                        roi_slice_range[1],
                                        roi_slice_range[0],
                                        roi_slice_range[1],
                                        roi_slice_range[0],
                                        roi_slice_range[1],
                                        roi_slice_range[0],
                                        roi_slice_range[1]]),
                              mask_quadrilaterals.dtype)
            mask_quadrilaterals = mask_quadrilaterals - qoffset

            cliped_ymin = tf.maximum(0., mask_bboxes[:, 0])
            cliped_xmin = tf.maximum(0., mask_bboxes[:, 1])
            cliped_ymax = tf.minimum(tf.to_float(roi_slice_range[2]) - 1.,
                                     mask_bboxes[:, 2])
            cliped_xmax = tf.minimum(tf.to_float(roi_slice_range[3]) - 1.,
                                     mask_bboxes[:, 3])

            mask_bboxes = tf.stack([cliped_ymin,
                                    cliped_xmin,
                                    cliped_ymax,
                                    cliped_xmax], axis=-1)

            cliped_y1 = tf.maximum(0., mask_quadrilaterals[:, 0])
            cliped_x1 = tf.maximum(0., mask_quadrilaterals[:, 1])
            cliped_y2 = tf.maximum(0., mask_quadrilaterals[:, 2])
            cliped_x2 = tf.minimum(tf.to_float(roi_slice_range[3]) - 1.,
                                   mask_quadrilaterals[:, 3])
            cliped_y3 = tf.minimum(tf.to_float(roi_slice_range[2]) - 1.,
                                   mask_quadrilaterals[:, 4])
            cliped_x3 = tf.minimum(tf.to_float(roi_slice_range[3]) - 1.,
                                   mask_quadrilaterals[:, 5])
            cliped_y4 = tf.minimum(tf.to_float(roi_slice_range[2]) - 1.,
                                   mask_quadrilaterals[:, 6])
            cliped_x4 = tf.maximum(0., mask_quadrilaterals[:, 7])

            mask_quadrilaterals = tf.stack([cliped_y1,
                                            cliped_x1,
                                            cliped_y2,
                                            cliped_x2,
                                            cliped_y3,
                                            cliped_x3,
                                            cliped_y4,
                                            cliped_x4], axis=-1)

            return tf.cond(
                tf.logical_or(
                    tf.less(roi_slice_range[2], 1),
                    tf.less(roi_slice_range[3], 1)),
                lambda: (image, labels, bboxes, quadrilaterals),
                lambda: (tf.slice(image,
                                  [roi_slice_range[0],
                                   roi_slice_range[1],
                                   0],
                                  [roi_slice_range[2],
                                   roi_slice_range[3],
                                   -1]),
                         mask_labels, mask_bboxes, mask_quadrilaterals))

    with tf.name_scope('random_sample_patch'):
        image = tf.convert_to_tensor(image, name='image')

        min_iou_list = tf.convert_to_tensor(ratio_list)
        samples_min_iou =\
            tf.multinomial(
                logits=tf.log([[1. / len(ratio_list)] * len(ratio_list)]),
                num_samples=1)
        # samples_min_iou 2d-tf.Tensor-(batch_size=1,
        #                               num_samples=1)
        # logits 2d-tf.Tensor-(batch_size=1,
        #                      num_classes=len(ratio_list))
        # samples_min_iou[i, j] is a 0d-tf.Tensor-() containing index of a
        # ratio in ratio_list. In this case, we just need one value from
        # ratio_list. So, we choose batch_size=1 and num_samples=1.

        sampled_min_iou =\
            min_iou_list[tf.cast(samples_min_iou[0][0], tf.int32)]
        # sampled_min_iou contains a ratio drawn equally randomly from
        # ratio_list

        return tf.cond(
            tf.less(sampled_min_iou, 1.),
            lambda: sample_patch(image,
                                 labels,
                                 bboxes,
                                 quadrilaterals,
                                 sampled_min_iou),
            lambda: (image, labels, bboxes, quadrilaterals))


def unwhiten_image(image, output_rgb=True):
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    if not output_rgb:
        image_channels = tf.unstack(image, axis=-1, name='split_bgr')
        image = tf.stack([image_channels[2],
                          image_channels[1],
                          image_channels[0]],
                         axis=-1,
                         name='merge_rgb')
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2,
                        num_or_size_splits=num_channels,
                        value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(axis=2, values=channels)
