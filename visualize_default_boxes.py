from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

import tensorflow as tf
import numpy as np
import cv2

from config import textboxes_plusplus_config as config
from utility import drawing_toolbox
from utility import anchor_manipulator

tf.app.flags.DEFINE_integer(
    'train_image_size',
    config.TRAIN_IMAGE_SIZE,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'chosen_feature_map',
    5,
    'Chosen feature map to visualize.')
tf.app.flags.DEFINE_string(
    'image_path',
    './demo/demo.png',
    'The path where image is located.')

FLAGS=tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def main(_):
	target_shape=[FLAGS.train_image_size] * 2

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
	fm=FLAGS.chosen_feature_map
	anchors_ymin,\
	anchors_xmin,\
	anchors_ymax,\
	anchors_xmax=\
		anchor_processor.get_all_anchors_one_layer(
			anchor_heights_all_layers[fm],
			anchor_widths_all_layers[fm],
			num_anchors_per_location_all_layers[fm],
			config.ALL_LAYER_SHAPES[fm],
			config.ALL_LAYER_STRIDES[fm],
			config.ANCHOR_OFFSETS[fm],
			config.VERTICAL_OFFSETS[fm],
			name=None)

	with tf.Session() as sess:
		# shape=(num_anchor_locations_per_feature_map, num_anchors_per_location).
		anchors_ymin,\
		anchors_xmin,\
		anchors_ymax,\
		anchors_xmax=\
			sess.run([anchors_ymin,
					  anchors_xmin,
					  anchors_ymax,
					  anchors_xmax])

		input_image=cv2.imread(FLAGS.image_path)
		input_image=cv2.resize(
			input_image,
			tuple(target_shape),
			interpolation=cv2.INTER_AREA)

		grid_drawed_image=draw_grid(
			image=input_image.copy(),
			grid_shape=config.ALL_LAYER_SHAPES[fm],
			color=(0, 0, 0))
		cv2.imshow('grid_drawed_image', grid_drawed_image)

		num_anchor_locations_per_feature_map=len(anchors_ymin)
		num_anchors_per_location=len(anchors_ymin[0])

		location_index=int(random.random() * num_anchor_locations_per_feature_map)
		anchor_index=int(random.random() * num_anchors_per_location)

		location_index=4
		anchor_begin_index=0
		anchor_end_index=4

		anchors_ymin=anchors_ymin[location_index][anchor_begin_index:anchor_end_index]
		anchors_xmin=anchors_xmin[location_index][anchor_begin_index:anchor_end_index]
		anchors_ymax=anchors_ymax[location_index][anchor_begin_index:anchor_end_index]
		anchors_xmax=anchors_xmax[location_index][anchor_begin_index:anchor_end_index]

		# anchor=[anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax]
		anchors=[anchor for anchor in map(list, zip(*[anchors_ymin,
													  anchors_xmin,
													  anchors_ymax,
													  anchors_xmax]))]

		anchor_drawed_image=draw_anchors(
			grid_drawed_image.copy(),
			anchors)
		cv2.imshow('anchor_drawed_image', anchor_drawed_image)

		cv2.imwrite('grid_' + str(FLAGS.chosen_feature_map) + '.jpg', grid_drawed_image)
		cv2.imwrite('grid_' + str(FLAGS.chosen_feature_map) + '_with_anchors' + '.jpg', anchor_drawed_image)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

def draw_anchors(image, anchors):
	for anchor in anchors:
		image=draw_anchor(
			image,
			anchor,
			color=drawing_toolbox.get_color())
	return image

def draw_anchor(image, anchor, color):
	[anchor_ymin,
	 anchor_xmin,
	 anchor_ymax,
	 anchor_xmax]=anchor
	cv2.rectangle(image,
				  (anchor_xmin, anchor_ymin),
				  (anchor_xmax, anchor_ymax),
				  color=color,
				  thickness=2)
	return image

def draw_grid(image, grid_shape, color):
	grid_height,\
	grid_width=\
		grid_shape[0],\
		grid_shape[1]
	image_height,\
	image_width=\
		image.shape[:2]
	list_of_h_ends=list(np.linspace(
		0,
		image_height-1,
		grid_height+1,
		dtype=np.int32))
	list_of_v_ends=list(np.linspace(
		0,
		image_width-1,
		grid_width+1,
		dtype=np.int32))
	for h_end_index in np.arange(1, grid_height):
		cv2.line(image,
				 (0, list_of_h_ends[h_end_index]),
				 (image_width-1, list_of_h_ends[h_end_index]),
				 color=color,
				 thickness=2,
				 lineType=8)
	for v_end_index in np.arange(1, grid_width):
		cv2.line(image,
				 (list_of_v_ends[v_end_index], 0),
				 (list_of_v_ends[v_end_index], image_height-1),
				 color=color,
				 thickness=2,
				 lineType=8)
	return image

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()