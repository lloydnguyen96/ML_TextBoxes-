from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import six
import numpy as np
from pathlib import Path
import threading
import json
from datetime import datetime
import sys
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

LABELS_TO_NAMES = {
    'none': (0, 'Background'),
    'chinesetext': (1, 'Text')
}

tf.app.flags.DEFINE_string(
    'input_image_root',
    'dataset/rctw/RCTW_p3',
    'Where to look for input images')
tf.app.flags.DEFINE_string(
    'input_image_stem_patterns',
    'icdar2017rctw_train_v1.2/**/*.jpg',
    'Comma-separated list of stem patterns in root to look for leaves '
    '(images)')
tf.app.flags.DEFINE_string(
    'input_annotation_root',
    'dataset/rctw/RCTW_p3',
    'Where to look for input annotations')
tf.app.flags.DEFINE_string(
    'input_annotation_stem_patterns',
    'json_annotations/**/*.json',
    'Comma-separated list of stem patterns in root to look for leaves '
    '(annotations)')
tf.app.flags.DEFINE_string(
    'output_directory',
    './dataset/rctw/RCTW_p3/tfrecords',
    'Output dataset directory (.tfrecord files)')
tf.app.flags.DEFINE_string(
    'output_leaf_prefix',
    'train',
    'prefix name of leaf (.tfrecord files)')
tf.app.flags.DEFINE_integer(
    'num_shards',
    144,
    'Number of shards (.tfrecord files) created by all threads')
tf.app.flags.DEFINE_integer(
    'num_threads',
    6,
    'Number of threads to preprocess the images. It needs to commensurate '
    'with number of shards')
# RANDOM_SEED = 20190511
RANDOM_SEED = 20200812

FLAGS = tf.app.flags.FLAGS


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg =\
            tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb =\
            tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg =\
            tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg,
            feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(
            self._cmyk_to_rgb,
            feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg,
            feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def convert_dataset_to_tfrecords():
    """Convert a dataset and save it as TFRecord files."""
    def assign_jobs_to_thread(job, job_requirements):
        thread = threading.Thread(target=job, args=job_requirements)
        return thread

    def convert_to_example(image_path,
                           annotation_path,
                           coder=ImageCoder()):

        def process_image(image_path, coder=ImageCoder()):
            # Read the image file.
            with tf.gfile.FastGFile(image_path, 'rb') as f:
                image_data = f.read()

            # Decode the RGB JPEG
            image = coder.decode_jpeg(image_data)

            # Check that image converted to RGB
            assert(len(image.shape) == 3)
            assert(image.shape[2] == 3)
            height = image.shape[0]
            width = image.shape[1]
            assert height > 0
            assert width > 0

            return image_data, height, width

        image_data, height, width =\
            process_image(image_path, coder)

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

            # image shape
            size = annotation['size']
            shape = size
            height, width, _ = shape
            file_name = annotation['file_name'].encode('utf-8')

            # find annotations
            bboxes = []
            labels = []
            labels_text = []
            difficult = []
            quadrilaterals = []
            for text_line in annotation['text_lines']:
                label = text_line['text'].strip()
                # LABELS_TO_NAMES is a dict.
                # LABELS_TO_NAMES[label] is a value in a dict with key = label.
                # This value is a tuple. Tuple's first element is number
                # indicating the label.
                if label == 'none':
                    label = 'none'
                else:
                    label = 'chinesetext'
                labels_text.append(label.encode('utf-8'))
                labels.append(int(LABELS_TO_NAMES[label][0]))

                difficult.append(text_line['difficult'])

                bbox = text_line['standing_bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w - 1
                ymax = ymin + h - 1

                bboxes.append((float(ymin),
                               float(xmin),
                               float(ymax),
                               float(xmax)))

                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] =\
                    text_line['quadrilateral']

                x1 = float(x1)
                x2 = float(x2)
                x3 = float(x3)
                x4 = float(x4)
                y1 = float(y1)
                y2 = float(y2)
                y3 = float(y3)
                y4 = float(y4)

                quadrilaterals.append((x1, x2, x3, x4,
                                       y1, y2, y3, y4))
        # shape = list of three integer numbers height, width, channel.
        # bboxes = list of tuples of four points.
        # labels = list of class indices.

        # pylint: disable=expression-not-assigned
        ymin, xmin, ymax, xmax = map(list, zip(*bboxes))
        x1, x2, x3, x4,\
            y1, y2, y3, y4 = map(list, zip(*quadrilaterals))
        # pylint: enable=expression-not-assigned

        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(image_format),
            'image/shape': int64_feature(shape),
            'image/file_name': bytes_feature(file_name),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/quadrilateral/x1': float_feature(x1),
            'image/object/quadrilateral/y1': float_feature(y1),
            'image/object/quadrilateral/x2': float_feature(x2),
            'image/object/quadrilateral/y2': float_feature(y2),
            'image/object/quadrilateral/x3': float_feature(x3),
            'image/object/quadrilateral/y3': float_feature(y3),
            'image/object/quadrilateral/x4': float_feature(x4),
            'image/object/quadrilateral/y4': float_feature(y4),
            'image/object/label': int64_feature(labels),
            'image/object/label_text': bytes_list_feature(labels_text),
            'image/object/difficult': int64_feature(difficult)}))
        return example

    def write_serialized_image_examples_using_one_thread(
            thread_index,
            pairs_in_thread):
        """Write TFRecord files manipulated by one thread."""
        num_shards_per_thread = len(pairs_in_thread)
        thread_num_pairs =\
            sum([len(pairs_in_shard) for pairs_in_shard in pairs_in_thread])

        thread_num_processed_images = 0
        for i in tqdm(range(num_shards_per_thread)):
            shard_index_dataset = thread_index * num_shards_per_thread + i

            output_leaf_name =\
                '%s-%.5d-of-%.5d.tfrecord' % (FLAGS.output_leaf_prefix,
                                              shard_index_dataset,
                                              FLAGS.num_shards)
            output_path =\
                Path(FLAGS.output_directory).resolve().joinpath(
                    output_leaf_name)
            with tf.python_io.TFRecordWriter(str(output_path)) as writer:
                shard_num_processed_images = 0
                for image_path, annotation_path in tqdm(pairs_in_thread[i]):
                    assert image_path.stem == annotation_path.stem

                    example = convert_to_example(str(image_path),
                                                 str(annotation_path))
                    writer.write(example.SerializeToString())

                    thread_num_processed_images += 1
                    shard_num_processed_images += 1
                    if not thread_num_processed_images % 1000:
                        print('{} [thread {}]: Processed {} of {} '
                              'images'.format(
                                  datetime.now(),
                                  thread_index,
                                  thread_num_processed_images,
                                  thread_num_pairs))
                        sys.stdout.flush()
    # Each thread must generate an equal number of shards (files).
    assert not FLAGS.num_shards % FLAGS.num_threads
    print('Saving results to {}'.format(FLAGS.output_directory))

    input_image_filepaths =\
        [path
         for pattern in FLAGS.input_image_stem_patterns.split(',')
         for path in sorted(
             Path(FLAGS.input_image_root).resolve().glob(pattern))
         ]
    input_annotation_filepaths =\
        [path
         for pattern in FLAGS.input_annotation_stem_patterns.split(',')
         for path in sorted(
             Path(FLAGS.input_annotation_root).resolve().glob(pattern))
         ]
    pairs = list(zip(input_image_filepaths,
                     input_annotation_filepaths))

    ends = [int(e)
            for e in np.linspace(0, len(pairs), FLAGS.num_shards + 1)]
    pairs_in_shards = [pairs[ends[i]:ends[i+1]]
                       for i in range(FLAGS.num_shards)]

    ends = [int(e)
            for e in np.linspace(0,
                                 len(pairs_in_shards),
                                 FLAGS.num_threads + 1)]
    pairs_in_threads = [pairs_in_shards[ends[i]:ends[i+1]]
                        for i in range(FLAGS.num_threads)]

    # Coordinator manages the work of threads.
    coord = tf.train.Coordinator()
    threads = []
    for i in range(FLAGS.num_threads):
        job_requirements = [i,
                            pairs_in_threads[i]]
        thread =\
            assign_jobs_to_thread(
                job=write_serialized_image_examples_using_one_thread,
                job_requirements=job_requirements)
        thread.start()
        threads.append(thread)

    # Wait for all the threads to terminate.
    coord.join(threads)


def main(unused_argv):
    convert_dataset_to_tfrecords()


if __name__ == '__main__':
    tf.app.run()
