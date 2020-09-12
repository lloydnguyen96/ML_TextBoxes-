from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

LABELS_TO_NAMES = {
    'none': (0, 'Background'),
    'chinesetext': (1, 'Text')
}

# CTWD_p1
# NUM_IMAGES_OF_SPLIT = {
#     'train': 19901,
#     'val': 1496,
# }

# RCTW-7_raw (RCTW_p2)
# NUM_IMAGES_OF_SPLIT = {
#     'train': 8034,
#     'val': 4000,  # not yet updated
# }


# RCTW-7_p3
NUM_IMAGES_OF_SPLIT = {
    # 'train': 40562,
    'train': 37910,
    'val': 4000,  # not yet updated
}


def slim_get_batch(num_classes,
                   batch_size,
                   split_name,  # used to a specific group of .tfrecord files
                   file_pattern,  # the pattern of fullpaths to .tfrecord files
                   num_readers,  # number of parallel .tfrecord readers
                   num_preprocessing_threads,
                   image_preprocessing_fn,
                   anchor_encoding_fn,
                   num_epochs=None,
                   is_training=True):
    """slim_get_batch

    Args:
        num_classes (e.g., 2)
        batch_size (e.g., 16)
        split_name (e.g., 'train')
        file_pattern (e.g., './dataset/ctwd/tfrecords/train-*')
        num_readers (e.g., 8)
        num_preprocessing_threads (e.g., 24)
    """
    if split_name not in NUM_IMAGES_OF_SPLIT:
        raise ValueError('split name {} was not '
                         'recognized.'.format(split_name))

    # Features in CTWD TFRecords.
    # if shape=[1] -> shape=[batch_size, 1] after batching
    # if shape=[] -> shape=[batch_size, ] after batching
    # it means:
    # if encoded value is [5.0] -> decoded value with shape () or [] is 5.0
    # if encoded value is [5.0] -> decoded value with shape [1] is [5.0]
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([],  # or shape=[1] if you want
                                            tf.string,  # additional dimension
                                            default_value=''),
        'image/format': tf.FixedLenFeature([],
                                           tf.string,
                                           default_value='jpeg'),
        'image/file_name': tf.FixedLenFeature([],
                                              tf.string,
                                              default_value=''),
        'image/shape': tf.FixedLenFeature([3],
                                          tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/quadrilateral/y4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/difficult': tf.VarLenFeature(dtype=tf.int64)
    }
    # a proto = an example = a sample

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded',
                                              'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'file_name': slim.tfexample_decoder.Tensor('image/file_name'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'],
            'image/object/bbox/'),
        'object/quadrilateral/x1': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/x1'),
        'object/quadrilateral/x2': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/x2'),
        'object/quadrilateral/x3': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/x3'),
        'object/quadrilateral/x4': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/x4'),
        'object/quadrilateral/y1': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/y1'),
        'object/quadrilateral/y2': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/y2'),
        'object/quadrilateral/y3': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/y3'),
        'object/quadrilateral/y4': slim.tfexample_decoder.Tensor(
            'image/object/quadrilateral/y4'),
        'object/label': slim.tfexample_decoder.Tensor(
            'image/object/label'),
        'object/difficult': slim.tfexample_decoder.Tensor(
            'image/object/difficult')
    }

    # purpose: decode parsed tensors to suitable dtype and shape
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features,
        items_to_handlers)
    # decoder.decode(serialized_example, items) method:
    # 1: example = tf.parse_single_example(serialized_example,
    # keys_to_features) = dict{key: corresponding tensor}
    # 2: build keys_to_tensors = dict{handler key: corresponding tensor} of
    # EACH handler
    # e.g., handler slim.tfexample_decoder.Image needs two keys named
    # 'image/encoded' and 'image/format' so its keys_to_tensors will be
    # {'image/encoded': its tensor, 'image/format': its tensor}
    # 3: return [tensors_to_item(keys_to_tensors), ...]
    # tensors_to_item method:
    # - with handler Image: decode encoded image 0d-tf.Tensor tf.string (store
    # bytes) -> 3d-tf.Tensor tf.uint8
    # - with handler Tensor: reshape tensor if shape is provided
    # - with handler BoundingBox: stack 4 1d-tf.Tensor(num_bboxes, ) tf.float32
    # -> 2d-tf.Tensor(num_bboxes, 4) tf.float32
    # !!! Want to know how to parse a serialize example stored in .tfrecord
    # file -> let's inspect that file with:
    # for example in tf.python_io.tf_record_iterator('filename.tfrecord'):
    #       print(tf.train.Example.FromString(example))

    labels_to_names = {}
    for name, pair in LABELS_TO_NAMES.items():
        labels_to_names[pair[0]] = name
    # labels_to_names = {0: 'none', 1: 'chinesetext'}

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=NUM_IMAGES_OF_SPLIT[split_name],
        items_to_descriptions=None,
        num_classes=num_classes,
        labels_to_names=labels_to_names)
    # dataset is just an Python object with __dict__ attribute filled up with
    # passed arguments.
    # e.g., dataset.__dict__ =
    # {'data_sources': './dataset/ctwd/tfrecords/train-*',
    #  'reader': tf.TFRecordReader,
    #  'decoder': slim.tfexample_decoder.TFExampleDecoder(...),
    #  'num_samples': 19901,
    #  'items_to_descriptions: None,
    #  'num_classes': 2,
    #  'labels_to_names': {0: 'none', 1: 'chinesetext'}
    #  }

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,  # 8
            common_queue_capacity=32*batch_size,
            common_queue_min=8*batch_size,
            shuffle=is_training,
            num_epochs=num_epochs  # not used
        )
        # name_scope: dataset_data_provider
        # I: (key, data) = parallel_reader.parallel_read
        #   I1: data_files = [fullpath to .tfrecord, fullpath to .tfrecord,
        #   ...]
        #   I2: name_scope: dataset_data_provider/parallel_read
        #   I-: filename_queue =
        #           tf_input.string_input_producer(
        #               data_files,
        #               seed=None,
        #               shuffle=True,
        #               capacity=32)
        #       I21: name_scope:
        #       dataset_data_provider/parallel_read/input_producer
        #       I22: convert data_files from Python <class 'str'> to tf.Tensor
        #       I23: create identity Tensor of data_files to pass to
        #       input_producer
        #       I24: call input_producer
        #           I241: name_scope: dataset_data_provider/
        #                             parallel_read/
        #                             input_producer/
        #                             input_producer
        #           I242: create one random_shuffle op
        #           I243: create one FIFOQueue with capacity=32,
        #           dtypes=[dtype, dtype, ...], shapes=[shape, shape, shape]
        #           I---: dtypes: type of a queue element, dtype: type of a
        #               tensor
        #           I---: shapes: shape of a queue element, shape: shape of a
        #               tensor
        #           I244: create one enqueue_many op:
        #               enq = q.enqueue_many([input_tensor])
        #           I---: add as many elements as possible to FIFOQueue
        #       I25: filename_queue is this FIFOQueue. First shuffle 128
        #       .tfrecord filepaths. Then, add as many elements as possible to
        #       this queue (capacity=32 elements, each element is a filepath)
        #       until run out of filepaths. Then, shuffle another identical
        #       list of 128 .tfrecord filepaths, etc.
        #   I3: create one RandomShuffleQueue if shuffle=True else one
        #   FIFOQueue (common_queue)
        #   I-: capacity = common_queue_capacity
        #   I-: common_queue_min used only for RandomShuffleQueue (the minimum
        #   number of element remained in Queue after dequeue and dequeue_many
        #   operation)
        #   I-: Queue element: types [tf.string, tf.string], shapes
        #   unspecified so each element may have different shape but the use
        #   of dequeue_many will be disallowed. However, len(types) == 2 and
        #   must be equal to len(shapes)
        #   I4: return ParallelReader(
        #               reader_class=tf.TFRecordReader,
        #               common_queue,
        #               num_readers=8).read(filename_queue)
        #       I41: create readers = list of 8 (num_readers)
        #       tf.TFRecordReader instances
        #       I42: call read method of each tf.TFRecordReader which create
        #       (does it actually create???) dequeue op for filename_queue and
        #       from one filename_queue element (filepath) return one tuple
        #       (key string scalar Tensor, value string scalar Tensor) example
        #       each time in which value is the serialized data stored in
        #       .tfrecord file. This (k, v) tuple is a common_queue element
        #       I43: create num_readers (8) enqueue ops for common_queue
        #       I44: create one dequeue op for common_queue
        #       I--: return this op (dequeue element in random order for
        #       RandomShuffleQueue)
        #       I--: (key, data) is (k, v)
        # II: tensors = dataset.decoder.decode(data, items) in which items =
        # all items from items_to_handlers
        # --: tensors = [decoded Tensor from parsed Tensor,
        #                decoded Tensor from parsed Tensor, ...] for one
        #                example
        # --: items = [item1, item2, ...] for one example
        # III: build items_to_tensors dict
        # {item1: tensor1, item2: tensor2, ...} for one example
        # IV: add one more element to items_to_tensors: items_to_tensors =
        # {'record_key': key, item1: tensor1, ...}

    # Dataflow: Writing process in ctwd_to_tfrecords.py
    # An image in filesystem (encoded as .jpg)
    #                   |
    #                  \/
    # image_data object (read in ctwd_to_tfrecords.py/process_image function)
    # type <class 'bytes'>
    #                   |
    #                  \/
    # object1 = tf.train.BytesList(value=[image_data])
    # type <class 'tensorflow.core.example.feature_pb2.BytesList'>
    #                   |
    #                  \/
    # object2 = tf.train.Feature(bytes_list=object1)
    # type <class 'tensorflow.core.example.feature_pb2.Feature'>
    #                   |
    #                  \/
    # object3 = features =
    #   tf.train.Features(feature={'image/encoded': object2})}
    # type <class 'tensorflow.core.example.feature_pb2.Features'>
    #                   |
    #                  \/
    # example = tf.train.Example(features=object3)
    # type <class 'tensorflow.core.example.example_pb2.Example'>
    #                   |
    #                  \/
    # serialized_example = example.SerializeToString()
    # type <class 'bytes'>
    # *** NOTE: Why do we have to convert image_data (type <class 'bytes'>) to
    # serialized_example with the same type <class 'bytes'>???
    # ==> Because we can leverage useful functions TensorFlow build: using
    # keys to retrieve appropriate data, build data pipeline with tf.data
    # or with QueueRunner below
    # ***

    # Dataflow: Reading in dataset_common.py
    # .tfrecord files in file system
    #                   |
    #                  \/
    # data_files = ['full path to .tfrecord file', ...]
    # Python <class 'list'> of <class 'str'>s
    #                   |
    #                  \/
    # (key, value) = (key, serialized_example) =
    # tf.TFRecordReader().read(filename_queue)
    # key's type Tensor("ReaderReadV2:0", shape=(), dtype=string). key value
    # is a string "full path to .tfrecord file:index of record"
    # serialized_example's type Tensor("ReaderReadV2:1",
    #                                  shape=(),
    #                                  dtype=string)
    # serialized_example value is bytes representation of read record
    # *** NOTE: serialized_example that we write to filesystem is of type
    # <class 'bytes'> instead of type Tensor. We use type Tensor because:
    # - We don't want to move back and forth, between Python (feed_dict, data)
    # and the underlying C++ wrapper (TF graph, ...)
    # - We want to build a data pipeline using queue which lie right on
    # TensorFlow graph that directly connect dataset (tensors) to network
    # graph. It will make use of multithreading and asynchronicity
    # ***
    # *** NOTE: first queue ==> filename_queue
    # - filename_queue element is 0d-Tensor tf.string containing full path to
    # one .tfrecord file
    # - tf.TFRecordReader.read(filename_queue) will:
    # + read a record (key, value) each time until the file that containing
    # that record running out of records
    # + create one dequeue op to dequeue that element from filename_queue
    # + we actually have a number of tf.TFRecordReader instances which read
    # concurrently records from filename_queue
    # ***
    # *** NOTE:
    # filename_queue:
    # - convert data_files from <class 'str'> to
    #   1d-Tensor(num_files,)-tf.string
    # - create one FIFOQueue:
    #   + capacity: the maximum number of elements this queue can contain
    # - create one enqueue op:
    #   + shuffle data_files (if shuffle is True) with seed provided (or
    #   default value if not provided) before enqueuing (each data_files will
    #   be shuffled only once)
    #   + num_epochs: queueuing data_files num_epochs times (if provided) or
    #   unlimited number of times if not
    # - tf.train.add_queue_runner(
    #       tf.train.QueueRunner(
    #           FIFOQueue,
    #           [enqueue_op] * numOfThreads
    #       )  # numOfThreads = 1
    #   )
    # + each enqueue_op (not dequeue_op and Queue) is operated by one thread
    # + dequeue_op will be operated whenever our session flow tensor to the
    # output of that queue (it means calling sess.run(tensor) that eventually
    # need some values from queue)
    # + enqueue_op and queue are managed by QueueRunner objects
    # ***
    # *** NOTE: Multithreading in Python and TensorFlow
    # - coord = tf.train.Coordinator(): coordinate the termination of a set of
    # threads
    # + coord.request_stop() are called by a thread. It will request the other
    # threads to stop by switching the return value of coord.should_stop() from
    # False to True
    # + coord.should_stop() is used by all threads to check whether or not they
    # should keep executing instructions
    # + coord.join(threads) is called by a thread to wait for threads to
    # terminate
    # - thread:
    # + in Python: created by threading.Thread(
    #                           target=function_executed_by_thread,
    #                           args=arguments_of_that_function)
    # + in TF: created in QueueRunner object
    # + In Python, we collect threads manually in a Python <class 'list'>:
    # [thread1, thread2, ...]. In TF, we collect threads by collecting
    # QueueRunner objects with tf.train.add_queue_runner(QRobjects)
    # + In Python, we call thread.start() to run each thread. In TF, executing
    # tf.train.start_queue_runners(coord=coord) will start all threads
    # collected above and return threads list
    # - Programming as follows:
    # + create Coordinator coord object
    # + create threads and start each of them coordinated by coord
    # + optionally use coord.request_stop, coord.should_stop and coord.join
    # ***
    # *** NOTE: second queue ==> common_queue
    # - common_queue element: (key, serialized_example) record
    # - common_queue type:
    # + RandomShuffleQueue: a queue that dequeues elements in random order.
    # Read documentation for detail information
    # + FIFOQueue: not used
    # - create num_readers enqueue_op operations (num_readers = number of
    # tf.TFRecordReader instances)
    # - create one dequeue_op
    # ***
    #                   |
    #                  \/
    # example_features = tf.parse_single_example(serialized_example,
    #                                            keys_to_features)
    # type python dict {key: corresponding TF Object}
    # e.g., {'image/object/bbox/xmax':
    # <tensorflow.python.framework.sparse_tensor.SparseTensor object at ...>,
    # ...}
    #                   |
    #                  \/
    # e.g., example_features['image/encoded'] (encoded_image)
    # ==> decoded_image = tf.decode_jpeg(encoded_image, channels=3)
    # type Tensor("DecodeJpeg:0", shape=(?, ?, 3), dtype=uint8)"
    # - This process and other stuffs are done in handler
    # slim.tfexample_decoder.Image
    # - same for the other features
    # - both example_features and decoded tensors are retrieved using
    # slim.tfexample_decoder.TFExampleDecoder(...)

    [org_image,
     file_name,
     shape,
     glabels_raw,
     gbboxes_raw,
     gquadrilaterals_x1,
     gquadrilaterals_x2,
     gquadrilaterals_x3,
     gquadrilaterals_x4,
     gquadrilaterals_y1,
     gquadrilaterals_y2,
     gquadrilaterals_y3,
     gquadrilaterals_y4,
     isdifficult] =\
        provider.get([
            'image',
            'file_name',
            'shape',
            'object/label',
            'object/bbox',
            'object/quadrilateral/x1',
            'object/quadrilateral/x2',
            'object/quadrilateral/x3',
            'object/quadrilateral/x4',
            'object/quadrilateral/y1',
            'object/quadrilateral/y2',
            'object/quadrilateral/y3',
            'object/quadrilateral/y4',
            'object/difficult'])
    # provider.get(items)
    # ==> return [self._items_to_tensors[item] for item in items]

    gquadrilaterals_raw = tf.stack([gquadrilaterals_y1,
                                    gquadrilaterals_x1,
                                    gquadrilaterals_y2,
                                    gquadrilaterals_x2,
                                    gquadrilaterals_y3,
                                    gquadrilaterals_x3,
                                    gquadrilaterals_y4,
                                    gquadrilaterals_x4],
                                   axis=1)
    # org_image: 3d-tf.Tensor-(?, ?, 3)-uint8
    # file_name 0d-tf.Tensor-()-string
    # shape 1d-tf.Tensor-(3,)-int64
    # glabels_raw 1d-tf.Tensor-(?,)-int64
    # gbboxes_raw 2d-tf.Tensor-(?, 4)-float32
    # gquadrilaterals_x1 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_x2 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_x3 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_x4 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_y1 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_y2 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_y3 1d-tf.Tensor-(?,)-float32
    # gquadrilaterals_y4 1d-tf.Tensor-(?,)-float32
    # isdifficult 1d-tf.Tensor-(?,)-int64
    # gquadrilaterals_raw 2d-tf.Tensor-(?, 8)-float32

    """ USE BOTH DIFFICULT and NONDIFFICULT EXAMPLES
    if is_training:
        # if all is difficult, then keep the first one
        isdifficult_mask =\
            tf.cond(
                # if exists at least one non-difficult object
                tf.count_nonzero(isdifficult,
                                 dtype=tf.int32) < tf.shape(isdifficult)[0],
                # then create mask to keep non-difficult objects
                lambda: isdifficult < tf.ones_like(isdifficult),
                # else, select the first difficult object
                lambda: tf.one_hot(0,
                                   tf.shape(isdifficult)[0],
                                   on_value=True,
                                   off_value=False,
                                   dtype=tf.bool))
        # isdifficult_mask.get_shape():  (n_objects_per_image,)
        # isdifficult_mask[x] is True: object with index x is not difficult.
        # isdifficult_mask[x] is False: otherwise.

        # Keep easy-to-detect objects using isdifficult_mask.
        glabels_raw = tf.boolean_mask(glabels_raw, isdifficult_mask)
        gbboxes_raw = tf.boolean_mask(gbboxes_raw, isdifficult_mask)
        gquadrilaterals_raw = tf.boolean_mask(gquadrilaterals_raw,
                                              isdifficult_mask)
    """

    # preprocess image, labels, bboxes and quadrilaterals
    tensors_to_batch = []
    if is_training:
        image, glabels, gbboxes, gquadrilaterals =\
            image_preprocessing_fn(org_image,
                                   glabels_raw,
                                   gbboxes_raw,
                                   gquadrilaterals_raw)
        gt_targets, gt_labels, gt_scores =\
            anchor_encoding_fn(glabels,
                               gbboxes,
                               gquadrilaterals)
        tensors_to_batch =\
            [image, file_name, shape, gt_targets, gt_labels, gt_scores]
    else:
        image, gbboxes, gquadrilaterals =\
            image_preprocessing_fn(org_image,
                                   glabels_raw,
                                   gbboxes_raw,
                                   gquadrilaterals_raw)
        gt_targets, gt_labels, gt_scores =\
            anchor_encoding_fn(glabels_raw,
                               gbboxes,
                               gquadrilaterals)
        tensors_to_batch =\
            [image, file_name, shape, gt_targets, gt_labels, gt_scores]
    # dataset bbox, quadrilateral structure:
    # - json: bbox [xmin, ymin, w, h]
    #         quadrilateral [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # - tfrecord: discrete as xmin, ymin, xmax, ymax, x1, x2, ..., y1, y2, ...
    # - after loading:
    #         bbox [ymin, xmin, ymax, xmax]
    #         quadrilateral [y1, x1, y2, x2, ..., y4, x4]
    # - after encoding:
    #         target = bbox + quadrilateral =
    #         [cy', cx', h', w', y1', x1', y2', x2', ..., y4', x4']
    #         "'" means decoded value

    # create PaddingFIFOQueue if dynamic_pad else FIFOQueue
    # create num_preprocessing_threads enqueue_op(s)
    # enqueue_op = enqueue_many if enqueue_many else enqueue (default)
    # create batch_size dequeue_op(s)
    # dequeue_op = dequeue_many if not allow_smaller_final_batch else
    # dequeue_up_to
    # create QueueRunner object and add it to queue runner's collection
    return tf.train.batch(
        tensors_to_batch,
        dynamic_pad=(not is_training),
        batch_size=batch_size,
        allow_smaller_final_batch=(not is_training),
        num_threads=num_preprocessing_threads,
        capacity=64 * batch_size)
