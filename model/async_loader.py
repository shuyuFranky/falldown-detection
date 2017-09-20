from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def get_multi_scale_crop_size(height, width, crop_size, scale_ratios, max_distort):
    crop_sizes = []
    base_size = min(height, width)
    for i in xrange(len(scale_ratios)):
        crop_h = int(base_size * scale_ratios[i])
        if abs(crop_h - crop_size) < 3:
            crop_h = crop_size
        for j in xrange(len(scale_ratios)):
            crop_w = int(base_size * scale_ratios[j])
            if abs(crop_w - crop_size) < 3:
                crop_w = crop_size
            # append this cropping size into the list
            if abs(j - i) <= max_distort:
                crop_sizes.append([crop_h, crop_w])
    return crop_sizes

def get_fix_offset(h, w, crop_height, crop_width):
    crop_offsets = []
    height_off = (h - crop_height) / 4
    width_off = (w - crop_width) / 4
    crop_offsets.append(tf.stack([0, 0]))
    crop_offsets.append(tf.stack([0, tf.to_int32(4 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(4 * height_off), 0]))
    crop_offsets.append(tf.stack([tf.to_int32(4 * height_off), tf.to_int32(4 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(2 * height_off), tf.to_int32(2 * width_off)]))
    # more fix crop
    crop_offsets.append(tf.stack([0, tf.to_int32(2 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(4 * height_off), tf.to_int32(2 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(2 * height_off), 0]))
    crop_offsets.append(tf.stack([tf.to_int32(2 * height_off), tf.to_int32(4 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(height_off), tf.to_int32(width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(height_off), tf.to_int32(3 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(3 * height_off), tf.to_int32(width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(3 * height_off), tf.to_int32(3 * width_off)]))

    crop_offsets = tf.stack(crop_offsets)
    return crop_offsets

def one_image(modality, image_name, offset_str, start_offset, height, width, crop,
              ho, wo, crop_size, crop_height, crop_width, preprocessing_fn, random_mirror, length=1):
    channels = 3 * length
    if modality is None:
        id = "0"
        file_contents = tf.read_file(image_name)
        image = tf.image.decode_jpeg(file_contents, channels=3)
    elif modality == 'RGB':
        images = []
        for o in xrange(length):
            id = tf.gather(offset_str, start_offset+o)
            file_contents = tf.read_file(image_name+"/flow_i_"+id+".jpg")
            image = tf.image.decode_jpeg(file_contents, channels=3)
            images.append(image)
        image = tf.concat(images, 2)
    elif modality == 'flow' or modality == 'warp':
        images = []
        for o in xrange(length):
            id = tf.gather(offset_str, start_offset+o)
            file_contents = tf.read_file(image_name+"/flow_x_"+id+".jpg")
            image1 = tf.image.decode_jpeg(file_contents, channels=1)
            image1 = tf.to_float(image1)
            file_contents = tf.read_file(image_name+"/flow_y_"+id+".jpg")
            image2 = tf.image.decode_jpeg(file_contents, channels=1)
            image2 = tf.to_float(image2)
            if length <= 1:
                image3 = 0.7064*tf.sqrt(image1*image1+image2*image2)
                image = tf.concat([image1, image2, image3], 2)
            else:
                image = tf.concat([image1, image2], 2)
            images.append(image)
        image = tf.concat(images, 2)
        if length > 1:
            channels = 2 * length
    else:
        raise NotImplementedError('Modality %s is not supported.'%modality)
    image = tf.image.resize_images(image, [height, width], method=0)
    image.set_shape([height, width, channels])
    if crop == 0:
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image.set_shape([crop_size, crop_size, channels])
    elif crop == 1 or crop == 2:
        image = tf.slice(image, tf.stack([ho, wo, 0]), tf.stack([crop_height, crop_width, -1]))
    else:
        raise NotImplementedError('Crop mode %d is not supported.'%crop)
    # augment after crop
    image = preprocessing_fn(image, crop_size, crop_size, random_mirror=random_mirror)

    return image

def read_one_image(name_label_queue, multi_scale_crop_sizes, config):
    # name: video name
    # label: video label
    # multi_scale_crop_sizes: the multi scale crop sizes, select one to crop and resize to config['crop_size']
    # config:
    #   width, height, crop_size, n_steps, modality
    #   crop(0 for center crop, 1 for random crop, 2 for fix crop)
    #   augment(True, False)
    image_name = name_label_queue[0]
    label = name_label_queue[1]
    label = tf.to_int32(label)

    # crop
    crop_index = tf.random_uniform((), maxval=multi_scale_crop_sizes.get_shape()[0].value, dtype=tf.int32)
    crop_height = tf.gather(tf.gather(multi_scale_crop_sizes, crop_index), 0)
    crop_width = tf.gather(tf.gather(multi_scale_crop_sizes, crop_index), 1)
    if config['crop'] == 0:
        ho = None
        wo = None
    elif config['crop'] == 1:
        ho = tf.random_uniform((), maxval=config['height']-crop_height+1, dtype=tf.int32)
        wo = tf.random_uniform((), maxval=config['width']-crop_width+1, dtype=tf.int32)
    elif config['crop'] == 2:
        fix_offsets = get_fix_offset(int(config['width']), int(config['height']), crop_height, crop_width)
        offset_index = tf.random_uniform((), maxval=fix_offsets.get_shape()[0].value, dtype=tf.int32)
        ho = tf.gather(tf.gather(fix_offsets, offset_index), 0)
        wo = tf.gather(tf.gather(fix_offsets, offset_index), 1)
    else:
        raise NotImplementedError('Crop mode %d is not supported.'%config['crop'])

    image = one_image(None, image_name, None, None,
                        config['height'], config['width'], config['crop'],
                        ho, wo, config['crop_size'], crop_height, crop_width,
                        config['preprocessing_fn'], True)

    return image, label

def read_fix_image(name_label_setting_queue, config):
    # name: video name
    # label: video label
    # hos: frame height offsets
    # wos: frame width offsets
    # mirrors: frame mirror (True or False)
    # config:
    #   width, height, crop_size
    #   augment(True, False)
    image_name = name_label_setting_queue[0]
    label = name_label_setting_queue[1]
    ho = name_label_setting_queue[2]
    wo = name_label_setting_queue[3]
    mirror = name_label_setting_queue[4]

    image = one_image(None, image_name, None, None,
                        config['height'], config['width'], 1,
                        ho, wo, config['crop_size'], config['crop_size'], config['crop_size'],
                        config['preprocessing_fn'], False)
    image = tf.cond(mirror, lambda:tf.image.flip_left_right(image), lambda:image)

    return image, label

def image_inputs(groundtruth_path, data_path, batch_size, scale_size, crop_size,
                 preprocessing_fn,
                 shuffle=False, label_from_one=False, crop=0,
                 max_distort=1, scale_ratios=[1,.875,.75,.66]):
    config = {'width':scale_size, 'height':scale_size, 'crop_size':crop_size,
              'crop':crop, 'preprocessing_fn':preprocessing_fn}

    gt_lines = open(groundtruth_path).readlines()
    gt_pairs = [line.split() for line in gt_lines]
    paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
    if len(gt_pairs[0]) == 2:
        labels = np.array([int(p[1]) for p in gt_pairs])
        if label_from_one:
            labels -= 1
    else:
        raise NotImplementedError('Ground truth file should contain one label.')
    dataset_size = len(labels)
    print('%d samples in list.'%len(labels))

    multi_scale_crop_sizes = get_multi_scale_crop_size(scale_size, scale_size,
                                                       crop_size, scale_ratios, max_distort)
    multi_scale_crop_sizes = tf.convert_to_tensor(multi_scale_crop_sizes, dtype=tf.int32)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.slice_input_producer([paths, labels],
                                                    shuffle=shuffle)

    # Read examples from files in the filename queue.
    image, label = read_one_image(filename_queue, multi_scale_crop_sizes, config)

    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = 512
    num_preprocess_threads = 64
    capacity = min_queue_examples + (num_preprocess_threads + 2) * batch_size

    if shuffle:
        images, label_batch, name = tf.train.shuffle_batch(
            [image, label, filename_queue[0]],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, name = tf.train.batch(
            [image, label, filename_queue[0]],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity)

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, crop_size, crop_size, 3])
    # Display the training images in the visualizer.
    tf.summary.image('images', images, max_outputs=batch_size)

    return dataset_size, images, tf.reshape(label_batch, [batch_size]), name

def divided_image_inputs(groundtruth_path, data_path, batch_size, scale_size, crop_size,
                 preprocessing_fn,
                 shuffle=False, label_from_one=False, crop=0,
                 max_distort=1, scale_ratios=[1,.875,.75,.66]):
    config = {'width':scale_size, 'height':scale_size, 'crop_size':crop_size,
              'crop':crop, 'preprocessing_fn':preprocessing_fn}

    gt_lines = open(groundtruth_path).readlines()
    gt_pairs = [line.split() for line in gt_lines]
    paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
    if len(gt_pairs[0]) == 2:
        labels = np.array([int(p[1]) for p in gt_pairs])
        if label_from_one:
            labels -= 1
    else:
        raise NotImplementedError('Ground truth file should contain one label.')
    dataset_size = len(labels)
    print('%d samples in list.'%len(labels))

    CLASS_NUM = np.max(labels) + 1
    class_paths = [[] for i in xrange(CLASS_NUM)]
    for p, l in zip(paths, labels):
        class_paths[l].append(p)

    multi_scale_crop_sizes = get_multi_scale_crop_size(scale_size, scale_size,
                                                       crop_size, scale_ratios, max_distort)
    multi_scale_crop_sizes = tf.convert_to_tensor(multi_scale_crop_sizes, dtype=tf.int32)

    images_output = []
    labels_output = []
    names_output = []
    if FLAGS.last_class_more_sample:
        last_class_times = int(CLASS_NUM/2)
        tmp_batch_size = int(batch_size/(CLASS_NUM + last_class_times - 1))
    else:
        tmp_batch_size = int(batch_size/CLASS_NUM)
    assert np.mod(batch_size, tmp_batch_size) == 0
    for c in xrange(CLASS_NUM):
        if c == CLASS_NUM - 1 and FLAGS.last_class_more_sample:
            tmp_batch_size *= last_class_times
        paths = class_paths[c]
        labels = [c] * len(paths)
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.slice_input_producer([paths, labels],
                                                        shuffle=shuffle)

        # Read examples from files in the filename queue.
        image, label = read_one_image(filename_queue, multi_scale_crop_sizes, config)

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 256
        num_preprocess_threads = 16
        capacity = min_queue_examples + (num_preprocess_threads + 2) * tmp_batch_size

        if shuffle:
            images, label_batch, name = tf.train.shuffle_batch(
                [image, label, filename_queue[0]],
                batch_size=tmp_batch_size,
                num_threads=num_preprocess_threads,
                capacity=capacity,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch, name = tf.train.batch(
                [image, label, filename_queue[0]],
                batch_size=tmp_batch_size,
                num_threads=num_preprocess_threads,
                capacity=capacity)

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[tmp_batch_size, crop_size, crop_size, 3])
        # Display the training images in the visualizer.
        tf.summary.image('images_class_%d'%c, images, max_outputs=tmp_batch_size)
        images_output.append(images)
        labels_output.append(label_batch)
        names_output.append(name)

    images = tf.concat(images_output, 0)
    label_batch = tf.concat(labels_output, 0)
    name = tf.concat(names_output, 0)

    return dataset_size, images, tf.reshape(label_batch, [batch_size]), name

def multi_sample_image_inputs(groundtruth_path, data_path, batch_size,
                            scale_size, crop_size, preprocessing_fn,
                            sample_num=25,
                            label_from_one=False):
    config = {'width':scale_size, 'height':scale_size, 'crop_size':crop_size,
              'preprocessing_fn':preprocessing_fn}

    gt_lines = open(groundtruth_path).readlines()
    gt_pairs = [line.split() for line in gt_lines]
    ori_paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
    if len(gt_pairs[0]) == 2:
        ori_labels = np.array([int(p[1]) for p in gt_pairs])
        if label_from_one:
            ori_labels -= 1
    else:
        raise NotImplementedError('Ground truth file should contain one label.')
    print('%d videos in list.'%len(ori_labels))

    # generate multi sample
    paths = []
    labels = []
    hos = []
    wos = []
    mirrors = []
    crop_off = (scale_size - crop_size) / 2
    crop_pos = [[0, 0],
                [0, int(2*crop_off)],
                [int(crop_off), int(crop_off)],
                [int(2*crop_off), 0],
                [int(2*crop_off), int(2*crop_off)]]
    for i in xrange(len(ori_labels)):
        # 4 corners and center and their flips
        for j in xrange(10):
            paths.append(ori_paths[i])
            labels.append(ori_labels[i])
            hos.append(crop_pos[j%len(crop_pos)][0])
            wos.append(crop_pos[j%len(crop_pos)][1])
            if j >= len(crop_pos):
                mirrors.append(True)
            else:
                mirrors.append(False)

    dataset_size = len(labels)
    print("%d samples in total."%dataset_size)

    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    hos = tf.convert_to_tensor(hos, dtype=tf.int32)
    wos = tf.convert_to_tensor(wos, dtype=tf.int32)
    mirrors = tf.convert_to_tensor(mirrors, dtype=tf.bool)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.slice_input_producer([paths, labels, hos, wos, mirrors],
                                                    num_epochs=1,
                                                    shuffle=False)
    # Read examples from files in the filename queue.
    image, label = read_fix_image(filename_queue, config)

    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = 512
    num_preprocess_threads = 16
    capacity = min_queue_examples + (num_preprocess_threads + 2) * batch_size

    images, label_batch, name = tf.train.batch(
        [image, label, filename_queue[0]],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=capacity)

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, crop_size, crop_size, 3])
    # Display the training images in the visualizer.
    tf.summary.image('images', images, max_outputs=batch_size)

    return dataset_size, images, tf.reshape(label_batch, [batch_size]), name

