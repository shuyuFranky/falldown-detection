# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import math
from datetime import datetime

from tensorflow.python.ops import control_flow_ops
from deployment import model_deploy
from deployment import train_util
from nets import nets_factory
from preprocessing import preprocessing_factory
from collections import Counter

import async_loader

try:
    xrange
except NameError:
    xrange = range
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'mode', 'extract',
    'Run mode. extract.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'trace_every_n_steps', None,
    'The frequency with which the timeline is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

tf.app.flags.DEFINE_boolean(
    'log_device_placement', False,
    """Whether to log device placement.""")

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' "piecewise" or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_string(
    'decay_iteration', '10000',
    'Number of iterations after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_float(
    'grad_clipping', None,
    """Gradient cliping by norm.""")

tf.app.flags.DEFINE_boolean(
    'no_decay', False,
    """Whether decay the learning rate of recovered variables.""")

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_list', '', 'The list of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_boolean(
    'even_input', False, 'Whether use the class split even inputs.')

tf.app.flags.DEFINE_boolean(
    'last_class_more_sample', False, 'Make sure the last class will have more samples.')

tf.app.flags.DEFINE_integer(
    'NUM_CLASSES', 101,
    'The number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'resize_image_size', 256, 'Train image size')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None,
    'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('top_k', 5,
                            """Top k accuracy.""")

tf.app.flags.DEFINE_string(
    'feature_dir', '/tmp/tfmodel/',
    'Directory where features are written to.')

tf.app.flags.DEFINE_float(
    'test_ratio', 1.0,
    'The percentage of dataset for test.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'npy_weights', None,
    'The path to a weights.npy from which to fine-tune.')

tf.app.flags.DEFINE_boolean(
    'no_restore_exclude', False,
    'Prevent checkpoint_exclude_scopes parameters.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_end_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.flags.DEFINE_integer(
    "eval_interval_secs", 1200,
    "Interval between evaluation runs.")

FLAGS = tf.app.flags.FLAGS



def async_extract():
    # Check training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.fatal("Training directory %s not found.", train_dir)
        return

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        ####################
        # Select the network #
        ####################
        network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=FLAGS.NUM_CLASSES,
                    weight_decay=FLAGS.weight_decay,
                    is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                    preprocessing_name,
                    is_training=False)


        test_size, test_data, test_label, test_names = async_loader.multi_sample_image_inputs(FLAGS.dataset_list,
                                            FLAGS.dataset_dir,
                                            FLAGS.batch_size,
                                            FLAGS.resize_image_size,
                                            FLAGS.train_image_size,
                                            image_preprocessing_fn,
                                            sample_num=5,
                                            label_from_one=(FLAGS.labels_offset>0))
        print("Batch size %d"%test_data.get_shape()[0].value)

        batch_size_per_gpu = FLAGS.batch_size
        global_step_tensor = slim.create_global_step()

        # Calculate the gradients for each model tower.
        with tf.device('/gpu:0'):
            probs, end_points = network_fn(test_data)
            top_k_op = tf.nn.in_top_k(probs, test_label, FLAGS.top_k)
        saver = tf.train.Saver()
        init = tf.local_variables_initializer()
        g.finalize()
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)

        tf.logging.info("Starting evaluation at " + time.strftime(
                "%Y-%m-%d-%H:%M:%S", time.localtime()))
        model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not model_path:
            tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                            FLAGS.train_dir)
        else:
            with tf.Session(config=sess_config) as sess:
                # Load model from checkpoint.
                tf.logging.info("Loading model from checkpoint: %s", model_path)
                sess.run(init)
                saver.restore(sess, model_path)
                global_step = tf.train.global_step(sess, global_step_tensor.name)
                tf.logging.info("Successfully loaded %s at global step = %d.",
                                os.path.basename(model_path), global_step)

                # Start the queue runners.
                tf.train.start_queue_runners()

                # Run evaluation on the latest checkpoint.
                print("Extracting......")
                num_eval_batches = int(
                        math.ceil(float(test_size) / float(batch_size_per_gpu)))
                assert (num_eval_batches*batch_size_per_gpu) == test_size
                correct = 0
                count = 0
                for i in xrange(num_eval_batches):
                    test_start_time = time.time()
                    ret, pre, name = sess.run([top_k_op, probs, test_names])
                    correct += np.sum(ret)
                    for b in xrange(pre.shape[0]):
                        fp = open('%s/%s'%(FLAGS.feature_dir, os.path.basename(name[b])), 'a')
                        for f in xrange(pre.shape[1]):
                            fp.write('%f '%pre[b, f])
                        fp.write('\n')
                        fp.close()
                    test_duration = time.time() - test_start_time
                    count += len(ret)
                    cur_accuracy = float(correct)*100/count

                    test_examples_per_sec = float(batch_size_per_gpu) / test_duration

                msg = '{:>6.2f}%, {:>6}/{:<6}'.format(cur_accuracy, count, test_size)
                format_str = ('%s: total batch %d, accuracy=%s, (%.1f examples/sec; %.3f '
                        'sec/batch)')
                print (format_str % (datetime.now(), num_eval_batches, msg,
                                test_examples_per_sec, test_duration))


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.mode == 'extract':
        async_extract()
    else:
        tf.logging.fatal("Error mode.")

if __name__ == '__main__':
    tf.app.run()
