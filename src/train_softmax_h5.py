
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from datetime import datetime
import time
import random
import tensorflow as tf
import numpy as np
import h5py
import prepare
import tensorflow.contrib.slim as slim
import inception_resnet_v1 as network

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ["CUDA_VISIBLE_DEVICES"]='5,6,7'
logs_base_dir = r'log'
models_base_dir = r'model'
data_path = os.path.abspath('LTE_dataset_3c_1216.h5')

embedding_size = 128
optimizer = 'ADAM' #['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']
seed = 666
batch_size = 4
lr = 0.001
keep_probability = 1.0
weight_decay = 0.0
center_loss_factor = 0.0
learning_rate_decay_epochs = 100
learning_rate_decay_factor = 0.95
moving_average_decay = 0.9999
pretrained_model = False
max_nrof_epochs = 50

np.random.seed(seed=seed)
random.seed(seed)

with h5py.File(data_path, 'r') as h5f:
    feature_shape = h5f['features'].shape
    nrof_samples = feature_shape[0]
    nrof_classes = int(h5f['labels'][:].max() + 1)

train_indices, valid_indices = prepare.get_train_valid_indices(nrof_samples)
nrof_train_samples = train_indices.shape[0]
epoch_size = int(np.floor(nrof_train_samples / batch_size))

def main():



    with tf.Graph().as_default():

        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

        log_dir = os.path.join(os.path.expanduser(logs_base_dir), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        model_dir = os.path.join(os.path.expanduser(models_base_dir), subdir)
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)

        tf.set_random_seed(seed)
        global_step = tf.Variable(0, trainable=False)


        # Get a list of image paths and their labels

        assert nrof_train_samples > 0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')



        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')


        images_placeholder = tf.placeholder(tf.float32, shape=(None,)+feature_shape[1:], name='images')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None), name='labels')

        print('feature shape :', (None,) + feature_shape[1:])

        print('Total number of classes: %d' % nrof_classes)

        print('Total number of examples: %d' % nrof_train_samples)

        print('Building training graph')

        # Build the inference graph
        prelogits, end_points = network.inference(images_placeholder, keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size,
                                         weight_decay=weight_decay)
        print(end_points)

        logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(weight_decay),
                                      scope='Logits', reuse=False)
        logits = tf.identity(logits, 'logits')

        softmax = tf.nn.softmax(logits,name='Softmax')
        prediction = tf.argmax(softmax, 1)
        accuracy  = tf.reduce_mean(tf.to_float(tf.equal(prediction,labels_placeholder)), name='accuracy')

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # # Add center loss
        # if  center_loss_factor > 0.0:
        #     prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
        #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   learning_rate_decay_epochs * epoch_size,
                                                   learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_placeholder, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = train_operation(total_loss, global_step, optimizer,
                                 learning_rate, moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            h5f = h5py.File(data_path,'r')
            h5_features = h5f['features']
            h5_labels = h5f['labels']
            epoch = 0
            while epoch < max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch += 1
                # Train for one epoch
                train(sess, h5_features, h5_labels, train_indices, images_placeholder, labels_placeholder, learning_rate_placeholder, phase_train_placeholder,
                      epoch, global_step, total_loss, accuracy ,train_op, summary_op, summary_writer, regularization_losses)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)


def train( sess, h5_features, h5_labels, train_indices, images_placeholder, labels_placeholder,learning_rate_placeholder, phase_train_placeholder,
           epoch, global_step, loss, accuracy, train_op, summary_op, summary_writer, regularization_losses):
    batch_number = 0
    shuffle_idx = np.random.permutation(nrof_train_samples)
    # Training loop
    train_time = 0
    while batch_number < epoch_size:
        start_time = time.time()
        batch_idx = sorted(train_indices[shuffle_idx[batch_number * batch_size : batch_number * batch_size + batch_size]])
        feed_dict = {learning_rate_placeholder: lr,
                     phase_train_placeholder: True,
                     images_placeholder: h5_features[batch_idx],
                     labels_placeholder: h5_labels[batch_idx] }
        if (batch_number % 100 == 0):
            err,acc, _, step, reg_loss, summary_str = sess.run(
                [loss, accuracy, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err,acc, _, step, reg_loss = sess.run([loss, accuracy, train_op, global_step, regularization_losses],
                                              feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAcc %2.3f\t RegLoss %2.3f' %
              (epoch, batch_number + 1, epoch_size, duration, err,acc,np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def train_operation(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


if __name__ == '__main__':
    main()

