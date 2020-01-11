from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from absl import logging
import gin.tf
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf
import os
import json
import numpy as np

from meta_dataset import learner

MAX_WAY = 50  # The maximum number of classes we will see in any batch.

def proto_fc_weights(self, prototypes, zero_pad_to_max_way=False):
  """Computes the Prototypical fc layer's weights.

  Args:
    prototypes: Tensor of shape [num_classes, embedding_size]
    zero_pad_to_max_way: Whether to zero pad to max num way.

  Returns:
    fc_weights: Tensor of shape [embedding_size, num_classes] or
      [embedding_size, MAX_WAY] when zero_pad_to_max_way is True.
  """
  fc_weights = 2 * prototypes
  fc_weights = tf.transpose(fc_weights)
  if zero_pad_to_max_way:
    paddings = [[0, 0], [0, MAX_WAY - tf.shape(fc_weights)[1]]]
    fc_weights = tf.pad(fc_weights, paddings, 'CONSTANT', constant_values=0)
  return fc_weights

def proto_fc_bias(self, prototypes, zero_pad_to_max_way=False):
  """Computes the Prototypical fc layer's bias.

  Args:
    prototypes: Tensor of shape [num_classes, embedding_size]
    zero_pad_to_max_way: Whether to zero pad to max num way.

  Returns:
    fc_bias: Tensor of shape [num_classes] or [MAX_WAY]
      when zero_pad_to_max_way is True.
  """
  fc_bias = -tf.square(tf.norm(prototypes, axis=1))
  if zero_pad_to_max_way:
    paddings = [[0, MAX_WAY - tf.shape(fc_bias)[0]]]
    fc_bias = tf.pad(fc_bias, paddings, 'CONSTANT', constant_values=0)
  return fc_bias


@gin.configurable
class GraphFinetuneLearner(learner.BaselineLearner):
  """A Baseline Network with test-time finetuning."""

  def __init__(self,
               is_training,
               transductive_batch_norm,
               backprop_through_moments,
               ema_object,
               embedding_fn,
               reader,
               num_train_classes,
               num_test_classes,
               weight_decay,
               num_finetune_steps,
               finetune_lr,
               num_nodes,
               max_ancestors,
               max_graph_dist,
               ancestors_filepath,
               graph_dists_filepath,
               proto_init,
               loss_type,
               graph_loss_scale,
               debug_log=False,
               finetune_all_layers=False,
               finetune_with_adam=False):
    """Initializes a baseline learner.

    Args:
      is_training: If we are training or not.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
      backprop_through_moments: Whether to allow gradients to flow through the
        given support set moments. Only applies to non-transductive batch norm.
      ema_object: An Exponential Moving Average (EMA).
      embedding_fn: A callable for embedding images.
      reader: A SplitReader that reads episodes or batches.
      num_train_classes: The total number of classes of the dataset.
      num_test_classes: The number of classes in each episode.
      weight_decay: coefficient for L2 regularization.
      num_finetune_steps: number of finetune steps.
      finetune_lr: the learning rate used for finetuning.
      debug_log: If True, print out debug logs.
      finetune_all_layers: Whether to finetune all embedding variables. If
        False, only trains a linear classifier on top of the embedding.
      finetune_with_adam: Whether to use Adam for the within-episode finetuning.
        If False, gradient descent is used instead.
    """

    self.num_finetune_steps = num_finetune_steps
    self.finetune_lr = finetune_lr
    self.debug_log = debug_log
    self.finetune_all_layers = finetune_all_layers
    self.finetune_with_adam = finetune_with_adam
    self.proto_init = proto_init
    self.loss_type = loss_type

    if finetune_with_adam:
      self.finetune_opt = tf.train.AdamOptimizer(self.finetune_lr)

    self.max_ancestors = max_ancestors
    self.num_nodes = num_nodes

    if self.loss_type == 'ancestors':
      # load ancestors
      with tf.io.gfile.GFile(ancestors_filepath) as f:
        ancestor_dict = json.load(f)

      # create ancestor lookup table size of (total_num_classes, max_ancestors)
      ancestor_lookup = []
      for _, ancestors in sorted(ancestor_dict.items()):
        if len(ancestors) < self.max_ancestors:
          ancestors += ancestors
          ancestor_lookup.append(ancestors[:self.max_ancestors])
        else:
          ancestor_lookup.append(ancestors[:self.max_ancestors])
      self.ancestor_lookup = tf.constant(ancestor_lookup, dtype=tf.int64)

    if self.loss_type == 'graph_dists':
      # load graph distances
      with tf.io.gfile.GFile(graph_dists_filepath) as f:
        graph_dists = np.genfromtxt(f, delimiter='\t')
      self.graph_dists = tf.constant(graph_dists, dtype=tf.int64)

    self.graph_loss_scale = graph_loss_scale
    self.max_graph_dist = max_graph_dist
    # Note: the weight_decay value provided here overrides the value gin might
    # have for BaselineLearner's own weight_decay.
    super(GraphFinetuneLearner,
          self).__init__(is_training, transductive_batch_norm,
                         backprop_through_moments, ema_object, embedding_fn,
                         reader, num_train_classes, num_test_classes,
                         weight_decay)

    if not self.is_training:
      self.way = learner.compute_way(self.data)

  def forward_pass(self):
    if self.is_training:
      images = self.data.images
      embeddings_params_moments = self.embedding_fn(images, self.is_training)
      self.train_embeddings = embeddings_params_moments['embeddings']
      self.train_embeddings_var_dict = embeddings_params_moments['params']
    else:
      train_embeddings_params_moments = self.embedding_fn(
          self.data.train_images, self.is_training)
      self.train_embeddings = train_embeddings_params_moments['embeddings']
      self.train_embeddings_var_dict = train_embeddings_params_moments['params']
      support_set_moments = None
      if not self.transductive_batch_norm:
        support_set_moments = train_embeddings_params_moments['moments']
      test_embeddings = self.embedding_fn(
          self.data.test_images,
          self.is_training,
          moments=support_set_moments,
          backprop_through_moments=self.backprop_through_moments)
      self.test_embeddings = test_embeddings['embeddings']

  def compute_logits(self):
    """Computes the logits."""
    logits, node_logits = None, None

    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      # Always maps to a space whose dimensionality is the number of classes
      # at meta-training time.
      embedding_dims = self.train_embeddings.get_shape().as_list()[-1]
      node_embeddings = learner.weight_variable([embedding_dims, self.num_nodes])
      node_bias = learner.bias_variable([self.num_nodes])
      self.node_embeddings = node_embeddings
      self.node_bias = node_bias

    if self.is_training:
      logits, node_logits = self.forward_pass_fc(self.train_embeddings)
    else:
      # ------------------------ Finetuning -------------------------------
      # Possibly make copies of embedding variables, if they will get modified.
      # This is for making temporary-only updates to the embedding network
      # which will not persist after the end of the episode.
      make_copies = self.finetune_all_layers
      (self.embedding_vars_keys, self.embedding_vars,
       embedding_vars_copy_ops) = learner.get_embeddings_vars_copy_ops(
           self.train_embeddings_var_dict, make_copies)
      embedding_vars_copy_op = tf.group(*embedding_vars_copy_ops)

      # A Variable for the weights of the fc layer, a Variable for the bias of the
      # fc layer, and a list of operations (possibly empty) that copies them.
      (self.node_embeddings_finetune, self.node_bias_finetune, fc_vars_copy_ops) = learner.get_fc_vars_copy_ops(
          node_embeddings, node_bias, make_copies=True)
      fc_vars_copy_op = tf.group(*fc_vars_copy_ops)

      # Compute the initial training loss (only for printing purposes). This
      # line is also needed for adding the fc variables to the graph so that the
      # tf.all_variables() line below detects them.
      logits, node_logits = self._fc_layer(self.train_embeddings, self.data.train_labels, self.data.train_class_ids)
      finetune_loss = self._classification_loss(logits, node_logits, self.data.train_labels,
                                                self.way)

      # Decide which variables to finetune.
      fc_vars, vars_to_finetune = [], []
      vars_to_finetune.append(self.node_embeddings_finetune)
      vars_to_finetune.append(self.node_bias_finetune)
      if self.finetune_all_layers:
        vars_to_finetune.extend(self.embedding_vars)
      self.vars_to_finetune = vars_to_finetune
      logging.info('Finetuning will optimize variables: %s', vars_to_finetune)

      for i in range(self.num_finetune_steps):
        if i == 0:
          # Randomly initialize the fc layer.
          # fc_reset = tf.variables_initializer(var_list=fc_vars)

          # Adam related variables are created when minimize() is called.
          # We create an unused op here to put all adam varariables under
          # the 'adam_opt' namescope and create a reset op to reinitialize
          # these variables before the first finetune step.
          adam_reset = tf.no_op()
          if self.finetune_with_adam:
            with tf.variable_scope('adam_opt'):
              unused_op = self.finetune_opt.minimize(
                  finetune_loss, var_list=vars_to_finetune)
            adam_reset = tf.variables_initializer(self.finetune_opt.variables())
          with tf.control_dependencies(
              [adam_reset, finetune_loss, embedding_vars_copy_op, fc_vars_copy_op] +
              vars_to_finetune):
            print_op = tf.no_op()
            if self.debug_log:
              print_op = tf.print([
                  'step: %d' % i, vars_to_finetune[0][0, 0], 'loss:',
                  finetune_loss
              ])

            with tf.control_dependencies([print_op]):
              # Get the operation for finetuning.
              # (The logits and loss are returned just for printing).
              logits, finetune_loss, finetune_op = self._get_finetune_op()

              if self.debug_log:
                # Test logits are computed only for printing logs.
                test_embeddings = self.embedding_fn(
                    self.data.test_images,
                    self.is_training,
                    params=collections.OrderedDict(
                        zip(self.embedding_vars_keys, self.embedding_vars)),
                    reuse=True)['embeddings']
                test_logits, _ = self._fc_layer(test_embeddings, self.data.test_labels, self.data.test_class_ids)

        else:
          with tf.control_dependencies([finetune_op, finetune_loss] +
                                       vars_to_finetune):
            print_op = tf.no_op()
            if self.debug_log:
              print_op = tf.print([
                  'step: %d' % i, vars_to_finetune[0][0, 0],
                  'loss:',
                  finetune_loss, 'accuracy:',
                  self._compute_accuracy(logits, self.data.train_labels),
                  'test accuracy:',
                  self._compute_accuracy(test_logits, self.data.test_labels),
              ])

            with tf.control_dependencies([print_op]):
              # Get the operation for finetuning.
              # (The logits and loss are returned just for printing).
              logits, finetune_loss, finetune_op = self._get_finetune_op()

              if self.debug_log:
                # Test logits are computed only for printing logs.
                test_embeddings = self.embedding_fn(
                    self.data.test_images,
                    self.is_training,
                    params=collections.OrderedDict(
                        zip(self.embedding_vars_keys, self.embedding_vars)),
                    reuse=True)['embeddings']
                test_logits, _ = self._fc_layer(test_embeddings, self.data.test_labels, self.data.test_class_ids)

      # Finetuning is now over, compute the test performance using the updated
      # fc layer, and possibly the updated embedding network.
      with tf.control_dependencies([finetune_op] + vars_to_finetune):
        test_embeddings = self.embedding_fn(
            self.data.test_images,
            self.is_training,
            params=collections.OrderedDict(
                zip(self.embedding_vars_keys, self.embedding_vars)),
            reuse=True)['embeddings']
        test_logits, _ = self._fc_layer(test_embeddings, self.data.test_labels, self.data.test_class_ids)

        if self.debug_log:
          # The train logits are computed only for printing.
          train_embeddings = self.embedding_fn(
              self.data.train_images,
              self.is_training,
              params=collections.OrderedDict(
                  zip(self.embedding_vars_keys, self.embedding_vars)),
              reuse=True)['embeddings']
          logits, _ = self._fc_layer(train_embeddings, self.data.train_labels, self.data.train_class_ids)

        print_op = tf.no_op()
        if self.debug_log:
          print_op = tf.print([
              'accuracy: ',
              self._compute_accuracy(logits, self.data.train_labels),
              'test accuracy: ',
              self._compute_accuracy(test_logits, self.data.test_labels)
          ])
        with tf.control_dependencies([print_op]):
          self.test_logits, self.test_node_logits = self._fc_layer(test_embeddings, self.data.test_labels, self.data.test_class_ids)
          logits = self.test_logits
          node_logits = self.test_node_logits
    return logits, node_logits

  def graph_classifier_logits(self, embeddings, label_to_id, num_nodes, cosine_classifier,
                               cosine_logits_multiplier, use_weight_norm, proto_init, node_embeddings, node_bias):

    logits, node_logits = None, None
    if use_weight_norm:
      raise NotImplementedError
    else:
      # if init_fc:
      #   if proto_init:
      #     raise NotImplementedError
      #     # prototypes = compute_prototypes(embeddings, class_ids)
      #     # self.node_embeddings = proto_fc_weights(
      #     #     prototypes, zero_pad_to_max_way=True)
      #     # self.node_bias = proto_fc_bias(
      #     #     prototypes, zero_pad_to_max_way=True)
      #   else:
      #     self.node_embeddings = learner.weight_variable([embedding_dims, num_nodes])
      #     self.node_bias = learner.bias_variable([num_nodes])
      #   node_embeddings = self.node_embeddings
      #   node_bias = self.node_bias
      # else:
      #   node_embeddings = self.node_embeddings_finetune
      #   node_bias = self.node_bias_finetune
      # if is_training:
        # node_embeddings = self.node_embeddings
        # node_bias = self.node_bias
      # else:
      #   node_embeddings = self.node_embeddings_finetune
      #   node_bias = self.node_bias_finetune
      class_embeddings = tf.gather(node_embeddings, label_to_id, axis=-1)
      class_bias = tf.gather(node_bias, label_to_id, axis=-1)
      logits = learner.linear_classifier_forward_pass(embeddings, class_embeddings, class_bias,
                                              cosine_classifier,
                                              cosine_logits_multiplier, False)

      node_logits = learner.linear_classifier_forward_pass(embeddings, node_embeddings, node_bias,
                                            cosine_classifier,
                                            cosine_logits_multiplier, False)
    return logits, node_logits

  def forward_pass_fc(self, embeddings):
    """Passes the provided embeddings through the fc layer to get the logits."""
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      logits, node_logits = self.graph_classifier_logits(embeddings,
                                        tf.range(self.num_train_classes),
                                        self.num_nodes,
                                        self.cosine_classifier,
                                        self.cosine_logits_multiplier,
                                        self.use_weight_norm,
                                        self.proto_init,
                                        self.node_embeddings,
                                        self.node_bias)
      return logits, node_logits


  def _fc_layer(self, embedding, labels, classes):
    """The fully connected layer to be finetuned."""
    idx, _ = tf.unique(labels)
    unique_classes, _ = tf.unique(classes)
    label_to_id = tf.gather(unique_classes, idx)
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
      logits, node_logits = self.graph_classifier_logits(embedding,
                                        label_to_id,
                                        self.num_nodes,
                                        self.cosine_classifier,
                                        self.cosine_logits_multiplier,
                                        self.use_weight_norm,
                                        self.proto_init,
                                        self.node_embeddings_finetune,
                                        self.node_bias_finetune)
    return logits, node_logits

  def _get_finetune_op(self):
    """Returns the operation for performing a finetuning step."""
    if self.finetune_all_layers:
      # Must re-do the forward pass because the embedding has changed.
      train_embeddings = self.embedding_fn(
          self.data.train_images,
          self.is_training,
          params=collections.OrderedDict(
              zip(self.embedding_vars_keys, self.embedding_vars)),
          reuse=True)['embeddings']
    else:
      train_embeddings = self.train_embeddings
    logits, node_logits = self._fc_layer(self.train_embeddings, self.data.train_labels, self.data.train_class_ids)
    finetune_loss = self._classification_loss(logits, node_logits, self.data.train_labels,
                                              self.way)
    # Perform one step of finetuning.
    if self.finetune_with_adam:
      finetune_op = self.finetune_opt.minimize(
          finetune_loss, var_list=self.vars_to_finetune)
    else:
      # Apply vanilla gradient descent instead of Adam.
      update_ops = learner.gradient_descent_step(finetune_loss, self.vars_to_finetune,
                                         True, False,
                                         self.finetune_lr)['update_ops']
      finetune_op = tf.group(*update_ops)
    return logits, finetune_loss, finetune_op

  def create_loss(self, logits, node_logits, labels, class_ids, num_classes):
    if self.loss_type == "cross_entropy":
      onehot_labels = tf.one_hot(labels, num_classes)
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)
    elif self.loss_type == "graph_dists":
      graph_dists = tf.gather(self.graph_dists, class_ids)
      num_all_classes = graph_dists.shape[-1]
      all_class_logits = tf.gather(node_logits, tf.range(num_all_classes), axis=-1)
      mask = tf.cast(tf.greater(graph_dists, self.max_graph_dist), tf.float64) * 1e4
      graph_labels = tf.minimum((self.max_graph_dist + 1) / graph_dists, 1) - mask
      graph_labels = tf.nn.softmax(graph_labels)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=graph_labels, logits=all_class_logits))
    elif self.loss_type == "ancestors":
      onehot_labels = tf.one_hot(labels, num_classes)
      class_loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)

      ancestor_ids = tf.gather(self.ancestor_lookup, class_ids)
      ancestor_ids = tf.concat([tf.expand_dims(labels, -1), ancestor_ids], -1)
      ancestor_labels = tf.reduce_sum(tf.one_hot(ancestor_ids, self.num_nodes, on_value=1 / (self.max_ancestors + 1)), 1)
      graph_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ancestor_labels, logits=node_logits)
      graph_loss = tf.reduce_mean(graph_loss)
      loss = class_loss + self.graph_loss_scale * graph_loss
    else:
      raise NotImplementedError

    regularization = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss += self.weight_decay * regularization
    return loss

  def compute_loss(self):
    """Computes the loss."""
    if self.is_training:
      labels = tf.cast(self.data.labels, tf.int64)
      self.train_logits, self.train_node_logits = self.compute_logits()
      with tf.name_scope('loss'):
        loss = self.create_loss(self.train_logits, self.train_node_logits, labels, labels, self.num_train_classes)
        return loss
    else:
      self.test_logits, _ = self.compute_logits()
      return tf.constant(0.0)

  def _classification_loss(self, logits, node_logits, labels, num_classes):
    """Computes softmax cross entropy loss."""
    labels = tf.cast(labels, tf.int64)
    class_ids = tf.cast(self.data.train_class_ids, tf.int64)
    with tf.name_scope('finetuning_loss'):
      loss = self.create_loss(logits, node_logits, labels, class_ids, num_classes)
    return loss

  def _compute_accuracy(self, logits, labels):
    """Computes the accuracy on the given episode."""
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # print_op = tf.print(['logits', logits, 'predictions', predictions, 'logits.shape', logits.shape, 'predictions.shape', predictions.shape, 'labels', labels])
    # with tf.control_dependencies([print_op]):
    correct = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy