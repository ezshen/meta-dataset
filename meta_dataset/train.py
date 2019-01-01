# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
r"""Script for training models on the benchmark.

Launching command for batch baseline:
# pylint: disable=line-too-long
python -m meta_dataset.train \
  --train_checkpoint_dir=/tmp/bench --summary_dir=/tmp/bench \
  --records_root_dir=<records_root> \
  --alsologtostderr \
  --gin_config=meta_dataset/learn/gin/default/<exp_name>.gin
  --gin_bindings="LearnerConfig.experiment_name='<exp_name>'"
# pylint: enable=line-too-long

where:
  <exp_name> is e.g. 'debug_proto_mini_imagenet'

To override elements from the config, you can use arguments of the form:
  For gin: --gin_bindings='foo = 1000000'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import logging
import gin.tf
from meta_dataset import data
from meta_dataset import trainer
from meta_dataset.data import config  # pylint: disable=unused-import
import tensorflow as tf

DEFAULT_SAVING_DIRECTORY = '/tmp/metadataset'

tf.flags.DEFINE_string('train_checkpoint_dir',
                       os.path.join(DEFAULT_SAVING_DIRECTORY, 'checkpoints'),
                       'The directory to save checkpoints.')
tf.flags.DEFINE_string('summary_dir',
                       os.path.join(DEFAULT_SAVING_DIRECTORY, 'summaries'),
                       'The directory for writing summaries.')
tf.flags.DEFINE_bool(
    'reload_eval_checkpoint_gin_config', False,
    'Whether to reload an operative Gin configuration along with a checkpoint '
    'for evaluation.')

tf.flags.DEFINE_bool(
    'is_training', True, 'Whether we are in the training phase. '
    'Used to control whether to perform training or evaluation.')

tf.flags.DEFINE_multi_string('gin_config', None,
                             'List of paths to the config files.')
tf.flags.DEFINE_multi_string('gin_bindings', None,
                             'List of Gin parameter bindings.')

tf.flags.DEFINE_string(
    'eval_imbalance_dataset', '', 'A dataset on which to perform evaluation '
    'for assessing how class imbalance affects performance in binary episodes. '
    'By default it is empty and no imbalance analysis is performed.')

tf.flags.DEFINE_bool(
    'eval_finegrainedness', False, 'Whether to perform only 2-way ImageNet '
    'evaluation for assessing performance as a function of how finegrained '
    'each task is. This differs from usual ImageNet eval in the sampling '
    'procedure used to get episodes, and therefore requires its own setting.')

tf.flags.DEFINE_enum(
    'eval_finegrainedness_split', 'train', ['train', 'valid', 'test'], 'The '
    'split whose results we want to use for the fine-grainedness analysis.'
    'Contrary to most analyses which are performed on the test split only, the '
    'fine-grainedness analysis may also be performed on the train or valid '
    'sub-graphs of ImageNet too, since the test sub-graph evidently does not '
    'exhibit enough variation in the fine-grainedness of its different tasks '
    'to allow for a meaningful analysis.')

tf.flags.DEFINE_multi_enum(
    'omit_from_saving_and_reloading',
    ['num_left_in_epoch', 'finetune', 'linear_classifier', 'adam_opt'], [
        'num_left_in_epoch', 'finetune', 'linear_classifier', 'adam_opt', 'fc',
        'weight_copy'
    ],
    'A comma-separated list of substrings such that all variables containing '
    'them should not be saved and reloaded.')

FLAGS = tf.flags.FLAGS


def parse_cmdline_gin_configurations():
  """Parse Gin configurations from all command-line sources."""
  with gin.unlock_config():
    gin.parse_config_files_and_bindings(
        FLAGS.gin_config, FLAGS.gin_bindings, finalize_config=True)


def operative_config_path(checkpoint_dir,
                          operative_config_filename='operative_config.gin'):
  return os.path.join(checkpoint_dir, operative_config_filename)


def load_operative_gin_configurations(checkpoint_dir):
  """Load operative Gin configurations from the given checkpoint directory."""
  gin_log_file = operative_config_path(checkpoint_dir)
  with gin.unlock_config():
    gin.parse_config_file(gin_log_file)
  gin.finalize_config()
  logging.info('Operative Gin configurations loaded from `checkpoint_dir`: %s',
               gin_log_file)


def record_operative_gin_configurations(checkpoint_dir):
  """Record operative Gin configurations in the given checkpoint directory."""
  gin_log_file = operative_config_path(checkpoint_dir)
  # If it exists already, rename it instead of overwriting it.
  # This just saves the previous one, not all the ones before.
  if tf.io.gfile.exists(gin_log_file):
    tf.io.gfile.rename(gin_log_file, gin_log_file + '.old', overwrite=True)
  with tf.io.gfile.GFile(gin_log_file, 'w') as f:
    f.write(gin.operative_config_str())


def main(unused_argv):

  # Parse Gin configurations passed to this script.
  parse_cmdline_gin_configurations()

  # Try to reload a previously recorded Gin configuration.
  # TODO(eringrant): Allow querying of a value to be bound without actually
  # binding it to avoid the redundant call to `parse_cmdline_gin_configurations`
  # below.
  checkpoint_for_eval = gin.query_parameter('LearnerConfig.checkpoint_for_eval')
  if checkpoint_for_eval and FLAGS.reload_eval_checkpoint_gin_config:
    eval_checkpoint_dir = os.path.dirname(checkpoint_for_eval)
    load_operative_gin_configurations(eval_checkpoint_dir)

    # Reload the command-line Gin configuration to allow overriding of the Gin
    # configuration loaded from the checkpoint directory.
    parse_cmdline_gin_configurations()

  # Wrap object instantiations to print out full Gin configuration on failure.
  try:
    learner_config = trainer.LearnerConfig()

    (train_datasets, eval_datasets, restrict_classes,
     restrict_num_per_class) = trainer.get_datasets_and_restrictions()

    train_learner = None
    if FLAGS.is_training or (FLAGS.eval_finegrainedness and
                             FLAGS.eval_finegrainedness_split == 'train'):
      # If eval_finegrainedness is True, even in pure evaluation mode we still
      # require a train learner, since we may perform this analysis on the
      # training sub-graph of ImageNet too.
      train_learner = trainer.NAME_TO_LEARNER[learner_config.train_learner]
    eval_learner = trainer.NAME_TO_LEARNER[learner_config.eval_learner]

    # Get a trainer or evaluator.
    trainer_kwargs = {
        'train_learner': train_learner,
        'eval_learner': eval_learner,
        'is_training': FLAGS.is_training,
        'train_dataset_list': train_datasets,
        'eval_dataset_list': eval_datasets,
        'restrict_classes': restrict_classes,
        'restrict_num_per_class': restrict_num_per_class,
        'checkpoint_dir': FLAGS.train_checkpoint_dir,
        'summary_dir': FLAGS.summary_dir,
        'records_root_dir': FLAGS.records_root_dir,
        'eval_finegrainedness': FLAGS.eval_finegrainedness,
        'eval_finegrainedness_split': FLAGS.eval_finegrainedness_split,
        'eval_imbalance_dataset': FLAGS.eval_imbalance_dataset,
        'omit_from_saving_and_reloading': FLAGS.omit_from_saving_and_reloading,
    }
    if learner_config.episodic:
      trainer_instance = trainer.EpisodicTrainer(**trainer_kwargs)
      if learner_config.train_learner not in trainer.EPISODIC_LEARNER_NAMES:
        raise ValueError(
            'When "episodic" is True, "train_learner" should be an episodic one, '
            'among {}.'.format(trainer.EPISODIC_LEARNER_NAMES))
    else:
      trainer_instance = trainer.BatchTrainer(**trainer_kwargs)
      if learner_config.train_learner not in trainer.BATCH_LEARNER_NAMES:
        raise ValueError(
            'When "episodic" is False, "train_learner" should be a batch one, '
            'among {}.'.format(trainer.BATCH_LEARNER_NAMES))

  except ValueError as e:
    logging.info('Full Gin configurations:\n%s', gin.config_str())
    raise e

  # All configurable objects/functions should have been instantiated/called.
  logging.info('Operative Gin configurations:\n%s', gin.operative_config_str())
  if FLAGS.is_training and FLAGS.train_checkpoint_dir:
    record_operative_gin_configurations(FLAGS.train_checkpoint_dir)

  datasets = train_datasets if FLAGS.is_training else eval_datasets
  logging.info('Starting %s for dataset(s) %s...',
               'training' if FLAGS.is_training else 'evaluation', datasets)
  if FLAGS.is_training:
    trainer_instance.train()
  elif set(datasets).intersection(trainer.DATASETS_WITH_EXAMPLE_SPLITS):
    if not data.POOL_SUPPORTED:
      raise NotImplementedError('Example-level splits or pools not supported.')
  else:
    if len(datasets) != 1:
      raise ValueError('Requested datasets {} for evaluation, but evaluation '
                       'should be performed on individual datasets '
                       'only.'.format(datasets))

    eval_split = 'test'
    if FLAGS.eval_finegrainedness:
      eval_split = FLAGS.eval_finegrainedness_split

    trainer_instance.evaluate(eval_split)

  # Flushes the event file to disk and closes the file.
  if trainer_instance.summary_writer:
    trainer_instance.summary_writer.close()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
