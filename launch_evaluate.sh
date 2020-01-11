#!/bin/bash

export DATASRC=/mnt/disks/0/mini_imagenet
export SPLITS=~/meta-dataset/meta_dataset/dataset_conversion
export RECORDS=/mnt/disks/0/records
export SOURCE=mini_imagenet
export EXPROOT=/mnt/disks/0/exps/mini_imagenet
export MODEL=graphfinetune
export EXPNAME=${MODEL}_${SOURCE}_ancestors_max_two_oneshot
export DATASET=mini_imagenet

# set BESTNUM to the "best_update_num" field in the corresponding best_....txt
export BESTNUM=7500
python -m meta_dataset.train \
  --is_training=False \
  --records_root_dir=$RECORDS \
  --summary_dir=${EXPROOT}/summaries/${EXPNAME}_eval_$DATASET \
  --gin_config=meta_dataset/learn/gin/graphfinetune/${EXPNAME}.gin \
  --gin_bindings="LearnerConfig.experiment_name='${EXPNAME}'" \
  --gin_bindings="LearnerConfig.pretrained_checkpoint=''" \
  --gin_bindings="LearnerConfig.checkpoint_for_eval='${EXPROOT}/checkpoints/${EXPNAME}/model_${BESTNUM}.ckpt'" \
  --gin_bindings="benchmark.eval_datasets='$DATASET'"