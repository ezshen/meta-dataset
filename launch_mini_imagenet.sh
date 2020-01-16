#!/bin/bash

export DATASRC=/mnt/disks/0/mini_imagenet
export SPLITS=~/meta-dataset/meta_dataset/dataset_conversion/splits
export RECORDS=/mnt/disks/0/records
export SOURCE=mini_imagenet
export EXPROOT=/mnt/disks/0/exps/mini_imagenet

for MODEL in graphfinetune
do
  export EXPNAME=${MODEL}_${SOURCE}_cross_entropy_oneshot_11
  python -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${EXPNAME} \
    --summary_dir=${EXPROOT}/summaries/${EXPNAME} \
    --gin_config=meta_dataset/learn/gin/graphfinetune/${EXPNAME}.gin \
    --gin_bindings="LearnerConfig.experiment_name='$EXPNAME'" \

done