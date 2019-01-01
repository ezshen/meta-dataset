#!/bin/bash

export DATASRC=/mnt/disks/0/imagenet
export SPLITS=~/meta-dataset/meta_dataset/dataset_conversion
export RECORDS=/mnt/disks/0/records
export SOURCE=imagenet
export EXPROOT=/mnt/disks/0/exps/imagenet

for MODEL in baselinefinetune
do
  export EXPNAME=${MODEL}_${SOURCE}
  python -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${EXPNAME} \
    --summary_dir=${EXPROOT}/summaries/${EXPNAME} \
    --gin_config=meta_dataset/learn/gin/best/${EXPNAME}.gin \
    --gin_bindings="LearnerConfig.experiment_name='$EXPNAME'"
done