#!/bin/bash

if [ -n "$1" ]; then
  ID=$1
else
  LAST_EPOCH=$(find saved_weights/25chan_corr/ -name contran-*.model |
      sed -n "s/saved_weights\/25chan_corr\/contran-\([0-9][0-9]*\)\.model/\1/p" |
      sort -n |
      tail -n 1)
  echo "Starting training at epoch $LAST_EPOCH"
  ID=$LAST_EPOCH
fi

# export CUDA_VISIBLE_DEVICES=0
python3 main_run.py $ID 25chan_corr