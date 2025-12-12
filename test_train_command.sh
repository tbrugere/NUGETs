#!/bin/bash
export PYTHONPATH="$WORK/research/pytorch_heterogeneous_batching:$PYTHONPATH"
python -m nugets train \
    --batch-size 12 --learning-rate 1.e-6 \
    --task "WassersteinDistanceTask" \
    --dataset workdir/datasets/configs/dummy_growing_circles.yaml \
    --backbone-type CoupledNetwork  \
    --backbone-latent-dimension 256 \
    --backbone-p 2 \
    --backbone-decoder-distance SinkhornLoss \
    --backbone-encoder-backbone Transformer \
        --backbone-encoder-n-heads 12 \
        --backbone-encoder-n-layers 6 \
        --backbone-encoder-d-model 768 \
        --backbone-encoder-feed-forward-hidden-dim 512\
        --backbone-aggregation mean
