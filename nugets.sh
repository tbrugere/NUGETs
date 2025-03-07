#!/bin/bash
gcloud config set project nugets-452122
export GCLOUD_PROJECT=nugets-452122
export PYTHONPATH="$HOME/research/pytorch_heterogeneous_batching:$PYTHONPATH"
uv run ipython --pdb -m nugets -- "$@"
