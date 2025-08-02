
#!/bin/bash
export GCLOUD_PROJECT=nugets-452122
export PYTHONPATH="$HOME/research/pytorch_heterogeneous_batching:$PYTHONPATH"
exec uv run --no-project python -m nugets -- "$@"
