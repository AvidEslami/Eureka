#!/bin/bash
# Validate the .sif on an H100 before submitting the real job.
#
# Usage (inside an interactive Slurm allocation):
#   salloc --account=def-rhinehar --gres=gpu:h100:1 --cpus-per-task=8 \
#          --mem=64G --time=1:00:00
#   bash docker/smoke_test.sh
#
# Pass criteria printed at the end. Any failure should be diagnosed before
# touching sbatch.

set -euo pipefail

module load apptainer

# Resolve $PROJECT and $SCRATCH (not auto-exported on Nibi).
: "${PROJECT:=$HOME/projects/def-rhinehar}"
: "${SCRATCH:=$HOME/scratch}"
export PROJECT SCRATCH

SIF="${SIF:-$PROJECT/containers/eureka.sif}"
CODE="${CODE:-$PROJECT/eureka/src}"

if [ ! -f "$SIF" ]; then
    echo "ERROR: SIF not found at $SIF" >&2
    echo "Build it on the login node first:" >&2
    echo "  module load apptainer && apptainer build $SIF docker-archive://eureka.tar.gz" >&2
    exit 1
fi

if [ ! -d "$CODE/eureka" ]; then
    echo "ERROR: Eureka source not found at $CODE/eureka" >&2
    echo "Run docker/setup_cluster.sh on the login node first." >&2
    exit 1
fi

echo "=== nvidia-smi (outside container) ==="
nvidia-smi

echo ""
echo "=== nvidia-smi (inside container, via --nv) ==="
apptainer exec --nv "$SIF" nvidia-smi

echo ""
echo "=== Vulkan headless check ==="
# vulkaninfo --summary is brief; full vulkaninfo is verbose
apptainer exec --nv "$SIF" vulkaninfo --summary 2>&1 | head -40 \
    || echo "WARNING: vulkaninfo failed — IsaacGym headless rendering may not work"

echo ""
echo "=== Python imports inside eureka env ==="
apptainer exec --nv \
    --bind "$CODE":/workspace/eureka \
    --bind "${SLURM_TMPDIR:-/tmp}":/tmp \
    --env PYTHONPATH=/workspace/eureka:/workspace/eureka/isaacgymenvs:/workspace/eureka/rl_games \
    --env LD_LIBRARY_PATH=/opt/conda/envs/eureka/lib \
    "$SIF" \
    bash -lc 'conda run -n eureka --no-capture-output python -c "
# IsaacGym MUST be imported before torch (it patches CUDA init).
import isaacgym
print(\"isaacgym: ok\")
import isaacgymenvs
print(\"isaacgymenvs: ok\")
import rl_games
print(\"rl_games: ok\")
import torch
print(\"torch:\", torch.__version__, \"  cuda:\", torch.version.cuda)
print(\"cuda available:\", torch.cuda.is_available())
print(\"device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")
"'

echo ""
echo "=== Python imports inside vlm env ==="
apptainer exec --nv "$SIF" bash -lc 'conda run -n vlm --no-capture-output python -c "
from google import genai
print(\"google-genai: ok\")
"'

echo ""
echo "=== Smoke test PASSED ==="
echo "All imports succeeded. Ready to sbatch docker/job.sh."
