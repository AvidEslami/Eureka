#!/bin/bash
#SBATCH --job-name=eureka
#SBATCH --account=def-rhinehar
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Eureka reward_learning_pipeline.py on Nibi.
#
# Prerequisites (one-time):
#   1. docker/setup_cluster.sh was run on the login node
#      ($PROJECT/eureka/src cloned, $SCRATCH/eureka/* dirs created, symlinks in place)
#   2. $HOME/.eureka_env exists (chmod 600) with GOOGLE_API_KEY + WANDB_API_KEY
#   3. auto_preference_data_open_{inward,outward}/ rsync'd to $PROJECT/eureka/data/
#      and symlinked into $PROJECT/eureka/src/eureka/
#   4. $PROJECT/containers/eureka.sif built (see docker/build.sh)

set -euo pipefail

module load apptainer

# --- Resolve $PROJECT and $SCRATCH (not auto-exported on Nibi) ---
: "${PROJECT:=$HOME/projects/def-rhinehar}"
: "${SCRATCH:=$HOME/scratch}"
export PROJECT SCRATCH

# --- Paths ---
SIF="$PROJECT/containers/eureka.sif"
CODE="$PROJECT/eureka/src"       # bind-mounted at /workspace/eureka
OUT="$SCRATCH/eureka"            # experiments/ + wandb/

# --- Secrets (GOOGLE_API_KEY, WANDB_API_KEY, optionally WANDB_ENTITY) ---
# shellcheck disable=SC1090
source "$HOME/.eureka_env"

# --- Sanity: log the GPU we got ---
echo "=== Node $(hostname) GPU ==="
apptainer exec --nv "$SIF" nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "============================"

# --- The actual job ---
# Bind mounts:
#   $CODE -> /workspace/eureka         (Eureka source from $PROJECT)
#   $OUT/experiments -> /workspace/eureka/eureka/experiments
#   $OUT/wandb       -> /workspace/eureka/eureka/wandb
#   $SLURM_TMPDIR    -> /tmp           (node-local NVMe; IsaacGym dumps tensors here)
#
# PYTHONPATH lets the eureka env import isaacgymenvs, rl_games, and eureka
# modules from the bind-mounted source without needing pip install -e at runtime.
#
# --env KEY=$KEY: Apptainer requires the key=value form (unlike Docker's
# bare --env KEY). The right side is expanded from the shell env populated
# by `source ~/.eureka_env`.

apptainer exec --nv \
    --bind "$CODE":/workspace/eureka \
    --bind "$OUT/experiments":/workspace/eureka/eureka/experiments \
    --bind "$OUT/wandb":/workspace/eureka/eureka/wandb \
    --bind "$PROJECT/eureka/overrides":/overrides \
    --bind "$SLURM_TMPDIR":/tmp \
    --env PYTHONPATH=/overrides:/workspace/eureka:/workspace/eureka/isaacgymenvs:/workspace/eureka/rl_games \
    --env LD_LIBRARY_PATH=/opt/conda/envs/eureka/lib \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env "GOOGLE_API_KEY=$GOOGLE_API_KEY" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "WANDB_ENTITY=${WANDB_ENTITY:-george-xue-university-of-toronto}" \
    "$SIF" \
    bash -lc 'conda run -n eureka --no-capture-output \
        python -u /workspace/eureka/eureka/reward_learning_pipeline.py \
            --task ShadowHandDoorOpenInward'
# headless=True is passed by the pipeline itself to the underlying train.py
# subprocess (reward_learning_pipeline.py:1033) — no need to specify here.
