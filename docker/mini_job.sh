#!/bin/bash
#SBATCH --job-name=eureka-mini
#SBATCH --account=def-rhinehar
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Mini Eureka run: 1 iteration, ~30 RL epochs, 3 checkpoints, preference mode.
# Used to validate the cluster setup end-to-end (container + GPU + wandb +
# Gemini API). Targets ~10 min of actual compute; --time=00:30:00 is a safety
# ceiling. NOT intended for real training.

set -euo pipefail

module load apptainer

: "${PROJECT:=$HOME/projects/def-rhinehar}"
: "${SCRATCH:=$HOME/scratch}"
export PROJECT SCRATCH

SIF="$PROJECT/containers/eureka.sif"
CODE="$PROJECT/eureka/src"
OUT="$SCRATCH/eureka"

# shellcheck disable=SC1090
source "$HOME/.eureka_env"

echo "=== Node $(hostname) GPU ==="
apptainer exec --nv "$SIF" nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "============================"

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
            --task ShadowHandDoorOpenInward \
            --reward_type ground_truth \
            --num_iterations 1 \
            --rl_epochs 30 \
            --save_frequency 10 \
            --checkpoint_start 10 \
            --checkpoint_step 10 \
            --checkpoint_end 30'
