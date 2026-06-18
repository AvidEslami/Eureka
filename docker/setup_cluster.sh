#!/bin/bash
# One-time cluster bootstrap. Run on the Nibi login node after first SSH:
#
#   ssh gxue@nibi.alliancecan.ca
#   bash setup_cluster.sh        # this script, copied or pasted via heredoc
#
# Idempotent — safe to re-run. Does NOT rsync data from your laptop; that
# step is printed at the end with the exact command to run from your laptop.

set -euo pipefail

# --- Sanity: $PROJECT and $SCRATCH ---
# $PROJECT is not auto-exported on Nibi; resolve via the symlink in
# ~/projects/. $SCRATCH is set as a symlink at ~/scratch.
if [ -z "${PROJECT:-}" ]; then
    if [ -d "$HOME/projects/def-rhinehar" ]; then
        export PROJECT="$HOME/projects/def-rhinehar"
        echo "Resolved \$PROJECT = $PROJECT"
    else
        echo "ERROR: cannot resolve \$PROJECT. Expected ~/projects/def-rhinehar to exist." >&2
        exit 1
    fi
fi
if [ -z "${SCRATCH:-}" ]; then
    if [ -d "$HOME/scratch" ]; then
        export SCRATCH="$HOME/scratch"
        echo "Resolved \$SCRATCH = $SCRATCH"
    else
        echo "ERROR: cannot resolve \$SCRATCH. Expected ~/scratch to exist." >&2
        exit 1
    fi
fi

EUREKA_REPO="https://github.com/AvidEslami/Eureka.git"
EUREKA_BRANCH="feat/wandb-and-rollout-tasks"
SRC_DIR="$PROJECT/eureka/src"
DATA_DIR="$PROJECT/eureka/data"
CONT_DIR="$PROJECT/containers"
OUT_DIR="$SCRATCH/eureka"

echo "=== Creating directories ==="
mkdir -p "$CONT_DIR" "$DATA_DIR" "$OUT_DIR/experiments" "$OUT_DIR/wandb"

echo "=== Cloning Eureka repo to $SRC_DIR ==="
if [ -d "$SRC_DIR/.git" ]; then
    echo "Repo already cloned. Pulling latest on $EUREKA_BRANCH ..."
    git -C "$SRC_DIR" fetch origin "$EUREKA_BRANCH"
    git -C "$SRC_DIR" checkout "$EUREKA_BRANCH"
    git -C "$SRC_DIR" pull --ff-only origin "$EUREKA_BRANCH"
else
    git clone --branch "$EUREKA_BRANCH" "$EUREKA_REPO" "$SRC_DIR"
fi

echo "=== Wiring symlinks: outputs -> \$SCRATCH ==="
# experiments/ and wandb/ in the source tree point at $SCRATCH so the pipeline
# writes to fast storage while still resolving its default EUREKA_DIR paths.
for sub in experiments wandb; do
    target_in_src="$SRC_DIR/eureka/$sub"
    target_in_scratch="$OUT_DIR/$sub"
    # If a real (non-symlink) dir already exists in the source tree, move it out
    # of the way to avoid silent shadowing.
    if [ -e "$target_in_src" ] && [ ! -L "$target_in_src" ]; then
        echo "  $target_in_src exists as a real dir; renaming to .pre_symlink"
        mv "$target_in_src" "${target_in_src}.pre_symlink.$(date +%s)"
    fi
    ln -sfn "$target_in_scratch" "$target_in_src"
    echo "  $target_in_src -> $target_in_scratch"
done

echo "=== Creating ~/.eureka_env skeleton (if missing) ==="
if [ ! -f "$HOME/.eureka_env" ]; then
    cat > "$HOME/.eureka_env" <<'EOF'
# Fill these in. Sourced by docker/job.sh before apptainer exec.
# Get GOOGLE_API_KEY from https://aistudio.google.com/apikey
# Get WANDB_API_KEY  from https://wandb.ai/authorize
export GOOGLE_API_KEY=
export WANDB_API_KEY=
# Optional: override the default wandb entity (default: george-xue-university-of-toronto)
# export WANDB_ENTITY=
EOF
    chmod 600 "$HOME/.eureka_env"
    echo "  Created $HOME/.eureka_env (chmod 600)"
else
    echo "  $HOME/.eureka_env already exists; not overwriting"
fi

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "TODO from here:"
echo ""
echo "1. Fill in $HOME/.eureka_env with your API keys."
echo ""
echo "2. From your LAPTOP, rsync the preference data to \$PROJECT:"
echo "     rsync -avh --partial --info=progress2 \\"
echo "       /home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data_open_inward \\"
echo "       /home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data_open_outward \\"
echo "       gxue@nibi.alliancecan.ca:$DATA_DIR/"
echo ""
echo "3. Back on Nibi, symlink the data into the source tree:"
echo "     ln -sfn $DATA_DIR/auto_preference_data_open_inward  $SRC_DIR/eureka/auto_preference_data_open_inward"
echo "     ln -sfn $DATA_DIR/auto_preference_data_open_outward $SRC_DIR/eureka/auto_preference_data_open_outward"
echo ""
echo "4. From your LAPTOP, build + ship the container image:"
echo "     cd /home/gx22/Desktop/isaacgym/python/Eureka"
echo "     bash docker/build.sh"
echo "     docker save eureka:nibi-cu118 | gzip > eureka.tar.gz"
echo "     rsync -avh --partial --append-verify --info=progress2 \\"
echo "       eureka.tar.gz gxue@nibi.alliancecan.ca:$CONT_DIR/"
echo ""
echo "5. Back on Nibi, build the .sif:"
echo "     module load apptainer"
echo "     cd $CONT_DIR"
echo "     apptainer build eureka.sif docker-archive://eureka.tar.gz"
echo ""
echo "6. Smoke test inside an interactive allocation:"
echo "     salloc --account=def-rhinehar --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=1:00:00"
echo "     bash $SRC_DIR/docker/smoke_test.sh"
echo ""
echo "7. Submit the real job:"
echo "     sbatch $SRC_DIR/docker/job.sh"
