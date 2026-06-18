#!/bin/bash
# Build the Eureka Docker image.
# Run from the Eureka repo root: bash docker/build.sh
#
# This script stages the necessary files into a temporary build context
# because Docker needs isaacgym (parent dir) + Eureka (this repo) together.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EUREKA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACGYM_ROOT="$(cd "$EUREKA_ROOT/.." && pwd)"  # /home/.../isaacgym/python/ (contains setup.py + isaacgym/)

echo "=== Eureka Docker Build ==="
echo "Eureka root: $EUREKA_ROOT"
echo "IsaacGym root: $ISAACGYM_ROOT"

# Create a temporary build context
BUILD_CTX=$(mktemp -d)
trap "rm -rf $BUILD_CTX" EXIT

echo "Build context: $BUILD_CTX"

# Copy Dockerfile and requirements
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_CTX/"
mkdir -p "$BUILD_CTX/docker"
cp "$SCRIPT_DIR/requirements_eureka.txt" "$BUILD_CTX/docker/"
cp "$SCRIPT_DIR/requirements_vlm.txt" "$BUILD_CTX/docker/"

# Copy IsaacGym package (the python/ directory with setup.py + isaacgym/)
# This is the only project source baked into the image. Eureka/isaacgymenvs/
# rl_games are bind-mounted from $PROJECT/eureka/src on the cluster at job
# start (see docker/job.sh + PYTHONPATH wiring).
echo "Copying IsaacGym package (~438MB)..."
mkdir -p "$BUILD_CTX/isaacgym_pkg"
cp "$ISAACGYM_ROOT/setup.py" "$BUILD_CTX/isaacgym_pkg/"
cp -r "$ISAACGYM_ROOT/isaacgym" "$BUILD_CTX/isaacgym_pkg/isaacgym"

echo "Build context size: $(du -sh $BUILD_CTX | cut -f1)"

# Build the image
echo "Building Docker image..."
IMAGE_TAG="eureka:nibi-cu118"
docker build -t "$IMAGE_TAG" "$BUILD_CTX"

echo ""
echo "=== Build complete ==="
echo "Image: $IMAGE_TAG"
echo ""
echo "Next steps (no registry needed — ships via tarball):"
echo "  1. Save image to a compressed tarball:"
echo "       docker save $IMAGE_TAG | gzip > eureka.tar.gz"
echo "  2. Rsync to Nibi (resumable):"
echo "       rsync -avh --partial --append-verify --info=progress2 \\"
echo "             eureka.tar.gz gxue@nibi.alliancecan.ca:\$PROJECT/containers/"
echo "  3. On Nibi login node, build the .sif:"
echo "       module load apptainer"
echo "       cd \$PROJECT/containers"
echo "       apptainer build eureka.sif docker-archive://eureka.tar.gz"
echo ""
echo "First-time cluster bootstrap: run docker/setup_cluster.sh on the login node."
echo "Smoke test inside salloc: docker/smoke_test.sh."
