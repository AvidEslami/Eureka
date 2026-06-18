# Eureka + VLM Docker Image for Compute Canada (Apptainer)
# Contains two conda environments:
#   - eureka (Python 3.8, torch 2.0.1+cu118, IsaacGym)
#   - vlm    (Python 3.10, google-genai for Gemini API)

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (Vulkan/EGL needed for IsaacGym headless rendering on H100)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    xvfb \
    libvulkan1 vulkan-tools libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda for bash (needed for conda run)
RUN conda init bash && \
    conda config --set auto_activate_base false

# Accept Anaconda's channel ToS (required since 2024 for the default channels;
# without this `conda create` errors with CondaToSNonInteractiveError).
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- eureka environment (Python 3.8, CUDA 11.8) ---
RUN conda create -n eureka python=3.8 -y

# Install PyTorch 2.0.1 with CUDA 11.8 (H100 sm_90 support)
RUN conda run -n eureka pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install eureka pip dependencies
COPY docker/requirements_eureka.txt /tmp/requirements_eureka.txt
RUN conda run -n eureka pip install --no-cache-dir -r /tmp/requirements_eureka.txt

# --- vlm environment (Python 3.10, lightweight) ---
RUN conda create -n vlm python=3.10 -y

COPY docker/requirements_vlm.txt /tmp/requirements_vlm.txt
RUN conda run -n vlm pip install --no-cache-dir -r /tmp/requirements_vlm.txt

# --- Copy IsaacGym (pre-compiled bindings) — baked because the .so files
#     are awkward to bind-mount and rarely change ---
COPY isaacgym_pkg/ /workspace/isaacgym/python/

# Install IsaacGym editable in the eureka env
RUN conda run -n eureka pip install -e /workspace/isaacgym/python/

# Eureka, isaacgymenvs, and rl_games source is NOT baked — it is bind-mounted
# at /workspace/eureka by job.sh, with PYTHONPATH set so that
#   import isaacgymenvs / import rl_games / import eureka_module
# resolve against the mount. None of the three define entry-point scripts or
# use pkg_resources metadata, so pip install -e isn't needed.

WORKDIR /workspace/eureka/eureka

# No ENTRYPOINT — apptainer exec supplies the command. Avoids double-wrapping
# conda when the job script already calls `conda run -n eureka ...`.
CMD ["bash"]
