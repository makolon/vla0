# Base image: NVIDIA CUDA 12.4.1 with development tools on Ubuntu 20.04
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# === Log Step 2: Install system dependencies ===
# This command first updates the list of available packages from the configured software repositories.
# If the update is successful, it then proceeds to install a series of software packages.
# - git: version control
# - wget: network downloader
# - curl: data transfer utility
# - vim: text editor
# - cmake: build system generator
# - ffmpeg: video processing for datasets and evals
# - libglib2.0-0: a core GLib library, often a dependency for software like OpenCV/CV2
# - libgl1-mesa-glx: Mesa OpenGL/GLX runtime library, for graphics, often required by MuJoCo
# - libosmesa6-dev: Mesa off-screen rendering development libraries, for MuJoCo headless rendering
# - libglew-dev: OpenGL Extension Wrangler Library development files; resolves <GL/glew.h> errors for mujoco_py
# - libegl1-mesa: Mesa EGL runtime library; resolves libEGL.so.1 errors for mujoco_py
# - libopengl0: Vendor-neutral GLdispatch library for OpenGL; resolves libOpenGL.so.0 errors for mujoco_py
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    cmake \
    ffmpeg \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libegl1-mesa \
    libopengl0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# === Log Step 4: Configure Environment Variables ===
# MUJOCO_GL: Instructs MuJoCo to use EGL for rendering.
ENV MUJOCO_GL=egl

# PATH: Add uv and CUDA to PATH.
ENV PATH /root/.local/bin:/usr/local/cuda/bin:$PATH

# LD_LIBRARY_PATH: Add MuJoCo to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Install uv for fast, reproducible Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# === Log Step 6 & 7: Project Setup and Installation ===
# Argument for the project directory name. This name will be used for the
# project's directory inside ${APP_DIR} (e.g., /app/RV-train if ARG is RV-train).
ARG PROJECT_DIR_NAME=RV-train
# Base directory for applications within the container.
ENV APP_DIR /app

# Copy the project into the container
COPY . ${APP_DIR}/${PROJECT_DIR_NAME}

# Set working directory to the project root inside the container.
# This path (e.g., /app/RV-train) now contains the project files.
WORKDIR ${APP_DIR}/${PROJECT_DIR_NAME}

# Create a uv-managed virtual environment (Python 3.10) and use it by default
ENV VIRTUAL_ENV=${APP_DIR}/${PROJECT_DIR_NAME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH
RUN uv venv --python 3.10 ${VIRTUAL_ENV}

# Install all locked dependencies (including extras) and the main project in editable mode
RUN uv sync --python ${VIRTUAL_ENV}/bin/python --frozen --all-extras
RUN echo "Current directory: $(pwd)" && \
    uv pip install --python ${VIRTUAL_ENV}/bin/python --no-deps -e ".[all]"

# Log Step 7: Install the RoboVerse library from within the project.
# This installs the RoboVerse library, located in ./libs/RoboVerse/ relative to
# the project root, in editable mode, with all optional dependencies.
RUN echo "Current directory: $(pwd)" && \
    uv pip install --python ${VIRTUAL_ENV}/bin/python -e "./libs/RoboVerse[all]"

# Remove .git directory to save space
RUN rm -rf .git
RUN rm -rf ./libs/RoboVerse/.git

# === Log Step 8: Install specific bitsandbytes version ===
# This command installs a specific pre-compiled wheel of bitsandbytes.
# This was to resolve a GLIBC version incompatibility issue, as the
# manylinux_2_24 wheel is compatible with older GLIBC versions like 2.31 (on Ubuntu 20.04).
RUN UV_SKIP_WHEEL_FILENAME_CHECK=1 \
    uv pip install --python ${VIRTUAL_ENV}/bin/python --force-reinstall \
    https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl

# === Log Step 10: Configure EGL for NVIDIA ===
# This step manually creates and configures a JSON file for NVIDIA's EGL driver.
# This is often a critical fix for mujoco_py when it fails to initialize OpenGL
# on systems with NVIDIA GPUs, especially when MUJOCO_GL=egl is set.
# Such failures can manifest as "RuntimeError: Failed to initialize OpenGL".
RUN echo '{ \
    "file_format_version" : "1.0.0", \
    "ICD" : { \
        "library_path" : "libEGL_nvidia.so.0" \
    } \
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    chmod 644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json
