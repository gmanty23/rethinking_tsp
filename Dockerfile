# Modern base image matching the CUDA 12 stack requested by your PyTorch 2.9.1 bindings
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies required for Conda, downloading, and C++ compilation (LKH solver)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH globally
ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory
WORKDIR /workspace

# Copy the conda environment file into the container
COPY environment.yml .

# NEW: Automatically accept Anaconda Terms of Service for default channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment exactly as defined
RUN conda env create -f environment.yml

# Ensure subsequent RUN commands in this Dockerfile use the new environment
SHELL ["conda", "run", "-n", "tsp_modern", "/bin/bash", "-c"]

# Copy the rest of the project codebase
# COPY . .
#COPY . .

# Set the entrypoint to automatically activate the environment when you run the container
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tsp_modern"]
CMD ["/bin/bash"]




# docker run --gpus all -it --rm --ipc=host \
#     -v $(pwd):/workspace \
#     windy-tsp-modern:latest bash