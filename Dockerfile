# Modern base image matching your CUDA 12 stack
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, and compilation essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set the working directory
WORKDIR /workspace

# Copy the curated requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project codebase
COPY . .

# (Optional) Compile LKH solver if you are running classical baselines
# RUN cd LKH-3.0.6 && make

# Default entrypoint
CMD ["/bin/bash"]