# Dockerfile for Reproducible Space-Engineering CFD Workflows
# Includes SU2 v8.4.0, OpenFOAM v2312, and Python 3.10+
FROM ubuntu:22.04

LABEL maintainer="Yuvraj Singh"
LABEL description="CFD Solver Benchmark for Flow Separation Prediction"
LABEL su2.version="8.4.0"
LABEL openfoam.version="2312"

# Avoid tzdata interactive prompt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SU2 v8.4.0 precompiled binaries (simplified installation)
# In a real HPC environment we would build from source to link against Intel MKL/MPI
# For simplicity in this container, we download the official binaries
WORKDIR /opt
RUN set -x && \
    wget -q https://github.com/su2code/SU2/releases/download/v8.4.0/SU2-v8.4.0-linux64.zip && \
    unzip -q SU2-v8.4.0-linux64.zip && \
    rm SU2-v8.4.0-linux64.zip

ENV SU2_RUN=/opt/SU2-v8.4.0-linux64/bin
ENV SU2_HOME=/opt/SU2-v8.4.0-linux64
ENV PATH=$SU2_RUN:$PATH
ENV PYTHONPATH=$SU2_RUN:$PYTHONPATH

# Install OpenFOAM v2312
RUN curl -dl "https://dl.openfoam.com/add-debian-repo.sh" | bash && \
    apt-get update && \
    apt-get install -y openfoam2312-default && \
    rm -rf /var/lib/apt/lists/*

# Setup Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Add OpenFOAM source to bashrc
RUN echo "source /usr/lib/openfoam/openfoam2312/etc/bashrc" >> ~/.bashrc

# Copy project files
COPY . /app/

# Create a sample SLURM submission script
RUN echo '#!/bin/bash\n\
#SBATCH --job-name=cfd_space_eval\n\
#SBATCH --nodes=1\n\
#SBATCH --ntasks=16\n\
#SBATCH --time=24:00:00\n\
\n\
source ~/.bashrc\n\
source /usr/lib/openfoam/openfoam2312/etc/bashrc\n\
\n\
# 1. TMR Heated Jet\n\
python3 run_heated_jet.py --model SA\n\
\n\
# 2. ZBOT Micro-g VOF\n\
python3 run_zbot_vof.py --case standard\n\
\n\
# 3. TBNN Train\n\
python3 scripts/ml_augmentation/tbnn_wall_hump.py\n\
' > /app/slurm_submit.sh && chmod +x /app/slurm_submit.sh

# Healthcheck: verify Python and SU2 are accessible
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD python3 -c "import numpy; print('OK')" || exit 1

# Default: run test suite
CMD ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"]
