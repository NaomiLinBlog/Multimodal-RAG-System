#!/bin/bash

# Multimodal RAG Environment Setup Script
# Version 1.0

# Exit on any error
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Ensure script is run with bash
if [ -z "$BASH_VERSION" ]
then
    print_error "Please run this script with bash"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null
then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Environment name
ENV_NAME="multimodal-rag3"

# 1. Remove existing environment if it exists
echo -e "${YELLOW}Checking and removing existing environment if present...${NC}"
conda info --envs | grep -q "$ENV_NAME" && conda remove -n "$ENV_NAME" --all -y || true

# 2. Create new conda environment
echo -e "${YELLOW}Creating new conda environment...${NC}"
conda create -n "$ENV_NAME" python=3.10.16 -y
print_status "Conda environment created successfully"

# 3. Activate the environment
echo -e "${YELLOW}Activating the environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
print_status "Environment activated"

# 4. Update system packages and install dev tools
echo -e "${YELLOW}Updating system packages and installing development tools...${NC}"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    poppler-utils

# 5. Ensure pip is up to date
pip install --upgrade pip setuptools wheel
print_status "Pip and basic tools updated"

# 6. Install core ML and data science packages
echo -e "${YELLOW}Installing core ML and data science packages...${NC}"
pip install \
    numpy \
    pandas \
    scikit-learn \
    torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 7. Install NLP and Transformer packages
echo -e "${YELLOW}Installing NLP and Transformer packages...${NC}"
pip install \
    "huggingface-hub>=0.24.0,<0.25.0" \
    transformers==4.48.0 \
    accelerate \
    bitsandbytes \
    peft \
    sentence-transformers \
    tokenizers

# 8. Install LLaMA Index related packages
echo -e "${YELLOW}Installing LLaMA Index packages...${NC}"
pip install \
    llama-index \
    llama-index-embeddings-huggingface \
    llama-index-llms-huggingface \
    llama-index-readers-file

# 9. Install additional utility packages
echo -e "${YELLOW}Installing additional utility packages...${NC}"
pip install \
    openai \
    tiktoken \
    pypdf \
    python-multipart \
    fastapi \
    uvicorn \
    pydantic \
    httpx \
    tqdm \
    pillow \
    requests \
    python-dateutil \
    PyYAML

# 10. Fix potential dependency conflicts
echo -e "${YELLOW}Resolving potential dependency conflicts...${NC}"
pip install \
    fsspec==2023.4.0 \
    requests_mock \
    clyent==1.2.1 \
    "PyYAML==6.0.1"

# 11. Optional: Install optional packages like flash-attn
#echo -e "${YELLOW}Attempting to install optional packages...${NC}"
#pip install flash-attn --no-build-isolation || print_warning "Flash-attn installation may have issues"

# 12. Install Triton for potential bitsandbytes compatibility
pip install triton || print_warning "Triton installation may have issues"

# 13. Verification step
echo -e "${YELLOW}Verifying installations...${NC}"
python -c "
import torch; print(f'PyTorch version: {torch.__version__}')
import transformers; print(f'Transformers version: {transformers.__version__}')
import bitsandbytes; print(f'BitsAndBytes version: {bitsandbytes.__version__}')
import huggingface_hub; print(f'Hugging Face Hub version: {huggingface_hub.__version__}')
"

# Final success message
print_status "Multimodal RAG environment setup completed successfully!"
print_warning "Remember to activate the environment with: conda activate $ENV_NAME"

# Optional cleanup
pip cache purge

echo -e "${GREEN}Setup complete! You can now start working on your Multimodal RAG project.${NC}"
