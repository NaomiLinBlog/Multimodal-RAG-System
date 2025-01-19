#!/bin/bash

# 更新套件列表並安裝系統依賴
echo "Updating package list and installing system dependencies..."
sudo apt-get update
sudo apt-get install -y poppler-utils

# 移除舊環境（如果存在）
echo "Removing old environment if exists..."
conda remove -n multimodal-rag2 --all -y

# 創建新的 conda 環境
echo "Creating new conda environment..."
conda create -n multimodal-rag2 python=3.10.16 -y

# 啟動環境
echo "Activating environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate multimodal-rag2

# 安裝基礎套件
echo "Installing PyTorch..."
pip install torch>=2.4.1

echo "Installing transformers and huggingface-hub..."
pip install "huggingface-hub>=0.24.0,<1.0"
pip install transformers==4.48.0

echo "Installing additional required packages..."
pip install -r <(cat << 'EOF'
accelerate>=1.0.1
bitsandbytes>=0.45.0
byaldi
flash-attn
sentence-transformers>=2.2.2
PyMuPDF>=1.23.8
python-multipart
pypdf>=3.0.0
pdf2image
numpy>=1.24.4
pandas>=2.0.3
scikit-learn>=1.3.2
fastapi>=0.115.4
uvicorn>=0.32.0
pydantic>=2.9.2
llama-index>=0.9.48
llama-index-embeddings-huggingface>=0.1.0
llama-index-llms-huggingface>=0.1.0
llama-index-readers-file>=0.1.0
tqdm>=4.65.0
pillow>=10.0.0
requests>=2.31.0
python-dateutil>=2.8.2
PyYAML>=6.0.1
EOF
)

# 驗證安裝
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes version: {bitsandbytes.__version__}')"

echo "Installation complete!"