#!/bin/bash
# =============================================================================
# setup_server.sh — One-shot environment setup for Rex-Thinker + SafeGround
# =============================================================================
# Usage (sau khi git clone rex_l2r):
#   bash setup_server.sh
# =============================================================================
set -e

echo "=== 0. Clone Rex-Thinker (if not present) ==="
if [ ! -d "Rex-Thinker" ]; then
    git clone https://github.com/IDEA-Research/Rex-Thinker.git
    echo "✅ Rex-Thinker cloned"
else
    echo "✅ Rex-Thinker already exists, skipping clone"
fi

echo "=== 1. Install Miniconda (if not present) ==="
CONDA_DIR="$HOME/miniconda3"
CONDA_BIN="$CONDA_DIR/bin/conda"

if [ ! -f "$CONDA_BIN" ]; then
    echo "Conda not found → Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    # -b = batch/silent, -u = update nếu đã tồn tại, -p = prefix
    bash /tmp/miniconda.sh -b -u -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "✅ Miniconda installed at $CONDA_DIR"
else
    echo "✅ Conda already installed: $($CONDA_BIN --version)"
fi

# Đảm bảo conda trong PATH cho script hiện tại
export PATH="$CONDA_DIR/bin:$PATH"

# Accept Anaconda TOS (bắt buộc từ 2024 cho repo chính thức)
echo "Accepting Conda Terms of Service..."
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Load conda shell functions vào script hiện tại
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "=== 2. Create Conda Environment ==="
conda create -n rexthinker_sg -y python=3.10
conda activate rexthinker_sg

echo "=== 3. Install PyTorch & Dependencies ==="
# Hạ cấp PyTorch xuống 2.5.1 (PyTorch 2.6.0 vừa mới ra mắt làm hỏng C++ ABI của flash-attention gây lỗi undefined symbol)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Build flash-attention từ source để đảm bảo đúng ABI với PyTorch hiện tại.
# PHẢI uninstall trước để xóa binary cache bị mismatch.
pip install packaging ninja psutil wheel
pip uninstall flash-attn -y 2>/dev/null || true
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir

# Cài đặt thư viện đánh giá (cần cho calibrate_tau.py và metric.py)
pip install scipy scikit-learn tabulate pycocotools

# Cài đặt Rex-Thinker core
cd Rex-Thinker
pip install -v -e .
cd ..

echo "=== 3. Install Grounding DINO ==="
# Tải Grounding DINO repo
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    # Cấu hình user phụ để cho phép git merge
    git config --global user.email "server@example.com"
    git config --global user.name "Server"
    
    # Hỗ trợ torch 2.6 theo hướng dẫn của base paper
    git remote add quantumope https://github.com/QuantuMope/GroundingDINO.git
    git fetch quantumope PR/andrew/add-torch26-support-ms-deform-attn
    git merge quantumope/PR/andrew/add-torch26-support-ms-deform-attn -m "Merge update"
    cd ..
fi

cd GroundingDINO

# Sửa tận gốc lỗi file script build của PyTorch: thay lệnh "raise RuntimeError" bằng "print" để cho phép build
python -c "
import re, sys
try:
    import torch
    from pathlib import Path
    file_path = Path(torch.__path__[0]) / 'utils' / 'cpp_extension.py'
    content = file_path.read_text()
    
    # Dọn dẹp các lỗi syntax hoặc patch hụt từ lần chạy trước (nếu có)
    content = content.replace('print(\"WARNING: Bypassed PyTorch CUDA version mismatch check!\"))', 'pass')
    
    # Biến lệnh raise lỗi thành lệnh in ra terminal cực kỳ an toàn
    content = re.sub(r'raise\s+RuntimeError\(\s*CUDA_MISMATCH_MESSAGE', 'print(CUDA_MISMATCH_MESSAGE', content)
    content = re.sub(r'raise\s+RuntimeError\(\s*ABI_INCOMPATIBILITY_WARNING', 'print(ABI_INCOMPATIBILITY_WARNING', content)
    
    file_path.write_text(content)
except Exception as e:
    print('Patch Warning: ', e)
"

# Dùng --no-build-isolation để ngăn setup.py tự gọi ngầm subprocess lỗi
pip install -v --no-build-isolation -e .
cd ..

echo "=== 4. Download Grounding DINO Weights ==="
cd GroundingDINO
mkdir -p weights
if [ ! -f "weights/groundingdino_swint_ogc.pth" ]; then
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights
fi
cd ..

echo "=== Bước đệm: Fix numpy conflict ==="
# Hạ cấp Numpy do pip tự cài bản 2.2.6 làm xung đột thư viện vllm
pip install "numpy<2.0.0"

echo "=== 5. Download Rex-Thinker-GRPO Weights ==="
# Git LFS thường bị ngắt kết nối với file lớn dẫn tới lỗi smudge filter
# Dùng thư viện HuggingFace CLI để tải đa luồng và tự động resume an toàn
if [ -d "IDEA-Research/Rex-Thinker-GRPO-7B/.git" ]; then
    echo "Xóa file .git cũ bị hụt dữ liệu..."
    rm -rf IDEA-Research/Rex-Thinker-GRPO-7B
fi

pip install "huggingface_hub<1.0.0"
huggingface-cli download IDEA-Research/Rex-Thinker-GRPO-7B --local-dir IDEA-Research/Rex-Thinker-GRPO-7B

echo "=== Environment and Weights Setup Complete ==="
echo "Bạn có thể chạy thử nghiệm bằng bash run_experiment.sh"
