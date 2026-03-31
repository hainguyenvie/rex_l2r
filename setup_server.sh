#!/bin/bash
set -e

echo "=== 1. Create Conda Environment ==="
# Cài đặt môi trường conda mới
conda create -n rexthinker_sg -y python=3.10
# Kích hoạt conda trong bash script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rexthinker_sg

echo "=== 2. Install PyTorch & Dependencies ==="
# Cài đặt PyTorch hỗ trợ GPU
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Sửa lỗi build flash-attention (Né lỗi thiếu torch do pip bị cô lập môi trường chuẩn PEP 517)
# Khi dùng --no-build-isolation, ta phải tự cung cấp mọi thư viện phụ thuộc cho quá trình build.
pip install packaging ninja psutil wheel
pip install flash-attn --no-build-isolation

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

echo "=== 5. Download Rex-Thinker-GRPO Weights ==="
git lfs install
if [ ! -d "IDEA-Research/Rex-Thinker-GRPO-7B" ]; then
    mkdir -p IDEA-Research
    git clone https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B IDEA-Research/Rex-Thinker-GRPO-7B
fi

echo "=== Environment and Weights Setup Complete ==="
echo "Bạn có thể chạy thử nghiệm bằng bash run_experiment.sh"
