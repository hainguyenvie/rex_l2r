#!/bin/bash
set -e

# Load conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rexthinker_sg

# Cấu hình tham số ngưỡng từ chối mẫu
TAU=0.15

echo "=== 1. Chạy quá trình Suy Luận Rex-Thinker với UQ (K=10 Samples) ==="
cd Rex-Thinker
CUDA_VISIBLE_DEVICES=0 python demo/inference_safeground.py \
  --image_path demo/example_images/demo_helmet.png \
  --cate_name helmet \
  --ref_exp "the forth helmet from left" \
  --vis_path vis/example_output_sg.jpg \
  --num_samples 10 \
  --temperature 0.7 \
  --tau $TAU \
  --model_path ../IDEA-Research/Rex-Thinker-GRPO-7B \
  --gdino_config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --gdino_weights ../GroundingDINO/weights/groundingdino_swint_ogc.pth

echo "=== 2. Mô phỏng File Kết quả Evaluation Dataset (JSONL) ==="
# Đây là file JSONL mô phỏng do code script tự động tạo ra trong quá trình loop dataset 
# Nếu UCOM lơn -> tỷ lệ dự đoán sai càng cao. Label correct giả định.
echo '{"image_id": 1, "ucom": 0.05, "correct": true}' > results.jsonl
echo '{"image_id": 2, "ucom": 0.12, "correct": true}' >> results.jsonl
echo '{"image_id": 3, "ucom": 0.35, "correct": false}' >> results.jsonl
echo '{"image_id": 4, "ucom": 0.45, "correct": false}' >> results.jsonl

echo "=== 3. Chạy Hàm Hiệu Chuẩn Threshold (Calibration) ==="
python evaluation/calibrate_tau.py \
  --results_file results.jsonl \
  --target_fdr 0.05 \
  --alpha_conf 0.1

echo "Done! Hãy kiểm tra kết quả ảnh render tại Rex-Thinker/vis/example_output_sg.jpg"
