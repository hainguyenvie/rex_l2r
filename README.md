# rex_l2r — Rex-Thinker + SafeGround UQ

Research pipeline: bổ sung **Uncertainty-Aware Selective Prediction** (FDR-guaranteed rejection) vào Rex-Thinker Visual Grounding model.

## Chạy nhanh

```bash
# Bước 1: Clone repo
git clone <your-repo-url>
cd rex_l2r

# Bước 2: Setup môi trường + tải toàn bộ weights (~30-60 phút)
bash setup_server.sh

# Bước 3: Chạy full experiment lấy kết quả cho paper
bash run_experiment.sh
```

> ⚠️ Yêu cầu: GPU với CUDA 12.4+, RAM ≥ 40GB VRAM (H100/A100 recommended), Conda đã cài.

---

## `setup_server.sh` làm gì?

| Bước | Nội dung |
|------|---------|
| 0 | Clone Rex-Thinker từ IDEA-Research/Rex-Thinker |
| 1 | Tạo conda env `rexthinker_sg` (Python 3.10) |
| 2 | Cài PyTorch 2.6.0 + CUDA 12.4 + flash-attn + eval libs |
| 3 | Clone + build GroundingDINO (với torch 2.6 patch) |
| 4 | Download weights GroundingDINO (swint_ogc.pth) |
| - | Fix numpy conflict (downgrade < 2.0.0) |
| 5 | Download Rex-Thinker-GRPO-7B từ HuggingFace |

## `run_experiment.sh` làm gì?

| Bước | Nội dung | Output |
|------|---------|--------|
| 1 | Download HumanRef dataset | `data/IDEA-Research/HumanRef/` |
| 2 | Inference K=8 + UQ (UCOM) trên toàn dataset | `results/.../eval_sg_preds.jsonl` + `eval_sg_ucom.jsonl` |
| 3 | Compute Recall/Precision/DensityF1 | `results/.../metric_output/comparison.md` |
| 4 | Merge + Calibrate tau → AUROC/AUARC | Printed to stdout |

## Cấu trúc repo

```
rex_l2r/
├── setup_server.sh              ← Chạy đầu tiên
├── run_experiment.sh            ← Chạy sau setup
├── plan.md                      ← Báo cáo khảo sát gốc
├── plan_revised.md              ← Kế hoạch paper (revised)
├── rex_thinker.md               ← Paper Rex-Thinker đầy đủ
├── safeguardpaper.md            ← Paper SafeGround đầy đủ
└── Rex-Thinker/
    ├── demo/
    │   ├── discrete_uq.py       ← UCOM computation (TA/IE/CD)
    │   └── inference_safeground.py  ← Single image inference + UQ
    └── evaluation/
        ├── eval_safeground.py   ← Full dataset eval + UQ [MAIN]
        ├── calibrate_tau.py     ← Clopper-Pearson FDR calibration
        └── metric.py            ← Recall/Precision/DF1 computation
```

## Sharding (chạy trên nhiều GPU)

```bash
# GPU 0: sample 0-3000
START_IDX=0    END_IDX=3000 CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh &

# GPU 1: sample 3000-6001
START_IDX=3000 END_IDX=6001 CUDA_VISIBLE_DEVICES=1 bash run_experiment.sh &

wait
echo "Both shards done"
```
