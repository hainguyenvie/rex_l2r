#!/bin/bash
# =============================================================================
# run_experiment.sh — Full Pipeline: Rex-Thinker-GRPO + SafeGround UQ
# =============================================================================
# Pipeline:
#   Step 1: Download HumanRef dataset (nếu chưa có)
#   Step 2: Run eval_safeground.py (K=8 samples, UQ, save preds + ucom)
#   Step 3: Compute metric.py (Recall / Precision / DF1)
#   Step 4: Merge ucom_results + correctness labels → calibrate_tau.py
# =============================================================================
set -e

# ── Activate conda env ────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rexthinker_sg

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH="../IDEA-Research/Rex-Thinker-GRPO-7B"
ANNO_PATH="data/IDEA-Research/HumanRef/annotations.jsonl"
IMAGE_DIR="data/IDEA-Research/HumanRef/images"
GDINO_CFG="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_W="../GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAVE_DIR="results/safeground_run1"
NUM_SAMPLES=8       # == GRPO rollout samples (aligned with paper)
TEMPERATURE=0.7
TARGET_FDR=0.05
ALPHA_CONF=0.1

cd Rex-Thinker

# =============================================================================
echo "=== Step 1: Download HumanRef Dataset (if not present) ==="
# =============================================================================
if [ ! -f "$ANNO_PATH" ]; then
    echo "HumanRef not found → Downloading from HuggingFace..."
    mkdir -p data/IDEA-Research
    
    # Download annotations + images via huggingface-cli
    huggingface-cli download IDEA-Research/HumanRef \
        --repo-type dataset \
        --local-dir data/IDEA-Research/HumanRef
    
    echo "✅ HumanRef downloaded to data/IDEA-Research/HumanRef/"
else
    echo "✅ HumanRef already exists at $ANNO_PATH"
fi

# Verify annotation file
echo "Annotation samples: $(wc -l < $ANNO_PATH)"

# =============================================================================
echo ""
echo "=== Step 2: Run Inference + UQ (K=$NUM_SAMPLES samples per query) ==="
echo "    → Model: Rex-Thinker-GRPO-7B"
echo "    → UQ: UCOM = 0.4*TA + 0.3*IE + 0.3*CD (SafeGround)"
echo "    → Output: $SAVE_DIR/eval_sg_preds.jsonl + eval_sg_ucom.jsonl"
# =============================================================================

# Support sharding: set START_IDX and END_IDX env vars to parallelize across GPUs
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-6001}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python evaluation/eval_safeground.py \
    --model_path       "$MODEL_PATH"  \
    --anno_path        "$ANNO_PATH"   \
    --image_root_dir   "$IMAGE_DIR"   \
    --save_dir         "$SAVE_DIR"    \
    --gdino_config     "$GDINO_CFG"   \
    --gdino_weights    "$GDINO_W"     \
    --num_samples      $NUM_SAMPLES   \
    --temperature      $TEMPERATURE   \
    --start_idx        $START_IDX     \
    --end_idx          $END_IDX

# =============================================================================
echo ""
echo "=== Step 3: Compute Grounding Accuracy Metrics ==="
echo "    → Recall / Precision / DensityF1 (same as Table 2 in paper)"
# =============================================================================

python evaluation/metric.py \
    --gt_path    "$ANNO_PATH" \
    --pred_path  "$SAVE_DIR/eval_sg_preds.jsonl" \
    --pred_names "Rex-Thinker-GRPO-SafeGround" \
    --dump_path  "$SAVE_DIR/metric_output"

echo "✅ Table saved to $SAVE_DIR/metric_output/comparison.md"

# =============================================================================
echo ""
echo "=== Step 4: Merge UCOM + Correctness → Calibrate Tau ==="
echo "    → AUROC, AUARC, Optimal tau for FDR <= $TARGET_FDR"
# =============================================================================

# Merge eval_sg_preds.jsonl (has 'extracted_predictions') with
# annotations.jsonl (has 'answer_boxes') to compute correctness labels.
python3 - << 'EOF'
import json, sys

def calculate_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter)

save_dir = "results/safeground_run1"
anno_path = "data/IDEA-Research/HumanRef/annotations.jsonl"

# Load ground truth
gt = {}
with open(anno_path) as f:
    for line in f:
        item = json.loads(line)
        gt[item["id"]] = item

# Load predictions
preds = {}
with open(f"{save_dir}/eval_sg_preds.jsonl") as f:
    for line in f:
        item = json.loads(line)
        preds[item["id"]] = item

# Load ucom scores
ucoms = {}
with open(f"{save_dir}/eval_sg_ucom.jsonl") as f:
    for line in f:
        item = json.loads(line)
        ucoms[item["id"]] = item

# Merge: compute correctness (IoU @ 0.5)
merged = []
for sid, ucom_item in ucoms.items():
    if sid not in gt or sid not in preds:
        continue
    gt_item = gt[sid]
    pred_boxes = preds[sid].get("extracted_predictions", [])
    gt_boxes   = gt_item.get("answer_boxes", [])
    domain     = gt_item.get("domain", "unknown")

    # Rejection domain: correct if pred is empty
    if domain == "rejection":
        correct = (len(pred_boxes) == 0)
    elif len(gt_boxes) == 0:
        correct = True
    elif len(pred_boxes) == 0:
        correct = False
    else:
        # Check if best prediction matches any gt at IoU >= 0.5
        best_iou = max(
            calculate_iou(p, g)
            for p in pred_boxes
            for g in gt_boxes
        )
        correct = (best_iou >= 0.5)

    merged.append({
        "id": sid,
        "ucom": ucom_item["ucom"],
        "correct": bool(correct),
        "domain": domain,
        "ta": ucom_item.get("ta", 0),
        "ie": ucom_item.get("ie", 0),
        "cd": ucom_item.get("cd", 0),
    })

out_path = f"{save_dir}/eval_sg_merged.jsonl"
with open(out_path, "w") as f:
    for item in merged:
        f.write(json.dumps(item) + "\n")

n_correct = sum(1 for x in merged if x["correct"])
print(f"Merged {len(merged)} samples → {out_path}")
print(f"Overall accuracy: {n_correct}/{len(merged)} = {n_correct/max(1,len(merged)):.1%}")

# Per-domain summary
from collections import defaultdict
domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
for item in merged:
    d = item["domain"]
    domain_stats[d]["total"] += 1
    if item["correct"]:
        domain_stats[d]["correct"] += 1

print("\nPer-domain accuracy:")
for d, stats in sorted(domain_stats.items()):
    acc = stats["correct"] / max(1, stats["total"])
    print(f"  {d:20s}: {stats['correct']:4d}/{stats['total']:4d} = {acc:.1%}")
EOF

python evaluation/calibrate_tau.py \
    --results_file "$SAVE_DIR/eval_sg_merged.jsonl" \
    --target_fdr   $TARGET_FDR \
    --alpha_conf   $ALPHA_CONF

# =============================================================================
echo ""
echo "==================================================================="
echo "✅ EXPERIMENT COMPLETE"
echo "==================================================================="
echo "Results saved to: $SAVE_DIR/"
echo ""
echo "Key files for paper:"
echo "  📊 $SAVE_DIR/metric_output/comparison.md    ← Table 2 equivalent"
echo "  📈 $SAVE_DIR/eval_sg_merged.jsonl           ← UCOM + correctness"
echo "  🔬 AUROC / AUARC / Optimal tau printed above"
echo "==================================================================="
