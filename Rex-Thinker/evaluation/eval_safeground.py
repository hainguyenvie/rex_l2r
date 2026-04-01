"""
eval_safeground.py
==================
Full evaluation pipeline: Rex-Thinker-GRPO inference + SafeGround UQ on HumanRef benchmark.

Pipeline:
  1. Load HumanRef annotations (same format as eval.py)
  2. For each sample: run K=8 stochastic inference passes
  3. Compute UCOM score from sampled box distributions
  4. Save per-sample predictions + UCOM to two output files:
     - eval_sg_preds.jsonl   → fed into metric.py for Recall/Precision/DF1
     - eval_sg_ucom.jsonl    → fed into calibrate_tau.py for AUROC/AUARC/tau

Usage:
  cd Rex-Thinker
  CUDA_VISIBLE_DEVICES=0 python evaluation/eval_safeground.py \\
    --model_path ../IDEA-Research/Rex-Thinker-GRPO-7B \\
    --anno_path data/IDEA-Research/HumanRef/annotations.jsonl \\
    --image_root_dir data/IDEA-Research/HumanRef/images \\
    --save_dir results/safeground_run1 \\
    --gdino_config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
    --gdino_weights ../GroundingDINO/weights/groundingdino_swint_ogc.pth \\
    --num_samples 8 \\
    --temperature 0.7 \\
    --start_idx 0 \\
    --end_idx 6001
"""

import argparse
import json
import os
import sys
import re
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm

# ── Detect attention backend ───────────────────────────────────────────────────
# flash-attn phải được build từ source cùng PyTorch version.
# Nếu bị ABI mismatch → uninstall và rebuild: pip uninstall flash-attn -y && pip install flash-attn --no-build-isolation
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: ABI test
    _ATTN_IMPL = "flash_attention_2"
    print("[INFO] Using flash_attention_2")
except Exception as _e:
    _ATTN_IMPL = "sdpa"
    print(f"[WARN] flash_attn broken ({type(_e).__name__}), using sdpa.")
    print("[WARN] Fix: pip uninstall flash-attn -y && pip install flash-attn --no-build-isolation")
    # Xóa flash_attn khỏi sys.modules để transformers không cố import nó nữa
    import sys as _sys
    for _k in list(_sys.modules.keys()):
        if "flash_attn" in _k:
            del _sys.modules[_k]
    # Patch transformers để không import flash_attn ở module level
    import importlib, types as _types
    _fake_fa = _types.ModuleType("flash_attn")
    _fake_fa.bert_padding = _types.ModuleType("flash_attn.bert_padding")
    _fake_fa.bert_padding.index_first_axis = None
    _fake_fa.bert_padding.pad_input = None
    _fake_fa.bert_padding.unpad_input = None
    _sys.modules["flash_attn"] = _fake_fa
    _sys.modules["flash_attn.bert_padding"] = _fake_fa.bert_padding
    _sys.modules["flash_attn.flash_attn_interface"] = _types.ModuleType("flash_attn.flash_attn_interface")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info, smart_resize

# GroundingDINO
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

# SafeGround UQ from demo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
from discrete_uq import compute_ucom, compute_ta, compute_ie, compute_cd

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Rex-Thinker + SafeGround UQ Evaluation")
    p.add_argument("--model_path", type=str, default="../IDEA-Research/Rex-Thinker-GRPO-7B")
    p.add_argument("--anno_path", type=str, default="data/IDEA-Research/HumanRef/annotations.jsonl")
    p.add_argument("--image_root_dir", type=str, default="data/IDEA-Research/HumanRef/images")
    p.add_argument("--save_dir", type=str, default="results/safeground_run1")

    # GroundingDINO
    p.add_argument("--gdino_config", type=str,
                   default="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gdino_weights", type=str,
                   default="../GroundingDINO/weights/groundingdino_swint_ogc.pth")
    p.add_argument("--gdino_box_thresh", type=float, default=0.25)
    p.add_argument("--gdino_text_thresh", type=float, default=0.25)

    # UQ params (aligned with GRPO training: 8 rollout samples)
    p.add_argument("--num_samples", type=int, default=8,
                   help="Number of stochastic samples per query (paper uses 8 in GRPO)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--iou_cluster_thresh", type=float, default=0.5,
                   help="IoU threshold to consider two boxes as 'same candidate'")

    # Pixel constraints (same as eval.py)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--max_new_tokens", type=int, default=2048)

    # Dataset slice
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=6001)

    return p.parse_args()


def parse_box_output(text: str):
    """Extract bounding boxes from model output. Returns list of [x0,y0,x1,y1]."""
    pattern = r"\[([0-9\.]+(?:, ?[0-9\.]+)*)\]"
    matches = re.findall(pattern, text)
    boxes = []
    for m in matches:
        nums = [float(x) if '.' in x else int(x) for x in m.split(',')]
        if len(nums) == 4:
            boxes.append(nums)
    return boxes


def convert_boxes_to_qwen25_format(gt_boxes, ori_w, ori_h, min_pixels, max_pixels):
    """Scale abs boxes to Qwen2.5-VL resized input space."""
    rh, rw = smart_resize(ori_h, ori_w, 28, min_pixels=min_pixels, max_pixels=max_pixels)
    out = []
    for x0, y0, x1, y1 in gt_boxes:
        x0 = max(0, min(int(x0 / ori_w * rw), rw - 1))
        y0 = max(0, min(int(y0 / ori_h * rh), rh - 1))
        x1 = max(0, min(int(x1 / ori_w * rw), rw - 1))
        y1 = max(0, min(int(y1 / ori_h * rh), rh - 1))
        out.append([x0, y0, x1, y1])
    return out


def run_gdino(image: Image.Image, category: str, gdino_model, box_thresh=0.25, text_thresh=0.25):
    """Run GroundingDINO on image and return list of abs boxes [[x0,y0,x1,y1],...]."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    boxes, logits, phrases = predict(
        model=gdino_model,
        image=image_transformed,
        caption=category,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
    )
    w, h = image.size
    abs_boxes = []
    for box in boxes:
        cx, cy, bw, bh = box
        x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        abs_boxes.append([float(x0), float(y0), float(x1), float(y1)])
    return abs_boxes


def calculate_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def boxes_to_prob_dist(sampled_boxes_list, candidate_boxes, iou_thresh=0.5):
    """
    Convert K sampled predicted boxes to a probability distribution over candidate box indices.
    Each sample's predicted box is matched to the closest candidate by IoU.
    Returns: list of probabilities (one per unique matched candidate + -1 for no-match)
    """
    matched_indices = []
    for pred_box in sampled_boxes_list:
        if pred_box is None:
            matched_indices.append(-1)  # rejection/no-box
            continue
        best_iou = 0.0
        best_idx = -1
        for ci, cbox in enumerate(candidate_boxes):
            iou = calculate_iou(pred_box, cbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = ci
        matched_indices.append(best_idx if best_iou >= iou_thresh else -1)

    counts = Counter(matched_indices)
    total = len(sampled_boxes_list)
    probs = [cnt / total for cnt in counts.values()]
    return probs, counts


def run_single_inference(model, processor, image, prompt, min_pixels, max_pixels, max_new_tokens,
                         temperature, do_sample=True):
    """Run one stochastic forward pass, return (pred_boxes_in_input_space, input_h, input_w)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")

    input_h = inputs["image_grid_thw"][0][1] * 14
    input_w = inputs["image_grid_thw"][0][2] * 14

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
    boxes_in_input_space = parse_box_output(output_text)
    return boxes_in_input_space, input_h.item(), input_w.item()


def scale_boxes_to_orig(boxes_input_space, input_h, input_w, orig_w, orig_h):
    """Convert boxes from Qwen input space back to original image coordinates."""
    out = []
    for b in boxes_input_space:
        x0 = b[0] / input_w * orig_w
        y0 = b[1] / input_h * orig_h
        x1 = b[2] / input_w * orig_w
        y1 = b[3] / input_h * orig_h
        out.append([x0, y0, x1, y1])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    pred_path = os.path.join(args.save_dir, "eval_sg_preds.jsonl")
    ucom_path = os.path.join(args.save_dir, "eval_sg_ucom.jsonl")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading GroundingDINO...")
    gdino_model = load_model(args.gdino_config, args.gdino_weights)
    gdino_model.eval()

    print(f"Loading Rex-Thinker-GRPO model (attn={_ATTN_IMPL})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation=_ATTN_IMPL, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model.eval()

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"Loading annotations from {args.anno_path}...")
    with open(args.anno_path, 'r') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    lines = lines[args.start_idx:args.end_idx]
    print(f"Processing {len(lines)} samples (idx {args.start_idx}:{args.end_idx})")

    # Determine category from first sample (HumanRef is person-centric)
    # For RefCOCOg, category comes from the annotation itself.
    # We support both via "cate_name" or "category" field in annotation.

    f_pred = open(pred_path, 'a')
    f_ucom = open(ucom_path, 'a')

    for line in tqdm(lines, desc="Eval+UQ"):
        sample_id = line["id"]
        image_path = os.path.join(args.image_root_dir, line["image_name"])
        referring = line["referring"]
        candidate_boxes_gt = line.get("candidate_boxes", [])  # abs coords from annotation
        domain = line.get("domain", "unknown")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open image {image_path}: {e}")
            continue

        w, h = image.size

        # Determine object category for GDINO
        # HumanRef is always "person"; RefCOCOg stores category differently
        category = line.get("cate_name", line.get("category", "person"))

        # ── Run GroundingDINO ─────────────────────────────────────────────────
        try:
            gdino_boxes = run_gdino(image, category, gdino_model,
                                    args.gdino_box_thresh, args.gdino_text_thresh)
        except Exception as e:
            print(f"[WARN] GDINO failed for id={sample_id}: {e}")
            gdino_boxes = []

        # Fall back to annotation candidate_boxes if DINO finds nothing
        if len(gdino_boxes) == 0:
            gdino_boxes = candidate_boxes_gt

        if len(gdino_boxes) == 0:
            # Cannot do anything, save empty prediction
            pred_record = {"id": sample_id, "extracted_predictions": [], "domain": domain}
            ucom_record = {"id": sample_id, "ucom": 1.0, "domain": domain,
                           "is_rejected": True, "ta": 1.0, "ie": 1.0, "cd": 1.0}
            f_pred.write(json.dumps(pred_record) + "\n")
            f_ucom.write(json.dumps(ucom_record) + "\n")
            continue

        # Convert GDINO boxes to Qwen2.5-VL input space for the prompt hint
        hint_boxes = convert_boxes_to_qwen25_format(
            gdino_boxes, w, h, args.min_pixels, args.max_pixels
        )
        hint = json.dumps({category: hint_boxes})
        prompt = (
            f"Hint: Object and its coordinates in this image: {hint}\n"
            f"Please detect {referring} in the image."
        )

        # ── K stochastic forward passes ───────────────────────────────────────
        sampled_first_boxes = []  # one predicted box per sample pass
        for k in range(args.num_samples):
            try:
                boxes_input, in_h, in_w = run_single_inference(
                    model, processor, image, prompt,
                    args.min_pixels, args.max_pixels, args.max_new_tokens,
                    args.temperature
                )
                if boxes_input:
                    # Take the first predicted box (most confident)
                    b_orig = scale_boxes_to_orig([boxes_input[0]], in_h, in_w, w, h)[0]
                    sampled_first_boxes.append(b_orig)
                else:
                    sampled_first_boxes.append(None)  # rejection signal
            except Exception as e:
                print(f"[WARN] Sample k={k} failed for id={sample_id}: {e}")
                sampled_first_boxes.append(None)

        # ── Compute UQ metrics ────────────────────────────────────────────────
        probs, idx_counts = boxes_to_prob_dist(sampled_first_boxes, gdino_boxes,
                                               args.iou_cluster_thresh)
        ta = compute_ta(probs)
        ie = compute_ie(probs)
        cd = compute_cd(probs)
        ucom = compute_ucom(probs, w_ta=0.4, w_ie=0.3, w_cd=0.3)

        # ── Determine final prediction (mode = most common candidate) ─────────
        # Exclude -1 (rejection index) from mode selection
        valid_counts = {k: v for k, v in idx_counts.items() if k != -1}
        if valid_counts:
            best_candidate_idx = max(valid_counts, key=valid_counts.get)
            final_box_orig = gdino_boxes[best_candidate_idx]
            final_boxes = [final_box_orig]
        else:
            # All samples rejected
            final_boxes = []

        # ── Write outputs ─────────────────────────────────────────────────────
        pred_record = {
            "id": sample_id,
            "extracted_predictions": final_boxes,
            "domain": domain,
            "ucom": round(ucom, 6),
        }

        ucom_record = {
            "id": sample_id,
            "ucom": round(ucom, 6),
            "ta": round(ta, 6),
            "ie": round(ie, 6),
            "cd": round(cd, 6),
            "domain": domain,
            "num_candidates": len(gdino_boxes),
            "k_rejected": sum(1 for b in sampled_first_boxes if b is None),
        }

        f_pred.write(json.dumps(pred_record) + "\n")
        f_ucom.write(json.dumps(ucom_record) + "\n")

    f_pred.close()
    f_ucom.close()

    print(f"\n✅ Done!")
    print(f"   Predictions  → {pred_path}")
    print(f"   UCOM scores  → {ucom_path}")
    print(f"\nNext steps:")
    print(f"  1. Compute Recall/Precision/DF1:")
    print(f"     python evaluation/metric.py \\")
    print(f"       --gt_path {args.anno_path} \\")
    print(f"       --pred_path {pred_path} \\")
    print(f"       --pred_names Rex-Thinker-GRPO-SafeGround")
    print(f"\n  2. Calibrate tau (after adding 'correct' labels to UCOM file):")
    print(f"     python evaluation/calibrate_tau.py \\")
    print(f"       --results_file <merged_ucom_with_labels.jsonl> \\")
    print(f"       --target_fdr 0.05 --alpha_conf 0.1")


if __name__ == "__main__":
    main()
