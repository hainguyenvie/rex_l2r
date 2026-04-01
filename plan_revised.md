# Kế hoạch Nghiên cứu (Revised): Uncertainty-Aware Selective Prediction cho Visual Grounding

## Tóm tắt

Framework **post-hoc, không cần fine-tuning** bổ sung khả năng **từ chối dự đoán có bảo đảm thống kê** cho CoT Visual Grounding model.

**Claim cốt lõi**:
> Bằng cách chạy K lần suy luận ngẫu nhiên và đo UCOM, ta cải thiện Rejection Score từ **68 → >82** trên HumanRef trong khi duy trì Recall/Precision với **FDR ≤ 5% (Clopper-Pearson guarantee)**.

**Venue mục tiêu**: EMNLP 2025 / ICLR 2026 Workshop on Reliable VLMs

---

## 1. Phạm vi (Thu hẹp so với plan.md gốc)

| Giữ lại | Bỏ ra khỏi claim |
|---------|-----------------|
| ✅ Selective Prediction (Learning to Reject) | ❌ **XAI** — CoT đã có sẵn, không phải contribution mới |
| ✅ Uncertainty Quantification (UCOM) | ❌ Fine-tuning / architecture changes |
| ✅ Statistical FDR Control (Clopper-Pearson) | ❌ Computational optimization |
| ✅ Post-hoc (không sửa model) | |

> **Lý do bỏ XAI**: Rex-Thinker đã có CoT reasoning sẵn rồi. Method của ta không thêm
> khả năng giải thích mới — chỉ thêm uncertainty signal. Claim XAI sẽ bị reviewer phản bác.

---

## 2. Baselines BẮT BUỘC (Thiếu trong plan.md gốc)

| ID | Tên | Uncertainty Signal | Status |
|----|-----|--------------------|--------|
| **B1** | Rex-Thinker-GRPO (gốc) | Logic CoT ("No" output) | ✅ Đã có |
| **B2** | GDINO-Confidence | Max DINO detection score | ❌ Cần implement |
| **B3** | Token-Entropy | Output token log-prob entropy | ❌ Cần implement |
| **Ours** | UCOM-K8 | Discrete UQ từ K=8 samples | ✅ Đã có |

> ⚠️ Nếu UCOM-K8 không tốt hơn cả B2 lẫn B3 → rethink trước khi submit.

---

## 3. Phương pháp

### 3.1 Stochastic CoT Sampling
K=8 inference passes, temperature=0.7 (aligned với GRPO training: 8 rollout samples).

### 3.2 UCOM Score
```
TA   = 1 - (p_top1 - p_top2)    [Margin ambiguity]
IE   = Normalized Entropy(p)     [Entropy dispersion]
CD   = 1 - sum(p_i²)            [HHI complement]
UCOM = 0.4·TA + 0.3·IE + 0.3·CD
```

### 3.3 Calibration: Clopper-Pearson FDR Bound
Tìm τ* từ validation set: maximize power sao cho FDR_upper_bound ≤ 5%.

### 3.4 Selective Prediction Rule
```
UCOM(x) > τ*  → REJECT
UCOM(x) ≤ τ*  → ACCEPT (mode của K samples)
```

---

## 4. Experiments

### 4.1 Main: HumanRef Benchmark (6 subsets)
- **Primary**: Rejection Score (baseline = 68/100, target ≥ 82)
- **Secondary**: Recall@0.5, DensityF1 (không giảm quá 3 điểm)
- So sánh 4 methods: B1, B2, B3, Ours

### 4.2 Calibration: AUARC Curves
- Plot Accuracy vs Rejection Rate cho 4 methods
- Expected: Ours có AUARC cao nhất

### 4.3 Ablation A — Effect of K (số samples)
| K | Expected AUROC |
|---|---------------|
| 1 | ~0.55 |
| 4 | ~0.68 |
| 8 | ~0.76 ← sweet spot |
| 16 | ~0.77 (marginal gain, 2x cost) |

### 4.4 Ablation B — UCOM Components
| Config | Expected |
|--------|---------|
| TA only | Trung bình |
| IE only | Trung bình |
| CD only | Trung bình |
| UCOM (full) | Cao nhất |

### 4.5 Bonus: OOD Generalization via RefCOCOg
- Null-expression / rejection cases trên RefCOCOg
- Kiểm tra τ* calibrated trên HumanRef có generalize không

---

## 5. Expected Results để Publishable

| Metric | Required | Lý do |
|--------|----------|-------|
| AUROC(UCOM, incorrect) | **> 0.75** | UQ signal meaningful |
| AUARC: Ours vs B1/B2/B3 | **Ours best** | Selective prediction superiority |
| Rejection Score | **≥ 82** (+14 vs baseline 68) | Concrete measurable improvement |
| Recall@0.5 drop | **≤ 3 điểm** | Acceptable trade-off |
| Empirical FDR | **≤ 5%** | Clopper-Pearson claim holds in practice |

---

## 6. Files cần thêm code

| File | Status |
|------|--------|
| `evaluation/eval_safeground.py` | ✅ Đã build |
| `evaluation/eval_baselines.py` | ❌ B2 (GDINO-Conf) + B3 (Token-Entropy) |
| `evaluation/ablation_k.py` | ❌ K sweep = 1,2,4,8,16 |
| `evaluation/plot_auarc.py` | ❌ AUARC figure cho paper |
| `run_experiment.sh` | ✅ Đã update (4-step pipeline) |

---

## 7. Sửa đổi so với plan.md gốc

| plan.md gốc | Revised | Impact |
|-------------|---------|--------|
| XAI là claim chính | **Bỏ** | Claim focus hơn, tránh reviewer phản bác |
| RefCOCO/+/g là primary | **HumanRef primary** | Có rejection subset sẵn để đo trực tiếp |
| OODBench là benchmark chính | **Bonus experiment** | Focus vào experiment chính trước |
| **Không có baselines** | **B2 + B3 bắt buộc** | **Mandatory for publication** |
| **Không có ablation** | **Ablation A + B** | **Mandatory for publication** |

---

## 8. Timeline

| Tuần | Công việc | Output |
|------|-----------|--------|
| T1 | Download HumanRef, chạy eval_safeground.py | Preds + UCOM scores |
| T1 | Implement B2, B3 baselines | eval_baselines.py |
| T2 | Ablation A (K sweep) + B (components) | Ablation tables |
| T2 | Phân tích, vẽ AUARC curves | Paper figures |
| T3 | Viết paper | Submit |

---

## Tài liệu tham khảo (Bổ sung)

[12] Angelopoulos et al., "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control," ICML 2021.
[13] Geifman & El-Yaniv, "Selective Prediction for Setup with Expert Deferral," NeurIPS 2019.
