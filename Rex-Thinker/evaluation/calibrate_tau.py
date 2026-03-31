import argparse
import json
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, auc
import os

def clopper_pearson_upper_bound(k, n, alpha_conf):
    """
    Calculate the upper Clopper-Pearson confidence bound.
    k: number of false discoveries (incorrect but accepted answers)
    n: total number of accepted answers
    alpha_conf: probability of bound failure (e.g., 0.1 for 90% confidence)
    """
    if n == 0:
        return 1.0
    return beta.ppf(1 - alpha_conf, k + 1, n - k)

def analyze_calibration(results_jsonl, target_fdr=0.05, alpha_conf=0.1):
    """
    Given a list of results containing {"ucom": float, "correct": bool},
    find the optimal threshold tau that guarantees FDR <= target_fdr
    with 1 - alpha_conf confidence.
    """
    results = []
    with open(results_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    uncertainties = np.array([r['ucom'] for r in results])
    correctness = np.array([r['correct'] for r in results])
    
    # 1. Compute Base Metrics (AUROC, AUARC)
    # Note: AUROC inputs are (true labels, prediction scores).
    # Since higher uncertainty should mean incorrect (label=0), 
    # we predict 'incorrect' probability as the uncertainty score.
    incorrect_labels = 1.0 - correctness
    try:
        auroc = roc_auc_score(incorrect_labels, uncertainties)
    except ValueError:
        auroc = 0.0 # Only one class present
        
    print(f"--- Meta Metrics ---")
    print(f"AUROC (Uncertainty vs Incorrectness): {auroc:.4f}")
    
    # AUARC
    sorted_idx = np.argsort(uncertainties)
    sorted_corr = correctness[sorted_idx]
    n = len(sorted_corr)
    
    accuracies = []
    rejection_rates = []
    for i in range(1, n + 1):
        acc = np.mean(sorted_corr[:i])
        rejection_rate = 1.0 - (i / n)
        accuracies.append(acc)
        rejection_rates.append(rejection_rate)
        
    auarc = auc(rejection_rates[::-1], accuracies[::-1])
    print(f"AUARC (Area Under Accuracy-Rejection Curve): {auarc:.4f}")
    
    # 2. Threshold Calibration via LTT (Learn Then Test) / Clopper-Pearson
    print(f"\n--- Calibration for target FDR={target_fdr:.2%} (Conf={1-alpha_conf:.0%}) ---")
    
    unique_taus = np.unique(uncertainties)
    unique_taus = np.sort(unique_taus)[::-1] # Test from highest tau to lowest (least conservative to most)
    
    optimal_tau = None
    best_power = 0.0
    
    for tau in unique_taus:
        accepted_mask = uncertainties <= tau
        n_accepted = np.sum(accepted_mask)
        if n_accepted == 0:
            continue
            
        n_incorrect_accepted = np.sum((~correctness[accepted_mask]))
        
        # Clopper pearson upper bound on FDR
        fdr_bound = clopper_pearson_upper_bound(n_incorrect_accepted, n_accepted, alpha_conf)
        
        if fdr_bound <= target_fdr:
            power = n_accepted / n # Fraction of data kept
            if power > best_power:
                best_power = power
                optimal_tau = tau
                
    if optimal_tau is not None:
        accepted_mask = uncertainties <= optimal_tau
        final_fdr = np.mean(~correctness[accepted_mask])
        print(f"✅ Found Optimal Threshold Tau: {optimal_tau:.4f}")
        print(f"   -> Resulting Empirical FDR: {final_fdr:.2%}")
        print(f"   -> Resulting Power (Accepted): {best_power:.2%}")
    else:
        print("❌ Could not find a valid threshold tau to satisfy the FDR requirements.")
        
    return optimal_tau

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, 
                        help="JSONL file containing validation results: {'ucom': 0.1, 'correct': true}")
    parser.add_argument("--target_fdr", type=float, default=0.05, 
                        help="Target False Discovery Rate (e.g. 0.05 for 5%)")
    parser.add_argument("--alpha_conf", type=float, default=0.1, 
                        help="Alpha confidence level for Clopper-Pearson (e.g. 0.1 for 90%)")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print("File not found! Please create a dummy JSONL file or provide validation results.")
    else:
        analyze_calibration(args.results_file, args.target_fdr, args.alpha_conf)
