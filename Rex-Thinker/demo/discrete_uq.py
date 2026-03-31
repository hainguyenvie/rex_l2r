import numpy as np
from typing import List

def compute_ta(probs: List[float]) -> float:
    """
    Top-Candidate Ambiguity (TA) - Margin-based uncertainty.
    TA = 1 - (p_top1 - p_top2)
    """
    if len(probs) == 0:
        return 1.0
    if len(probs) == 1:
        return 0.0
    
    sorted_probs = sorted(probs, reverse=True)
    p1 = sorted_probs[0]
    p2 = sorted_probs[1]
    return 1.0 - (p1 - p2)

def compute_ie(probs: List[float], epsilon: float = 1e-8) -> float:
    """
    Informational Dispersion (IE) - Entropy-based uncertainty.
    Normalized to [0, 1] using log(M) where M is number of active candidates.
    If M=1, IE is 0.
    """
    if len(probs) <= 1:
        return 0.0
        
    M = len(probs)
    entropy = -np.sum([p * np.log(p + epsilon) for p in probs if p > 0])
    max_entropy = np.log(M)
    
    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

def compute_cd(probs: List[float]) -> float:
    """
    Concentration Deficit (CD) - HHI Complement.
    CD = 1 - sum(p_i^2)
    """
    if len(probs) == 0:
        return 1.0
        
    hhi = sum([p**2 for p in probs])
    return 1.0 - hhi

def compute_ucom(probs: List[float], w_ta=0.4, w_ie=0.3, w_cd=0.3) -> float:
    """
    Combined Uncertainty (UCOM).
    """
    if len(probs) == 0:
        return 1.0
        
    ta = compute_ta(probs)
    ie = compute_ie(probs)
    cd = compute_cd(probs)
    
    # Normalize weights
    total_w = w_ta + w_ie + w_cd
    w_ta, w_ie, w_cd = w_ta/total_w, w_ie/total_w, w_cd/total_w
    
    ucom = w_ta * ta + w_ie * ie + w_cd * cd
    return min(1.0, max(0.0, float(ucom)))

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    calc_area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = calc_area(box1)
    box2_area = calc_area(box2)
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def cluster_boxes_and_get_probs(boxes_list: List[List[float]], iou_threshold=0.8) -> List[float]:
    """
    Cluster a list of predicted boxes using IoU and return the probability distribution
    over the identified clusters (regions).
    """
    if not boxes_list:
        return []
        
    clusters = [] # List of lists of boxes
    
    for box in boxes_list:
        matched = False
        for cluster in clusters:
            # Check IoU with the first box in the cluster (cluster representative)
            if calculate_iou(box, cluster[0]) >= iou_threshold:
                cluster.append(box)
                matched = True
                break
        if not matched:
            clusters.append([box])
            
    # Calculate probabilities
    total_boxes = len(boxes_list)
    probs = [len(cluster) / total_boxes for cluster in clusters]
    return probs
