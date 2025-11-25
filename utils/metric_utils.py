"""
Utility functions for computing calibration and classification metrics.

This module provides reusable metric computation functions for both
DeBERTa and LLM models, including ECE, Brier score, and calibration data.
"""

import numpy as np
from typing import Dict, Any, Tuple


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels (integers) of shape [N]
        y_prob: Predicted probabilities [N, C] for multi-class or [N] for binary
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (float)
    """
    # Handle both binary and multi-class cases
    if y_prob.ndim == 1:
        # Binary case: y_prob is already confidence scores
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
    else:
        # Multi-class case: extract max probability and predicted class
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Score (mean squared error between probabilities and one-hot labels).
    
    Args:
        y_true: True labels (integers) of shape [N]
        y_prob: Predicted probabilities [N, C] for multi-class or [N] for binary
        
    Returns:
        Brier score (float)
    """
    # Handle both binary and multi-class cases
    if y_prob.ndim == 1:
        # Binary case: convert to 2D
        y_prob_2d = np.stack([1 - y_prob, y_prob], axis=1)
    else:
        y_prob_2d = y_prob
    
    # Convert to one-hot
    n_classes = y_prob_2d.shape[1]
    y_true_onehot = np.zeros_like(y_prob_2d)
    
    # Handle out-of-bounds labels gracefully
    valid_mask = (y_true >= 0) & (y_true < n_classes)
    y_true_onehot[valid_mask, y_true[valid_mask]] = 1.0
    
    # Compute mean squared error
    return float(np.mean(np.sum((y_prob_2d - y_true_onehot) ** 2, axis=1)))


def compute_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, 
                             n_bins: int = 15) -> Dict[str, Any]:
    """
    Compute calibration bin statistics for plotting and analysis.
    
    Args:
        y_true: True labels (integers) of shape [N]
        y_prob: Predicted probabilities [N, C] for multi-class or [N] for binary
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with bin statistics:
            - bin_centers: Center of each bin
            - bin_accuracies: Accuracy in each bin
            - bin_confidences: Average confidence in each bin
            - bin_counts: Number of samples in each bin
    """
    # Handle both binary and multi-class cases
    if y_prob.ndim == 1:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
    else:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = in_bin.sum()
        
        if count > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(int(count))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)
    
    return {
        'bin_centers': bin_centers.tolist(),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'n_bins': n_bins
    }


def construct_llm_probabilities(llm_labels: np.ndarray, 
                                llm_confidences: np.ndarray,
                                n_classes: int) -> np.ndarray:
    """
    Construct approximate probability distribution from LLM labels and confidences.
    
    The LLM outputs a discrete label and a confidence score (e.g., 0.9 for "high").
    We approximate the full probability distribution by:
    - P(predicted_class) = confidence
    - P(other_classes) = (1 - confidence) / (n_classes - 1)
    
    Args:
        llm_labels: Predicted class labels [N]
        llm_confidences: Confidence scores [N] in range [0, 1]
        n_classes: Total number of classes
        
    Returns:
        Probability distribution [N, n_classes]
    """
    n_samples = len(llm_labels)
    y_probs = np.zeros((n_samples, n_classes))
    
    for i, (pred_label, conf) in enumerate(zip(llm_labels, llm_confidences)):
        if 0 <= pred_label < n_classes:
            # Assign confidence to predicted class
            y_probs[i, pred_label] = conf
            
            # Distribute remaining probability to other classes
            if n_classes > 1:
                remaining_prob = 1.0 - conf
                prob_per_other = remaining_prob / (n_classes - 1)
                for c in range(n_classes):
                    if c != pred_label:
                        y_probs[i, c] = prob_per_other
            else:
                # Edge case: single class (shouldn't happen in practice)
                y_probs[i, pred_label] = 1.0
    
    return y_probs


def binarize_for_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                             confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binarize multi-class predictions to correct/incorrect for calibration analysis.
    
    This is useful for FEVER and other multi-class tasks where we want to assess
    whether the model's confidence correlates with correctness (not specific classes).
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        confidences: Confidence scores [N]
        
    Returns:
        Tuple of (binary_labels, confidences):
            - binary_labels: 1 if correct, 0 if incorrect [N]
            - confidences: Original confidence scores [N]
    """
    binary_labels = (y_true == y_pred).astype(int)
    return binary_labels, confidences

