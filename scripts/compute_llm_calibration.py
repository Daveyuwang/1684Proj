"""
Compute calibration metrics (ECE, Brier) for LLM annotations.

This script:
1. Loads existing LLM annotation JSON files
2. Extracts labels and confidence scores
3. Constructs approximate probability distributions
4. Computes ECE and Brier scores
5. Saves results to JSON files for use in dashboard

For FEVER (3-class), we also compute binarized metrics (correct/incorrect).
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics_util import (
    compute_ece, 
    compute_brier_score, 
    compute_calibration_bins,
    construct_llm_probabilities,
    binarize_for_calibration
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dataset configuration
DATASET_CONFIG = {
    'imdb': {
        'path': 'results/llm_annotations_7b/imdb/imdb_llm_annotations.json',
        'n_classes': 2,
        'label_names': ['negative', 'positive']
    },
    'jigsaw': {
        'path': 'results/llm_annotations_7b/jigsaw/jigsaw_llm_annotations.json',
        'n_classes': 2,
        'label_names': ['non-toxic', 'toxic']
    },
    'fever': {
        'path': 'results/llm_annotations_7b/fever/fever_llm_annotations.json',
        'n_classes': 3,
        'label_names': ['refutes', 'NOT ENOUGH INFO', 'supports']
    }
}


def load_llm_annotations(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load LLM annotations from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of annotation dictionaries
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} annotations from {file_path.name}")
    return data


def extract_labels_and_confidences(annotations: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Extract labels and confidences from annotation list.
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        Dictionary with numpy arrays:
            - gold_labels: Ground truth labels
            - llm_labels: LLM predicted labels
            - confidences: Numeric confidence scores
    """
    # Filter out invalid annotations (llm_label == -1)
    valid_annotations = [a for a in annotations if a.get('llm_label', -1) >= 0]
    
    if len(valid_annotations) < len(annotations):
        logger.warning(f"Filtered out {len(annotations) - len(valid_annotations)} invalid annotations")
    
    gold_labels = np.array([a['gold_label'] for a in valid_annotations])
    llm_labels = np.array([a['llm_label'] for a in valid_annotations])
    confidences = np.array([a['llm_confidence_numeric'] for a in valid_annotations])
    
    return {
        'gold_labels': gold_labels,
        'llm_labels': llm_labels,
        'confidences': confidences,
        'n_valid': len(valid_annotations),
        'n_total': len(annotations)
    }


def compute_metrics_for_dataset(dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute calibration metrics for a single dataset.
    
    Args:
        dataset_name: Name of dataset (imdb, jigsaw, fever)
        config: Dataset configuration dictionary
        
    Returns:
        Dictionary with computed metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Load annotations
    file_path = project_root / config['path']
    annotations = load_llm_annotations(file_path)
    
    # Extract data
    data = extract_labels_and_confidences(annotations)
    y_true = data['gold_labels']
    y_pred = data['llm_labels']
    confidences = data['confidences']
    n_classes = config['n_classes']
    
    logger.info(f"Valid samples: {data['n_valid']} / {data['n_total']}")
    
    # 1. Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary' if n_classes == 2 else 'macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    avg_confidence = float(np.mean(confidences))
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Cohen's Kappa: {kappa:.4f}")
    logger.info(f"Avg Confidence: {avg_confidence:.4f}")
    
    # 2. Construct approximate probability distributions
    y_probs = construct_llm_probabilities(y_pred, confidences, n_classes)
    
    # 3. Compute calibration metrics (multi-class)
    ece = compute_ece(y_true, y_probs)
    brier = compute_brier_score(y_true, y_probs)
    
    logger.info(f"ECE (multi-class): {ece:.4f}")
    logger.info(f"Brier Score (multi-class): {brier:.4f}")
    
    # 4. Compute calibration bin statistics (for plotting)
    calibration_bins = compute_calibration_bins(y_true, y_probs, n_bins=15)
    
    # 5. For multi-class datasets (FEVER), also compute binarized metrics
    binarized_metrics = None
    if n_classes > 2:
        logger.info("\nComputing binarized metrics (correct/incorrect)...")
        binary_labels, binary_confidences = binarize_for_calibration(y_true, y_pred, confidences)
        
        # Treat confidence as probability of being correct
        binary_probs = binary_confidences  # Shape: [N]
        
        ece_binary = compute_ece(binary_labels, binary_probs)
        brier_binary = compute_brier_score(binary_labels, binary_probs)
        
        logger.info(f"ECE (binarized): {ece_binary:.4f}")
        logger.info(f"Brier Score (binarized): {brier_binary:.4f}")
        
        calibration_bins_binary = compute_calibration_bins(binary_labels, binary_probs, n_bins=15)
        
        binarized_metrics = {
            'ece': float(ece_binary),
            'brier_score': float(brier_binary),
            'calibration_bins': calibration_bins_binary,
            'description': 'Binarized to correct (1) vs incorrect (0) predictions'
        }
    
    # 6. Compile results
    results = {
        'dataset': dataset_name,
        'n_classes': n_classes,
        'label_names': config['label_names'],
        'n_samples': int(data['n_valid']),
        'n_total': int(data['n_total']),
        'classification_metrics': {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'cohen_kappa': float(kappa),
            'average_confidence': float(avg_confidence)
        },
        'calibration_metrics': {
            'ece': float(ece),
            'brier_score': float(brier),
            'calibration_bins': calibration_bins,
            'description': 'Multi-class calibration metrics'
        }
    }
    
    if binarized_metrics:
        results['calibration_metrics_binarized'] = binarized_metrics
    
    return results


def main():
    """
    Main function to compute calibration metrics for all datasets.
    """
    logger.info("="*60)
    logger.info("LLM CALIBRATION METRICS COMPUTATION")
    logger.info("="*60)
    logger.info("Computing ECE and Brier scores from existing LLM annotations")
    logger.info("No re-inference required - using saved JSON files")
    logger.info("="*60)
    
    # Output directory
    output_dir = project_root / "results" / "llm_annotations_7b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Process each dataset
    for dataset_name, config in DATASET_CONFIG.items():
        try:
            results = compute_metrics_for_dataset(dataset_name, config)
            all_results[dataset_name] = results
            
            # Save individual dataset results
            dataset_output_dir = output_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True)
            
            output_file = dataset_output_dir / f"{dataset_name}_calibration_metrics.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved calibration metrics to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}", exc_info=True)
            continue
    
    # Save combined results
    combined_output = output_dir / "all_datasets_calibration_metrics.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*60}")
    logger.info(f"{'Dataset':<10} | {'N':<6} | {'Acc':<6} | {'F1':<6} | {'Kappa':<6} | {'ECE':<6} | {'Brier':<6}")
    logger.info("-" * 70)
    
    for dataset_name, results in all_results.items():
        n = results['n_samples']
        acc = results['classification_metrics']['accuracy']
        f1 = results['classification_metrics']['f1_score']
        kappa = results['classification_metrics']['cohen_kappa']
        ece = results['calibration_metrics']['ece']
        brier = results['calibration_metrics']['brier_score']
        
        logger.info(f"{dataset_name:<10} | {n:<6} | {acc:.3f}  | {f1:.3f}  | {kappa:.3f}  | {ece:.3f}  | {brier:.3f}")
    
    logger.info(f"{'='*60}")
    logger.info(f"All results saved to: {output_dir}")
    logger.info(f"Combined results: {combined_output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

