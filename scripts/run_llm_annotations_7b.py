"""
Run LLM annotations with Qwen2.5-7B-Instruct model.

Model: Qwen/Qwen2.5-7B-Instruct
Reference: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

This script implements:
- Batch LLM annotation for all three datasets
- Agreement metrics computation
- Hard case identification
- CLI control for dataset selection
- Resume from checkpoint support

Usage:
  python scripts/run_llm_annotations_7b.py --datasets imdb
  python scripts/run_llm_annotations_7b.py --datasets imdb jigsaw
  python scripts/run_llm_annotations_7b.py --datasets all
  python scripts/run_llm_annotations_7b.py --datasets fever --resume
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, classification_report, confusion_matrix
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from data.dataset_loader import load_dataset_by_name
from llm.prompts import get_prompt_template, get_verbal_confidence_mapping, TaskType
from llm.annotator import QwenAnnotator
from utils.text_utils import compute_text_hash, get_text_preview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM annotations with Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=["imdb", "jigsaw", "fever", "all"],
        default=["all"], 
        help="Run one or more datasets. Use 'all' for all three."
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from latest checkpoint if exists."
    )
    parser.add_argument(
        "--checkpoint-interval", 
        type=int, 
        default=100,
        help="Save checkpoint every N samples (default: 100)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for inference (default: from config)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: process only 50 samples per dataset"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=50,
        help="Number of samples in test mode (default: 50)"
    )
    parser.add_argument(
        "--rationale",
        type=str,
        choices=["none", "selected", "all"],
        default="selected",
        help="Rationale generation strategy: none (no rationale), selected (on-demand), all (always)"
    )
    return parser.parse_args()


# Dataset to TaskType mapping
DATASET_TASK_MAP = {
    'imdb': TaskType.SENTIMENT,
    'jigsaw': TaskType.TOXICITY,
    'fever': TaskType.FACT_VERIFICATION
}

# Label mapping for each dataset
LABEL_MAPPINGS = {
    'imdb': {0: 'negative', 1: 'positive'},
    'jigsaw': {0: 'non-toxic', 1: 'toxic'},
    'fever': {0: 'refutes', 1: 'NOT ENOUGH INFO', 2: 'supports'}  # Note: FEVER uses 3 classes
}

# Reverse mappings (LLM outputs lowercase not_enough_info, we map to uppercase for consistency)
REVERSE_LABEL_MAPPINGS = {
    'imdb': {'negative': 0, 'positive': 1},
    'jigsaw': {'non-toxic': 0, 'toxic': 1},
    'fever': {'refutes': 0, 'not_enough_info': 1, 'supports': 2}  # LLM outputs not_enough_info
}

# For FEVER: convert LLM label to display format
def normalize_fever_label_for_display(label_str: str) -> str:
    """Convert FEVER label from LLM format to display format."""
    if label_str == 'not_enough_info':
        return 'NOT ENOUGH INFO'
    return label_str


def load_checkpoint(output_dir: Path, dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint if exists.
    
    Args:
        output_dir: Output directory
        dataset_name: Name of dataset
        
    Returns:
        Checkpoint data dict or None if not exists
    """
    # Look for checkpoints in checkpoints subdirectory
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_pattern = f"{dataset_name}_checkpoint_*.json"
    checkpoints = list(checkpoint_dir.glob(checkpoint_pattern))
    
    if not checkpoints:
        return None
    
    # Find latest checkpoint by number
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    logger.info(f"Found checkpoint: {latest_checkpoint.name}")
    
    try:
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_completed = len(data)
        logger.info(f"Loaded checkpoint: {num_completed} samples already processed")
        
        return {
            'results': data,
            'start_idx': num_completed,
            'checkpoint_file': latest_checkpoint
        }
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def annotate_dataset(dataset_name: str, test_df: pd.DataFrame, output_dir: Path, 
                     annotator: QwenAnnotator, batch_size: int = 8, checkpoint_interval: int = 100,
                     resume: bool = False) -> Dict[str, Any]:
    """
    Annotate a dataset using Qwen2.5-7B-Instruct model with batch inference.
    
    Args:
        dataset_name: Name of dataset
        test_df: Test DataFrame with text and labels
        output_dir: Directory to save results
        annotator: Initialized QwenAnnotator instance
        batch_size: Batch size for parallel inference (7B can handle larger batches)
        checkpoint_interval: Save checkpoint every N samples
        resume: Resume from checkpoint if exists
        
    Returns:
        Dictionary with annotation results and metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {dataset_name.upper()} with Qwen2.5-7B-Instruct")
    logger.info(f"{'='*60}")
    
    # Create dataset subdirectory for results
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(exist_ok=True)
    
    task_type = DATASET_TASK_MAP[dataset_name]
    confidence_mapping = get_verbal_confidence_mapping()
    
    # Extract texts and indices for batch processing
    texts = test_df['text'].tolist()
    gold_labels = test_df['label'].tolist()
    # Use DataFrame index as example_id (for Jigsaw, this is original_index)
    example_ids = test_df.index.tolist()
    
    # Check for checkpoint if resume mode
    start_idx = 0
    results = []
    failed_annotations = []
    
    if resume:
        checkpoint_data = load_checkpoint(output_dir, dataset_name)
        if checkpoint_data:
            results = checkpoint_data['results']
            start_idx = checkpoint_data['start_idx']
            logger.info(f"Resuming from sample {start_idx}")
        else:
            logger.info("No checkpoint found, starting from beginning")
    
    if start_idx >= len(texts):
        logger.info(f"All {len(texts)} samples already processed!")
        return {
            'results': results,
            'failed': failed_annotations,
            'validation_pass_rate': len(results) / len(texts) if len(texts) > 0 else 0
        }
    
    # Process in chunks to avoid memory accumulation
    logger.info(f"Running LLM batch annotations on {len(texts) - start_idx} samples (batch_size={batch_size})...")
    
    # Get annotation config
    save_full_text = config.ANNOTATION_CONFIG.get('save_full_text', False)
    
    # Process in chunks of 500 samples to manage memory
    chunk_size = 500
    for chunk_start in range(start_idx, len(texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        logger.info(f"Processing chunk {chunk_start}-{chunk_end}...")
        
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_labels = gold_labels[chunk_start:chunk_end]
        chunk_ids = example_ids[chunk_start:chunk_end]
        
        # Annotate this chunk
        annotation_results = annotator.annotate_batch(chunk_texts, batch_size=batch_size)
        
        # Process results immediately (don't accumulate)
        for rel_idx, (annotation_result, text, gold_label, example_id) in enumerate(zip(annotation_results, chunk_texts, chunk_labels, chunk_ids)):
            # Use actual DataFrame index (original_index for Jigsaw) as example_id
            # Keep row_id for backward compatibility (position in test_df)
            abs_idx = chunk_start + rel_idx
            
            if annotation_result.is_valid:
                # Convert label to numeric
                llm_label_str = annotation_result.label
                llm_label_numeric = REVERSE_LABEL_MAPPINGS[dataset_name].get(llm_label_str, -1)
                
                # Convert confidence to numeric
                llm_confidence_numeric = confidence_mapping[annotation_result.confidence]
                
                # For FEVER, convert display label
                llm_label_display = normalize_fever_label_for_display(llm_label_str) if dataset_name == 'fever' else llm_label_str
                
                # Build result dict with optimized storage
                result = {
                    'example_id': int(example_id),  # Real index (original_index for Jigsaw)
                    'row_id': abs_idx,  # Position in test_df (for backward compatibility)
                    'text_hash': compute_text_hash(text),
                    'text_len': len(text),
                    'gold_label': int(gold_label),
                    'gold_label_str': LABEL_MAPPINGS[dataset_name][gold_label],
                    'llm_label': llm_label_numeric,
                    'llm_label_str': llm_label_display,
                    'llm_confidence': annotation_result.confidence,
                    'llm_confidence_numeric': llm_confidence_numeric,
                    'llm_rationale': annotation_result.rationale,
                    'raw_response': annotation_result.raw_response,
                    'is_valid': True,
                    'error': None
                }
                
                # Only save full text if config says so (default: False to save storage)
                if save_full_text:
                    result['text'] = text
                
                results.append(result)
            else:
                failed_annotations.append({
                    'example_id': int(example_id),  # Real index
                    'row_id': abs_idx,  # Position in test_df
                    'text_hash': compute_text_hash(text),
                    'text_len': len(text),
                    'error': annotation_result.error,
                    'raw_output': annotation_result.raw_response
                })
                logger.warning(f"Failed annotation for example_id {example_id} (row {abs_idx}): {annotation_result.error}")
            
            # Save checkpoint to checkpoints subdirectory
            current_completed = len(results)
            if current_completed % checkpoint_interval == 0:
                checkpoint_dir = output_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_file = checkpoint_dir / f"{dataset_name}_checkpoint_{current_completed}.json"
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved: {current_completed} annotations completed")
        
        # Clean up chunk data
        del annotation_results, chunk_texts, chunk_labels
        import gc
        gc.collect()
    
    # Calculate validation pass rate
    validation_pass_rate = len(results) / len(test_df) if len(test_df) > 0 else 0
    logger.info(f"Validation pass rate: {validation_pass_rate:.2%} ({len(results)}/{len(test_df)})")
    
    # Save annotations to dataset subdirectory
    annotations_file = dataset_output_dir / f"{dataset_name}_llm_annotations.json"
    logger.info(f"Saving annotations to {annotations_file}...")
    
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save failed annotations to dataset subdirectory
    if failed_annotations:
        failed_file = dataset_output_dir / f"{dataset_name}_failed_annotations.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_annotations, f, indent=2, ensure_ascii=False)
        logger.warning(f"Saved {len(failed_annotations)} failed annotations to {failed_file}")
    
    return {
        'results': results,
        'failed': failed_annotations,
        'validation_pass_rate': validation_pass_rate
    }


def compute_agreement_metrics(results: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """
    Compute agreement metrics between LLM and gold labels with safety checks.
    
    Args:
        results: List of annotation results
        dataset_name: Name of dataset
        
    Returns:
        Dictionary of agreement metrics
    """
    logger.info(f"\nComputing agreement metrics for {dataset_name}...")
    
    # Extract labels
    gold_labels = [r['gold_label'] for r in results]
    llm_labels = [r['llm_label'] for r in results]
    llm_confidences = [r['llm_confidence_numeric'] for r in results]
    
    # Filter out invalid labels
    valid_indices = [i for i, lbl in enumerate(llm_labels) if lbl >= 0]
    gold_labels_valid = [gold_labels[i] for i in valid_indices]
    llm_labels_valid = [llm_labels[i] for i in valid_indices]
    llm_confidences_valid = [llm_confidences[i] for i in valid_indices]
    
    # Handle empty case
    if len(valid_indices) == 0:
        logger.warning(f"No valid annotations for {dataset_name}")
        return {
            'dataset': dataset_name,
            'num_samples': len(results),
            'num_valid': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'cohen_kappa': 0.0,
            'agreement_rate': 0.0,
            'average_confidence': 0.0,
            'classification_report': {},
            'confusion_matrix': []
        }
    
    # Basic metrics
    accuracy = accuracy_score(gold_labels_valid, llm_labels_valid)
    
    # Handle multi-class vs binary
    num_classes = len(set(gold_labels_valid))
    average = 'binary' if num_classes == 2 else 'macro'
    
    precision = precision_score(gold_labels_valid, llm_labels_valid, average=average, zero_division=0)
    recall = recall_score(gold_labels_valid, llm_labels_valid, average=average, zero_division=0)
    f1 = f1_score(gold_labels_valid, llm_labels_valid, average=average, zero_division=0)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(gold_labels_valid, llm_labels_valid)
    
    # Agreement rate
    agreement_rate = np.mean([g == l for g, l in zip(gold_labels_valid, llm_labels_valid)])
    
    # Classification report
    class_names = list(LABEL_MAPPINGS[dataset_name].values())
    report = classification_report(gold_labels_valid, llm_labels_valid, 
                                  target_names=class_names, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(gold_labels_valid, llm_labels_valid)
    
    metrics = {
        'dataset': dataset_name,
        'num_samples': len(results),
        'num_valid': len(valid_indices),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'cohen_kappa': float(kappa),
        'agreement_rate': float(agreement_rate),
        'average_confidence': float(np.mean(llm_confidences_valid)),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info("\n" + "="*60)
    logger.info("AGREEMENT METRICS SUMMARY")
    logger.info("="*60)
    logger.info(f"Accuracy:        {accuracy:.4f}")
    logger.info(f"Precision:       {precision:.4f}")
    logger.info(f"Recall:          {recall:.4f}")
    logger.info(f"F1-Score:        {f1:.4f}")
    logger.info(f"Cohen's Kappa:   {kappa:.4f}")
    logger.info(f"Agreement Rate:  {agreement_rate:.4f}")
    logger.info(f"Avg Confidence:  {np.mean(llm_confidences_valid):.4f}")
    logger.info("="*60)
    
    return metrics


def identify_hard_cases(results: List[Dict[str, Any]], dataset_name: str, 
                       texts: List[str], confidence_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Identify hard cases based on disagreement and low confidence.
    
    Args:
        results: List of annotation results
        dataset_name: Name of dataset
        texts: Original text list (for text_preview)
        confidence_threshold: Threshold for considering low confidence
        
    Returns:
        List of hard case examples (with text_preview for manual review)
    """
    logger.info(f"\nIdentifying hard cases for {dataset_name}...")
    
    # Get config for text preview
    save_preview = config.ANNOTATION_CONFIG.get('save_text_preview_for_hard_cases', True)
    preview_length = config.ANNOTATION_CONFIG.get('text_preview_length', 100)
    
    hard_cases = []
    
    for result in results:
        gold_label = result['gold_label']
        llm_label = result['llm_label']
        confidence = result['llm_confidence_numeric']
        
        # Hard case criteria
        is_disagreement = (gold_label != llm_label) and (llm_label >= 0)
        is_low_confidence = confidence < confidence_threshold
        
        if is_disagreement or is_low_confidence:
            hard_case = result.copy()
            hard_case['reason'] = []
            if is_disagreement:
                hard_case['reason'].append('disagreement')
            if is_low_confidence:
                hard_case['reason'].append('low_confidence')
            
            # Add text_preview for manual review
            if save_preview:
                row_id = result['row_id']
                if row_id < len(texts):
                    hard_case['text_preview'] = get_text_preview(texts[row_id], preview_length)
            
            hard_cases.append(hard_case)
    
    logger.info(f"Identified {len(hard_cases)} hard cases ({len(hard_cases)/len(results):.2%})")
    
    return hard_cases


def main():
    """
    Main function to run LLM annotations with Qwen2.5-7B-Instruct model.
    
    This version uses the 7B model which is optimized for:
    - Faster download and inference
    - Lower memory requirements
    - Quick results for midterm report
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = project_root / "results" / "llm_annotations_7b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    logger.info("Using Qwen2.5-7B-Instruct Model")
    logger.info("Model Reference: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct")
    
    # Determine which datasets to process
    if "all" in args.datasets:
        datasets = ['imdb', 'jigsaw', 'fever']
    else:
        datasets = args.datasets
    
    logger.info(f"Datasets to process: {', '.join(datasets)}")
    
    if args.test_mode:
        logger.info(f"TEST MODE: Processing {args.test_samples} samples per dataset")
    else:
        logger.info("FULL MODE: Processing all samples")
    
    if args.resume:
        logger.info("RESUME MODE: Will load from checkpoints if available")
    
    logger.info("="*60)
    
    # LLM configuration for 7B model
    llm_config = config.LLM_CONFIG_7B
    model_name = config.LLM_MODELS.get(llm_config.get("model_name", "qwen-7b"), "Qwen/Qwen2.5-7B-Instruct")
    do_sample = llm_config.get("do_sample", True)
    # Only get temperature if do_sample is True
    temperature = llm_config.get("temperature", 0.3) if do_sample else None
    seed = llm_config.get("seed", 42)
    batch_size = args.batch_size if args.batch_size else config.LLM_BATCH_SIZE_7B
    
    # Store config metadata
    config_metadata = {
        "model_name": model_name,
        "model_size": "7B",
        "model_type": "Qwen2.5-7B-Instruct",
        "seed": seed,
        "batch_size": batch_size,
        "do_sample": do_sample,
        "max_tokens": llm_config.get("max_tokens", 128),
        "repetition_penalty": llm_config.get("repetition_penalty", 1.05),
        "test_mode": args.test_mode,
        "test_samples": args.test_samples if args.test_mode else "full",
        "datasets": datasets,
        "resume": args.resume,
        "checkpoint_interval": args.checkpoint_interval,
        "annotation_storage": {
            "save_full_text": config.ANNOTATION_CONFIG.get("save_full_text", False),
            "text_preview_length": config.ANNOTATION_CONFIG.get("text_preview_length", 100),
            "save_text_preview_for_hard_cases": config.ANNOTATION_CONFIG.get("save_text_preview_for_hard_cases", True),
            "note": "Full text not saved by default (use row_id to retrieve from dataset). Hard cases include text_preview for manual review."
        }
    }
    
    # Add sampling params to metadata only if do_sample is True
    if do_sample:
        config_metadata["temperature"] = temperature
        config_metadata["top_p"] = llm_config.get("top_p", 0.8)
        config_metadata["top_k"] = llm_config.get("top_k", 20)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Do sample: {do_sample}")
    if do_sample:
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Top-p: {config_metadata['top_p']}, Top-k: {config_metadata['top_k']}")
    else:
        logger.info("Greedy decoding (no sampling)")
    logger.info(f"Seed: {seed} (for reproducibility)")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval}")
    
    # Save config metadata
    config_file = output_dir / "config_metadata.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Config metadata saved to {config_file}")
    
    # IMPORTANT: Load model once, reuse for all datasets (avoid reloading)
    logger.info("\n" + "="*60)
    logger.info("LOADING MODEL (once for all datasets)")
    logger.info("="*60)
    # Only pass temperature if do_sample is True
    if do_sample:
        annotator = QwenAnnotator(model_name=model_name, temperature=temperature, seed=seed)
    else:
        annotator = QwenAnnotator(model_name=model_name, seed=seed)
    annotator.load_model()
    logger.info("Model loaded successfully, will be reused for all datasets")
    
    # Process each dataset
    all_metrics = {}
    
    for dataset_name in datasets:
        try:
            # Switch to correct task type for this dataset
            task_type = DATASET_TASK_MAP[dataset_name]
            annotator.set_task_type(task_type)
            
            # Load dataset
            logger.info(f"\nLoading {dataset_name} dataset...")
            data_splits = load_dataset_by_name(dataset_name)
            test_df = data_splits['test']
            
            # Limit samples in test mode
            if args.test_mode:
                test_df = test_df.head(args.test_samples)
                logger.info(f"Test mode: Using {len(test_df)} samples")
            else:
                logger.info(f"Full mode: Processing {len(test_df)} samples")
            
            # Run annotations (reuse loaded model)
            annotation_data = annotate_dataset(
                dataset_name=dataset_name, 
                test_df=test_df, 
                output_dir=output_dir, 
                annotator=annotator, 
                batch_size=batch_size,
                checkpoint_interval=args.checkpoint_interval,
                resume=args.resume
            )
            
            # Compute metrics
            metrics = compute_agreement_metrics(annotation_data['results'], dataset_name)
            
            # Add config metadata to metrics
            metrics['config'] = config_metadata
            
            # Save metrics to dataset subdirectory
            dataset_output_dir = output_dir / dataset_name
            metrics_file = dataset_output_dir / f"{dataset_name}_agreement_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metrics to {metrics_file}")
            
            # Identify and save hard cases (with text_preview for manual review)
            texts = test_df['text'].tolist()  # Get original texts for preview
            hard_cases = identify_hard_cases(annotation_data['results'], dataset_name, texts)
            if hard_cases:
                hard_cases_file = dataset_output_dir / f"{dataset_name}_hard_cases.json"
                with open(hard_cases_file, 'w', encoding='utf-8') as f:
                    json.dump(hard_cases, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(hard_cases)} hard cases to {hard_cases_file}")
            
            all_metrics[dataset_name] = metrics
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}", exc_info=True)
            continue
    
    # Save combined metrics (only if multiple datasets were processed)
    if len(all_metrics) > 1:
        combined_metrics_file = output_dir / "all_datasets_metrics.json"
        with open(combined_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Combined metrics saved to: {combined_metrics_file}")
    else:
        logger.info("Single dataset processed - no combined metrics file created")
    
    logger.info("\n" + "="*60)
    logger.info("ALL PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Files: *_llm_annotations.json, *_agreement_metrics.json, *_hard_cases.json")


if __name__ == "__main__":
    main()

