"""
Train trust score classifiers for LLM annotation reliability prediction.

This script trains per-dataset classifiers (IMDb, Jigsaw, FEVER) with Logistic Regression
and Random Forest models, calibrates them, evaluates with coverage-based policies,
and exports predictions and metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve
)
import joblib

# Suppress sklearn FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.trust_scorer import load_joined_data, extract_features
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error with equal-frequency binning."""
    # Use equal-frequency binning
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # ensure last bin includes max value
    
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_prop = mask.mean()
            ece += np.abs(bin_acc - bin_conf) * bin_prop
    
    return ece


def train_classifier(X_train, y_train, X_dev, y_dev, model_type='lr', continuous_features=None, discrete_features=None):
    """
    Train and calibrate a trust score classifier.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels
        X_dev: Dev features DataFrame
        y_dev: Dev labels
        model_type: 'lr' or 'rf'
        continuous_features: List of continuous feature names
        discrete_features: List of discrete feature names
        
    Returns:
        Trained and calibrated pipeline
    """
    logger.info(f"Training {model_type} classifier...")
    
    # Define feature columns
    feature_cols = list(X_train.columns)
    if continuous_features is None:
        continuous_features = [
            'deberta_p_max', 'deberta_margin', 'deberta_entropy',
            'token_count', 'punctuation_density'
        ]
    if discrete_features is None:
        discrete_features = [
            'llm_confidence_numeric', 'llm_deberta_disagree',
            'confidence_clash_high_low', 'confidence_clash_low_high',
            'negation_present'
        ]
    
    # Filter to only existing features
    continuous_features = [f for f in continuous_features if f in feature_cols]
    discrete_features = [f for f in discrete_features if f in feature_cols]
    
    # Create preprocessing pipeline
    if model_type == 'lr':
        # Standardize continuous features for LR
        cont_indices = [feature_cols.index(f) for f in continuous_features]
        disc_indices = [feature_cols.index(f) for f in discrete_features]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), cont_indices),
                ('passthrough', 'passthrough', disc_indices)
            ]
        )
        
        # Logistic Regression
        classifier = LogisticRegression(
            solver='lbfgs',
            C=1.0,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000
        )
    else:  # rf
        # No preprocessing for RF
        preprocessor = 'passthrough'
        
        # Random Forest
        classifier = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    # Create uncalibrated pipeline
    uncalibrated_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Fit uncalibrated model
    uncalibrated_pipeline.fit(X_train.values, y_train)
    
    # Get uncalibrated predictions on dev
    y_dev_prob_uncal = uncalibrated_pipeline.predict_proba(X_dev.values)[:, 1]
    
    # Apply isotonic calibration on dev set
    logger.info("Applying isotonic calibration...")
    calibrated_classifier = CalibratedClassifierCV(
        uncalibrated_pipeline,
        method='isotonic',
        cv='prefit'
    )
    calibrated_classifier.fit(X_dev.values, y_dev)
    
    # Create final pipeline
    final_pipeline = calibrated_classifier
    
    # Calculate pre and post calibration ECE on dev
    ece_before = calculate_ece(y_dev, y_dev_prob_uncal, n_bins=10 if len(y_dev) >= 2000 else 5)
    y_dev_prob_cal = final_pipeline.predict_proba(X_dev.values)[:, 1]
    ece_after = calculate_ece(y_dev, y_dev_prob_cal, n_bins=10 if len(y_dev) >= 2000 else 5)
    
    logger.info(f"Dev ECE - Before calibration: {ece_before:.4f}, After calibration: {ece_after:.4f}")
    
    return final_pipeline, ece_before, ece_after


def tune_policies(y_true, y_scores, baseline_acc=None):
    """
    Tune decision policy thresholds on dev set.
    
    Args:
        y_true: True trust labels
        y_scores: Predicted trust scores
        baseline_acc: Baseline accuracy (optional)
        
    Returns:
        Dictionary of policy thresholds
    """
    logger.info("Tuning policy thresholds...")
    
    # Get unique scores to search over
    unique_scores = np.unique(y_scores)
    if len(unique_scores) > 200:
        thresholds = np.linspace(0, 1, 200)
    else:
        thresholds = np.sort(unique_scores)
    
    policies = {}
    
    # High-precision policy: maximize accepted accuracy with coverage >= 30%
    best_acc = 0
    best_thresh_hp = 0.5
    for thresh in thresholds:
        accepted = y_scores >= thresh
        if accepted.sum() == 0:
            continue
        coverage = accepted.mean()
        if coverage >= 0.30:
            acc = y_true[accepted].mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh_hp = thresh
    policies['high_precision'] = best_thresh_hp
    logger.info(f"High-precision threshold: {best_thresh_hp:.4f} (coverage >= 30%, accepted_acc = {best_acc:.4f})")
    
    # Balanced policy: maximize F1-score
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_thresh_bal = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
    policies['balanced'] = best_thresh_bal
    logger.info(f"Balanced threshold: {best_thresh_bal:.4f} (F1 = {f1_scores[best_f1_idx]:.4f})")
    
    # High-coverage policy: maximize coverage with accepted accuracy >= 80%
    target_acc = max(0.80, (baseline_acc + 0.05) if baseline_acc else 0.80)
    best_cov = 0
    best_thresh_hc = 0.5
    for thresh in thresholds:
        accepted = y_scores >= thresh
        if accepted.sum() == 0:
            continue
        acc = y_true[accepted].mean()
        if acc >= target_acc:
            coverage = accepted.mean()
            if coverage > best_cov:
                best_cov = coverage
                best_thresh_hc = thresh
    policies['high_coverage'] = best_thresh_hc
    logger.info(f"High-coverage threshold: {best_thresh_hc:.4f} (accepted_acc >= {target_acc:.2f}, coverage = {best_cov:.4f})")
    
    return policies


def evaluate_on_test(pipeline, X_test, y_test, policies, dataset_name, model_type, output_dir):
    """Evaluate calibrated model on test set and save metrics."""
    logger.info(f"Evaluating on test set ({len(y_test)} examples)...")
    
    # Get predictions
    y_prob = pipeline.predict_proba(X_test.values)[:, 1]
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name,
        'model_type': model_type,
        'n_test': len(y_test),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
        'brier_score': float(brier_score_loss(y_test, y_prob)),
        'ece': float(calculate_ece(y_test, y_prob, n_bins=10 if len(y_test) >= 2000 else 5)),
        'policy_thresholds': {k: float(v) for k, v in policies.items()},
        'policy_metrics': {}
    }
    
    # Evaluate each policy
    for policy_name, threshold in policies.items():
        accepted = y_prob >= threshold
        if accepted.sum() > 0:
            y_accepted_true = y_test[accepted]
            y_accepted_pred = (y_prob[accepted] >= 0.5).astype(int)
            
            policy_metrics = {
                'threshold': float(threshold),
                'coverage': float(accepted.mean()),
                'accepted_accuracy': float(y_accepted_true.mean()),
                'precision': float(precision_score(y_test, accepted, zero_division=0)),
                'recall': float(recall_score(y_test, accepted, zero_division=0)),
                'f1_score': float(f1_score(y_test, accepted, zero_division=0))
            }
        else:
            policy_metrics = {
                'threshold': float(threshold),
                'coverage': 0.0,
                'accepted_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        metrics['policy_metrics'][policy_name] = policy_metrics
        logger.info(f"  {policy_name}: coverage={policy_metrics['coverage']:.3f}, "
                   f"accepted_acc={policy_metrics['accepted_accuracy']:.3f}, "
                   f"F1={policy_metrics['f1_score']:.3f}")
    
    # Save metrics
    metrics_path = output_dir / f"{dataset_name}_{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info(f"Test metrics - ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, "
               f"Brier: {metrics['brier_score']:.4f}, ECE: {metrics['ece']:.4f}")
    
    return metrics, y_prob


def plot_reliability_and_coverage(y_test, y_prob, policies, dataset_name, model_type, output_dir):
    """Generate reliability diagram and coverage-accuracy curve."""
    logger.info("Generating plots...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Reliability diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    n_bins = 10 if len(y_test) >= 2000 else 5
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy='quantile')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(prob_pred, prob_true, 'o-', label='Model')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title(f'{dataset_name} - {model_type} - Reliability Diagram')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Coverage-accuracy curve
    sorted_indices = np.argsort(y_prob)[::-1]  # descending order
    y_sorted = y_test[sorted_indices]
    coverages = np.linspace(0, 1, 100)
    accuracies = []
    for cov in coverages:
        n_accept = int(cov * len(y_sorted))
        if n_accept > 0:
            acc = y_sorted[:n_accept].mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    ax2.plot(coverages, accuracies, 'b-', linewidth=2, label='Trust scorer')
    ax2.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
    
    # Mark policy operating points
    for policy_name, threshold in policies.items():
        accepted = y_prob >= threshold
        if accepted.sum() > 0:
            cov = accepted.mean()
            acc = y_test[accepted].mean()
            marker = 'o' if policy_name == 'balanced' else ('s' if policy_name == 'high_precision' else 'd')
            ax2.plot(cov, acc, marker, markersize=10, label=f'{policy_name}')
    
    ax2.set_xlabel('Coverage (fraction accepted)')
    ax2.set_ylabel('Accuracy on accepted')
    ax2.set_title(f'{dataset_name} - {model_type} - Coverage-Accuracy Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plot_path = plots_dir / f"{dataset_name}_{model_type}_reliability_coverage.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {plot_path}")


def export_predictions(df_test, y_prob, policies, dataset_name, model_type, output_dir):
    """Export per-example predictions to CSV."""
    logger.info("Exporting per-example predictions...")
    
    # Create export dataframe
    export_df = pd.DataFrame({
        'example_id': df_test['example_id'].values,
        'gold_label': df_test['gold_label'].values,
        'llm_label': df_test['llm_label'].values,
        'llm_confidence': df_test['llm_confidence'].values,
        'llm_confidence_numeric': df_test['llm_confidence_numeric'].values,
        'deberta_predicted_label': df_test['deberta_predicted_label'].values,
        'llm_deberta_disagree': df_test['llm_deberta_disagree'].values,
        'trust_score_calibrated': y_prob
    })
    
    # Add gold label strings if available
    if 'gold_label_str' in df_test.columns:
        export_df['gold_label_str'] = df_test['gold_label_str'].values
    if 'llm_label_str' in df_test.columns:
        export_df['llm_label_str'] = df_test['llm_label_str'].values
    
    # Add DeBERTa probabilities (wide format)
    prob_cols = [col for col in df_test.columns if col.startswith('prob_class')]
    for prob_col in prob_cols:
        export_df[prob_col] = df_test[prob_col].values
    
    # Add policy decisions
    for policy_name, threshold in policies.items():
        export_df[f'policy_{policy_name}_decision'] = (y_prob >= threshold).astype(int)
    
    # Add text if available
    if 'text' in df_test.columns:
        export_df['text'] = df_test['text'].values
    
    # Save to CSV
    export_path = output_dir / f"{dataset_name}_{model_type}_predictions.csv"
    export_df.to_csv(export_path, index=False)
    logger.info(f"Exported predictions to {export_path}")
    
    return export_df


def negative_control_test(X_train, y_train, X_dev, y_dev, model_type='lr'):
    """Run negative control (label shuffle) to check for leakage."""
    logger.info("Running negative control test (label shuffle)...")
    
    # Shuffle labels
    y_train_shuffled = y_train.copy()
    np.random.shuffle(y_train_shuffled)
    
    # Train model
    pipeline, _, _ = train_classifier(X_train, y_train_shuffled, X_dev, y_dev, model_type=model_type)
    
    # Evaluate
    y_dev_prob = pipeline.predict_proba(X_dev.values)[:, 1]
    auc = roc_auc_score(y_dev, y_dev_prob)
    
    logger.info(f"Negative control ROC-AUC: {auc:.4f} (expect ~0.5)")
    
    if auc > 0.6:
        logger.warning(f"Negative control AUC = {auc:.4f} > 0.6, possible data leakage!")
    else:
        logger.info("Negative control passed (no obvious leakage)")
    
    return auc


def train_dataset(dataset_name, model_type, output_dir, run_negative_control=True):
    """Train trust scorer for a single dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_type} for {dataset_name}")
    logger.info(f"{'='*80}\n")
    
    # Load and join data
    train_df, dev_df, test_df = load_joined_data(dataset_name)
    
    # Extract features
    train_df, dev_percentiles = extract_features(train_df, dev_percentiles=None)
    dev_df, _ = extract_features(dev_df, dev_percentiles=dev_percentiles)
    test_df, _ = extract_features(test_df, dev_percentiles=dev_percentiles)
    
    # Define feature columns
    feature_cols = [
        'llm_confidence_numeric',
        'deberta_p_max',
        'deberta_margin',
        'deberta_entropy',
        'llm_deberta_disagree',
        'confidence_clash_high_low',
        'confidence_clash_low_high',
        'token_count',
        'punctuation_density',
        'negation_present'
    ]
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['trust_label'].values
    X_dev = dev_df[feature_cols]
    y_dev = dev_df['trust_label'].values
    X_test = test_df[feature_cols]
    y_test = test_df['trust_label'].values
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Dev: {X_dev.shape}, Test: {X_test.shape}")
    logger.info(f"Trust label distribution - Train: {y_train.mean():.3f}, Dev: {y_dev.mean():.3f}, Test: {y_test.mean():.3f}")
    
    # Negative control test
    if run_negative_control:
        negative_control_test(X_train, y_train, X_dev, y_dev, model_type=model_type)
    
    # Train and calibrate model
    pipeline, ece_before, ece_after = train_classifier(
        X_train, y_train, X_dev, y_dev, model_type=model_type
    )
    
    # Tune policies on dev
    y_dev_prob = pipeline.predict_proba(X_dev.values)[:, 1]
    baseline_acc = y_train.mean()
    policies = tune_policies(y_dev, y_dev_prob, baseline_acc=baseline_acc)
    
    # Evaluate on test
    metrics, y_test_prob = evaluate_on_test(
        pipeline, X_test, y_test, policies, dataset_name, model_type, output_dir
    )
    
    # Add calibration metrics to saved metrics
    metrics['ece_before_calibration'] = float(ece_before)
    metrics['ece_after_calibration'] = float(ece_after)
    metrics_path = output_dir / f"{dataset_name}_{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    plot_reliability_and_coverage(y_test, y_test_prob, policies, dataset_name, model_type, output_dir)
    
    # Export predictions
    export_predictions(test_df, y_test_prob, policies, dataset_name, model_type, output_dir)
    
    # Save model pipeline
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    pipeline_path = models_dir / f"{dataset_name}_{model_type}_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Saved pipeline to {pipeline_path}")
    
    # Save configuration
    config_data = {
        'dataset': dataset_name,
        'model_type': model_type,
        'feature_columns': feature_cols,
        'dev_percentiles': {k: float(v) for k, v in dev_percentiles.items()},
        'policy_thresholds': {k: float(v) for k, v in policies.items()},
        'random_state': RANDOM_STATE,
        'package_versions': {
            'sklearn': __import__('sklearn').__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__
        }
    }
    config_path = models_dir / f"{dataset_name}_{model_type}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")
    
    logger.info(f"\nCompleted training {model_type} for {dataset_name}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train trust score classifiers')
    parser.add_argument('--dataset', type=str, choices=['imdb', 'jigsaw', 'fever', 'all'], default='all',
                       help='Dataset to train on')
    parser.add_argument('--model', type=str, choices=['lr', 'rf', 'all'], default='all',
                       help='Model type to train')
    parser.add_argument('--no-negative-control', action='store_true',
                       help='Skip negative control test')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = config.RESULTS_DIR / "trust_scorer"
    output_dir.mkdir(exist_ok=True)
    
    # Determine datasets and models to train
    datasets = ['imdb', 'jigsaw', 'fever'] if args.dataset == 'all' else [args.dataset]
    models = ['lr', 'rf'] if args.model == 'all' else [args.model]
    
    # Train all combinations
    all_metrics = {}
    for dataset in datasets:
        for model in models:
            try:
                metrics = train_dataset(
                    dataset, model, output_dir,
                    run_negative_control=not args.no_negative_control
                )
                all_metrics[f"{dataset}_{model}"] = metrics
            except Exception as e:
                logger.error(f"Failed to train {model} for {dataset}: {e}", exc_info=True)
    
    # Save summary of all results
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nSaved training summary to {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed for all datasets and models")
    logger.info("="*80)


if __name__ == "__main__":
    main()

