"""
Ablation study for trust score classifiers.

This script runs a progressive feature ablation study on IMDb and Jigsaw datasets
to understand the contribution of different feature groups to trust score prediction.

Feature Groups:
- G1: LLM Confidence (llm_confidence_numeric)
- G2: DeBERTa Confidence/Uncertainty (deberta_p_max, deberta_margin, deberta_entropy)
- G3: Disagreement/Clash (llm_deberta_disagree, confidence_clash_high_low, confidence_clash_low_high)
- G4: Text Features (token_count, punctuation_density, negation_present)

Configurations:
1. llm_only: G1
2. deberta_only: G2
3. llm_deberta: G1 + G2
4. llm_deberta_disagree: G1 + G2 + G3
5. full: G1 + G2 + G3 + G4
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)

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

# Seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def define_feature_configs():
    """
    Define feature configurations for ablation study.
    
    Returns:
        Dictionary mapping config names to feature lists
    """
    configs = {
        'llm_only': [
            'llm_confidence_numeric'
        ],
        'deberta_only': [
            'deberta_p_max',
            'deberta_margin',
            'deberta_entropy'
        ],
        'llm_deberta': [
            'llm_confidence_numeric',
            'deberta_p_max',
            'deberta_margin',
            'deberta_entropy'
        ],
        'llm_deberta_disagree': [
            'llm_confidence_numeric',
            'deberta_p_max',
            'deberta_margin',
            'deberta_entropy',
            'llm_deberta_disagree',
            'confidence_clash_high_low',
            'confidence_clash_low_high'
        ],
        'full': [
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
    }
    return configs


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


def train_classifier(X_train, y_train, X_dev, y_dev, feature_columns=None):
    """
    Train and calibrate a logistic regression trust score classifier.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels
        X_dev: Dev features DataFrame
        y_dev: Dev labels
        feature_columns: List of feature column names to use
        
    Returns:
        Tuple of (trained pipeline, ece_before, ece_after)
    """
    logger.info(f"Training LR classifier with {len(feature_columns)} features...")
    
    # Define base continuous and discrete features
    continuous_base = [
        'deberta_p_max', 'deberta_margin', 'deberta_entropy',
        'token_count', 'punctuation_density'
    ]
    discrete_base = [
        'llm_confidence_numeric', 'llm_deberta_disagree',
        'confidence_clash_high_low', 'confidence_clash_low_high',
        'negation_present'
    ]
    
    # Filter to only features in feature_columns
    continuous_features = [f for f in feature_columns if f in continuous_base]
    discrete_features = [f for f in feature_columns if f in discrete_base]
    
    logger.info(f"  Continuous: {continuous_features}")
    logger.info(f"  Discrete: {discrete_features}")
    
    # Get column indices
    cont_indices = [feature_columns.index(f) for f in continuous_features]
    disc_indices = [feature_columns.index(f) for f in discrete_features]
    
    # Create preprocessing pipeline
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
    
    # Calculate pre and post calibration ECE on dev
    ece_before = calculate_ece(y_dev, y_dev_prob_uncal, n_bins=10 if len(y_dev) >= 2000 else 5)
    y_dev_prob_cal = calibrated_classifier.predict_proba(X_dev.values)[:, 1]
    ece_after = calculate_ece(y_dev, y_dev_prob_cal, n_bins=10 if len(y_dev) >= 2000 else 5)
    
    logger.info(f"Dev ECE - Before: {ece_before:.4f}, After: {ece_after:.4f}")
    
    return calibrated_classifier, ece_before, ece_after


def tune_policies(y_true, y_scores):
    """
    Tune decision policy thresholds on dev set.
    
    Args:
        y_true: True trust labels
        y_scores: Predicted trust scores
        
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
    
    # Balanced policy: maximize F1-score
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_thresh_bal = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
    policies['balanced'] = best_thresh_bal
    
    # High-coverage policy: maximize coverage with accepted accuracy >= 80%
    target_acc = 0.80
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
    
    logger.info(f"  High-precision: {best_thresh_hp:.4f}")
    logger.info(f"  Balanced: {best_thresh_bal:.4f}")
    logger.info(f"  High-coverage: {best_thresh_hc:.4f}")
    
    return policies


def evaluate_on_test(pipeline, X_test, y_test, policies):
    """
    Evaluate calibrated model on test set.
    
    Args:
        pipeline: Trained and calibrated pipeline
        X_test: Test features
        y_test: Test labels
        policies: Dict of policy thresholds
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating on test set ({len(y_test)} examples)...")
    
    # Get predictions
    y_prob = pipeline.predict_proba(X_test.values)[:, 1]
    
    # Calculate overall metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = calculate_ece(y_test, y_prob, n_bins=10 if len(y_test) >= 2000 else 5)
    
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")
    logger.info(f"  Brier: {brier:.4f}")
    logger.info(f"  ECE: {ece:.4f}")
    
    # Evaluate each policy
    policy_metrics = {}
    for policy_name, threshold in policies.items():
        accepted = y_prob >= threshold
        n_accepted = accepted.sum()
        
        if n_accepted == 0:
            logger.warning(f"  {policy_name}: No examples accepted!")
            policy_metrics[policy_name] = {
                'threshold': threshold,
                'coverage': 0.0,
                'accepted_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            continue
        
        coverage = accepted.mean()
        accepted_accuracy = y_test[accepted].mean()
        
        # Calculate precision, recall, F1
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"  {policy_name}: Coverage={coverage:.3f}, Acc={accepted_accuracy:.3f}, F1={f1:.3f}")
        
        policy_metrics[policy_name] = {
            'threshold': threshold,
            'coverage': coverage,
            'accepted_accuracy': accepted_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    metrics = {
        'n_test': len(y_test),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'ece': ece,
        'policy_metrics': policy_metrics
    }
    
    return metrics


def train_single_config(config_name, features, dataset_name, train_df, dev_df, test_df, dev_percentiles):
    """
    Train and evaluate a single feature configuration.
    
    Args:
        config_name: Name of configuration (e.g., 'llm_only')
        features: List of feature column names
        dataset_name: Name of dataset ('imdb' or 'jigsaw')
        train_df: Training data
        dev_df: Dev data
        test_df: Test data
        dev_percentiles: Percentiles from dev set for feature extraction
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Config: {config_name} | Dataset: {dataset_name}")
    logger.info(f"Features ({len(features)}): {features}")
    logger.info(f"{'='*80}")
    
    # Select only the required features
    X_train = train_df[features]
    y_train = train_df['trust_label'].values
    
    X_dev = dev_df[features]
    y_dev = dev_df['trust_label'].values
    
    X_test = test_df[features]
    y_test = test_df['trust_label'].values
    
    # Train classifier
    pipeline, ece_before, ece_after = train_classifier(
        X_train, y_train, X_dev, y_dev, feature_columns=features
    )
    
    # Tune policies on dev
    y_dev_prob = pipeline.predict_proba(X_dev.values)[:, 1]
    policies = tune_policies(y_dev, y_dev_prob)
    
    # Evaluate on test
    metrics = evaluate_on_test(pipeline, X_test, y_test, policies)
    
    # Add calibration metrics
    metrics['ece_before_calibration'] = ece_before
    metrics['ece_after_calibration'] = ece_after
    
    # Add metadata
    metrics['config_name'] = config_name
    metrics['dataset'] = dataset_name
    metrics['n_features'] = len(features)
    metrics['features'] = features
    
    return metrics


def run_ablation_study(datasets=['imdb', 'jigsaw'], output_dir=None):
    """
    Run ablation study on specified datasets.
    
    Args:
        datasets: List of dataset names
        output_dir: Output directory for results
        
    Returns:
        Dictionary of all results
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "trust_scorer" / "ablation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Define configurations
    configs = define_feature_configs()
    logger.info(f"\nConfigurations: {list(configs.keys())}")
    logger.info(f"Datasets: {datasets}")
    
    # Store all results
    all_results = {}
    
    # Run ablation for each dataset
    for dataset_name in datasets:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# DATASET: {dataset_name.upper()}")
        logger.info(f"{'#'*80}")
        
        # Load data
        train_df, dev_df, test_df = load_joined_data(dataset_name)
        
        # Extract features
        train_df, dev_percentiles = extract_features(train_df, dev_percentiles=None)
        dev_df, _ = extract_features(dev_df, dev_percentiles=dev_percentiles)
        test_df, _ = extract_features(test_df, dev_percentiles=dev_percentiles)
        
        logger.info(f"Data loaded - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        
        # Run each configuration
        for config_name, features in configs.items():
            try:
                metrics = train_single_config(
                    config_name, features, dataset_name,
                    train_df, dev_df, test_df, dev_percentiles
                )
                
                # Store results
                result_key = f"{dataset_name}_{config_name}"
                all_results[result_key] = metrics
                
                # Save individual result
                result_file = output_dir / f"{result_key}_metrics.json"
                with open(result_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Saved: {result_file}")
                
            except Exception as e:
                logger.error(f"Error in {config_name} on {dataset_name}: {e}", exc_info=True)
                continue
    
    # Save summary
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved summary: {summary_file}")
    
    return all_results


def create_comparison_table(results, output_dir):
    """
    Create comparison table from ablation results.
    
    Args:
        results: Dictionary of all results
        output_dir: Output directory
        
    Returns:
        DataFrame with comparison table
    """
    logger.info("\nCreating comparison table...")
    
    rows = []
    for key, metrics in results.items():
        # Extract balanced policy metrics
        balanced = metrics.get('policy_metrics', {}).get('balanced', {})
        
        row = {
            'Dataset': metrics['dataset'],
            'Config': metrics['config_name'],
            'N_Features': metrics['n_features'],
            'ROC_AUC': metrics['roc_auc'],
            'PR_AUC': metrics['pr_auc'],
            'Brier': metrics['brier_score'],
            'ECE': metrics['ece'],
            'Balanced_Threshold': balanced.get('threshold', 0),
            'Balanced_Coverage': balanced.get('coverage', 0),
            'Balanced_Accuracy': balanced.get('accepted_accuracy', 0),
            'Balanced_Precision': balanced.get('precision', 0),
            'Balanced_Recall': balanced.get('recall', 0),
            'Balanced_F1': balanced.get('f1_score', 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by dataset and config order
    config_order = ['llm_only', 'deberta_only', 'llm_deberta', 'llm_deberta_disagree', 'full']
    df['Config_Order'] = df['Config'].apply(lambda x: config_order.index(x) if x in config_order else 999)
    df = df.sort_values(['Dataset', 'Config_Order']).drop('Config_Order', axis=1)
    
    # Save CSV
    csv_file = output_dir / "ablation_comparison_table.csv"
    df.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"Saved table: {csv_file}")
    
    # Print to console
    print("\n" + "="*120)
    print("ABLATION STUDY COMPARISON TABLE")
    print("="*120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(df.to_string(index=False))
    print("="*120)
    
    return df


def create_plots(results, output_dir):
    """
    Create visualization plots for ablation results.
    
    Args:
        results: Dictionary of all results
        output_dir: Output directory
    """
    logger.info("\nCreating visualization plots...")
    
    plot_dir = output_dir / "ablation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    data = []
    for key, metrics in results.items():
        balanced = metrics.get('policy_metrics', {}).get('balanced', {})
        data.append({
            'dataset': metrics['dataset'],
            'config': metrics['config_name'],
            'n_features': metrics['n_features'],
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'balanced_f1': balanced.get('f1_score', 0),
            'balanced_coverage': balanced.get('coverage', 0),
            'brier': metrics['brier_score'],
            'ece': metrics['ece']
        })
    
    df = pd.DataFrame(data)
    
    # Config order and pretty names mapping
    config_order = ['llm_only', 'deberta_only', 'llm_deberta', 'llm_deberta_disagree', 'full']
    config_labels = {
        'llm_only': 'LLM Only',
        'deberta_only': 'DeBERTa Only',
        'llm_deberta': 'LLM+DeBERTa',
        'llm_deberta_disagree': 'LLM+DeBERTa\n+Disagree',
        'full': 'Full Model'
    }
    
    df['config'] = pd.Categorical(df['config'], categories=config_order, ordered=True)
    df['config_label'] = df['config'].map(config_labels)
    df = df.sort_values(['dataset', 'config'])
    
    # Plot 1: ROC-AUC comparison (with pretty labels)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='config_label', y='roc_auc', hue='dataset', palette='Set2')
    plt.title('ROC-AUC Comparison Across Configurations', fontsize=14, fontweight='bold')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.xticks(rotation=0, ha='center')
    plt.ylim(0.5, 1.0)
    plt.legend(title='Dataset', loc='lower right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot1_file = plot_dir / "roc_auc_comparison.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot1_file}")
    plt.close()
    
    # Plot 2: Balanced F1 vs Coverage scatter (IMPROVED - handle overlaps)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    datasets = df['dataset'].unique()
    colors = {
        'llm_only': '#1f77b4',
        'deberta_only': '#ff7f0e',
        'llm_deberta': '#2ca02c',
        'llm_deberta_disagree': '#d62728',
        'full': '#9467bd'
    }
    markers = {
        'llm_only': 'o',
        'deberta_only': 's',
        'llm_deberta': '^',
        'llm_deberta_disagree': 'D',
        'full': 'v'
    }
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset]
        
        # Define custom offsets for each config to avoid overlaps
        if dataset == 'imdb':
            # IMDb: points are clustered around (1.0, 0.978)
            # Adjusted to ensure Full Model point is completely visible
            offsets = {
                'llm_only': (20, 18),
                'deberta_only': (-85, 12),
                'llm_deberta': (20, -22),
                'llm_deberta_disagree': (-100, -28),  # Move further left and down
                'full': (70, -6)  # Move even further right and slightly up
            }
        else:  # jigsaw
            # Jigsaw: more spread, but still some overlap
            offsets = {
                'llm_only': (20, 18),
                'deberta_only': (-85, 18),
                'llm_deberta': (20, -22),
                'llm_deberta_disagree': (20, 12),
                'full': (-70, -8)
            }
        
        for config in config_order:
            config_data = subset[subset['config'] == config]
            if len(config_data) > 0:
                x = config_data['balanced_coverage'].values[0]
                y = config_data['balanced_f1'].values[0]
                
                # Improved annotation positioning to avoid overlaps
                offset_x, offset_y = offsets.get(config, (10, 10))
                
                ax.annotate(config_labels[config].replace('\n', ' '), 
                           (x, y),
                           xytext=(offset_x, offset_y),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=colors.get(config, 'gray'), alpha=0.85, linewidth=1.2),
                           arrowprops=dict(arrowstyle='->', 
                                         connectionstyle='arc3,rad=0.2',
                                         color=colors.get(config, 'gray'),
                                         lw=1.2),
                           zorder=4)
                
                # Plot markers AFTER annotations so they appear on top
                ax.scatter(x, y, s=180, 
                          color=colors.get(config, 'gray'),
                          marker=markers.get(config, 'o'),
                          label=config_labels[config].replace('\n', ' '),
                          alpha=0.9, edgecolors='black', linewidths=1.8, zorder=10)
        
        ax.set_xlabel('Coverage (Balanced Policy)', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1 Score (Balanced Policy)', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, zorder=1)
        
        # Adjust limits based on dataset
        if dataset == 'imdb':
            ax.set_xlim(0.98, 1.008)  # Extended slightly for Full Model label
            ax.set_ylim(0.9738, 0.9795)
        else:  # jigsaw
            ax.set_xlim(0.74, 1.015)
            ax.set_ylim(0.87, 0.98)
        
        # Place legend with smaller marker icons to avoid overlap
        if idx == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.005, 0.98), 
                     fontsize=7.5, framealpha=0.95, edgecolor='gray',
                     handletextpad=0.4, borderpad=0.4, labelspacing=0.35,
                     markerscale=0.4)  # Much smaller markers in legend
    
    plt.suptitle('Balanced F1 vs Coverage Trade-off', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot2_file = plot_dir / "f1_coverage_scatter.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot2_file}")
    plt.close()
    
    # Plot 3: Performance gain from baseline (with pretty labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset].sort_values('config')
        
        # Compute relative gains from llm_only baseline
        baseline_auc = subset[subset['config'] == 'llm_only']['roc_auc'].values[0]
        subset['roc_auc_gain'] = ((subset['roc_auc'] - baseline_auc) / baseline_auc) * 100
        
        baseline_f1 = subset[subset['config'] == 'llm_only']['balanced_f1'].values[0]
        subset['f1_gain'] = ((subset['balanced_f1'] - baseline_f1) / baseline_f1) * 100
        
        x_pos = range(len(subset))
        ax.plot(x_pos, subset['roc_auc_gain'], marker='o', label='ROC-AUC Gain',
               linewidth=2.5, markersize=10, color='#2ca02c')
        ax.plot(x_pos, subset['f1_gain'], marker='s', label='F1 Gain',
               linewidth=2.5, markersize=10, color='#d62728')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subset['config_label'], rotation=0, ha='center')
        ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Relative Gain (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    plt.suptitle('Progressive Performance Gain from LLM-Only Baseline',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot3_file = plot_dir / "progressive_gain.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot3_file}")
    plt.close()
    
    logger.info(f"All plots saved to: {plot_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run trust scorer ablation study')
    parser.add_argument('--datasets', nargs='+', default=['imdb', 'jigsaw'],
                       help='Datasets to run ablation on (default: imdb jigsaw)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/trust_scorer/ablation)')
    parser.add_argument('--config', type=str, default=None,
                       help='Run specific config only (default: all)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("TRUST SCORER ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Output: {args.output or 'results/trust_scorer/ablation'}")
    
    # Run ablation study
    results = run_ablation_study(datasets=args.datasets, output_dir=args.output)
    
    # Create comparison table
    output_dir = Path(args.output) if args.output else config.RESULTS_DIR / "trust_scorer" / "ablation"
    create_comparison_table(results, output_dir)
    
    # Create plots
    if not args.skip_plots:
        create_plots(results, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total configurations tested: {len(results)}")


if __name__ == '__main__':
    main()

