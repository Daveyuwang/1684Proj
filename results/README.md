# Results

## DeBERTa Predictions

Contains complete test set predictions and calibration metrics for three datasets.

### Prediction Files

Each dataset has a predictions JSON file with the following structure:

- `example_id`: Sequential identifier
- `text`: Input text
- `gold_label`: Ground truth label (integer)
- `predicted_label`: Model prediction (integer)
- `logits`: Raw model outputs (before softmax)
- `probabilities`: Softmax probabilities for each class
- `entropy`: Prediction entropy (uncertainty measure)
- `max_prob`: Maximum probability (confidence score)
- `correct`: Boolean indicating if prediction matches gold label

**Files:**
- `imdb_deberta_predictions.json` (12,500 samples)
- `jigsaw_deberta_predictions.json` (20,000 samples, stratified sample)
- `fever_deberta_predictions.json` (9,999 samples)

### Metrics Files

Each dataset has a metrics JSON file containing:

- Overall performance: accuracy, precision, recall, F1-score
- Calibration metrics: ECE (Expected Calibration Error), Brier score
- Per-class metrics: precision, recall, F1-score, support for each class
- Average entropy and confidence scores

**Files:**
- `imdb_deberta_metrics.json`
- `jigsaw_deberta_metrics.json`
- `fever_deberta_metrics.json`

### Calibration Plots

Contains reliability diagrams and confusion matrices for each dataset:

- Calibration plots show model confidence vs actual accuracy
- Confusion matrices show prediction patterns across classes

**Files:**
- `{dataset}_calibration.png` - Reliability diagrams
- `{dataset}_confusion_matrix.png` - Normalized confusion matrices

### Performance Summary

| Dataset | Accuracy | F1-Score | ECE | Brier Score |
|---------|----------|----------|-----|-------------|
| IMDb    | 95.14%   | 0.9518   | 0.0371 | 0.0430   |
| Jigsaw  | 95.09%   | 0.6772   | 0.0129 | 0.0374   |
| FEVER   | 64.52%   | 0.6449   | 0.0566 | 0.1563   |

Lower ECE and Brier scores indicate better calibration.

---

## LLM Annotations (Qwen2.5-7B-Instruct)

Complete LLM annotation results for three datasets using Qwen2.5-7B-Instruct model.

### Dataset Results

Each dataset directory contains:
- `{dataset}_llm_annotations.json` - Complete annotation results
- `{dataset}_agreement_metrics.json` - Performance metrics
- `{dataset}_hard_cases.json` - Difficult samples for analysis

**Datasets:**
- `imdb/` - Sentiment analysis (12,500 samples, 95.7% accuracy)
- `fever/` - Fact verification (9,999 samples, 53.8% accuracy)  
- `jigsaw/` - Toxicity detection (20,000 samples, 77.6% accuracy)

### Annotation Format

Each annotation includes:
- `row_id`: Example identifier
- `text`: Input text (or preview for hard cases)
- `gold_label`: Ground truth label
- `llm_label`: LLM prediction
- `llm_confidence`: Verbal confidence (low/medium/high)
- `llm_confidence_numeric`: Numeric confidence (0.3/0.6/0.9)
- `llm_rationale`: Explanation (empty in current version)
- `is_valid`: JSON parsing success
- `raw_response`: Raw LLM output

### Jigsaw Stratified Sample

For Jigsaw, a stratified 20k sample was created from the full 97k dataset:
- Location: `jigsaw/jigsaw_sample/`
- Files:
  - `jigsaw_sample20k.csv` - Full sample data
  - `jigsaw_sample20k_ids.csv` - Sample IDs only
  - `jigsaw_20k_sampling_report.json` - Sampling statistics
  - `jigsaw_20k_sampling_comparison.png` - Label distribution visualization

### Performance Summary

| Dataset | Samples | Accuracy | F1-Score | Cohen's Kappa | Avg Confidence |
|---------|---------|----------|----------|---------------|----------------|
| IMDb    | 12,500  | 95.7%    | 0.957    | 0.915         | 0.873          |
| Jigsaw  | 20,000  | 77.6%    | 0.768    | 0.481         | 0.838          |
| FEVER   | 9,999   | 53.8%    | 0.543    | 0.308         | 0.615          |

### Hard Cases

Hard cases are identified based on LLM-gold label disagreement. These samples are particularly useful for:
- Understanding model failure modes
- Manual review and annotation refinement
- Dashboard visualization and filtering

---

## Trust Scorer

Complete trust score classifier implementation for predicting LLM annotation reliability.

### Model Architecture

- **Primary Model**: Logistic Regression (LR) with isotonic calibration
- **Comparison Model**: Random Forest (RF)
- **Features**: 10 features combining LLM confidence, DeBERTa predictions, and text characteristics

### Features

The trust scorer uses 10 features to predict annotation reliability:

**Continuous Features (6):**
1. `llm_confidence_numeric` - LLM confidence level (0.3/0.6/0.9)
2. `deberta_p_max` - Maximum DeBERTa probability
3. `deberta_margin` - Difference between top two probabilities
4. `deberta_entropy` - Prediction entropy (uncertainty)
5. `token_count` - Text length in tokens
6. `punctuation_density` - Punctuation ratio

**Binary Features (4):**
1. `llm_deberta_disagree` - LLM and DeBERTa predict different labels
2. `negation_present` - Text contains negation words
3. `confidence_clash_high_low` - LLM high confidence with DeBERTa low confidence
4. `confidence_clash_low_high` - LLM low confidence with DeBERTa high confidence

### Policy Thresholds

Each model provides three decision policies optimized on the development set:

**1. High-Precision Policy**
- Goal: Maximize accepted accuracy
- Constraint: Coverage >= 30%
- Use case: Critical applications requiring high reliability

**2. Balanced Policy**
- Goal: Maximize F1 score
- Constraint: None
- Use case: Most production scenarios

**3. High-Coverage Policy**
- Goal: Maximize coverage
- Constraint: Accepted accuracy >= 80%
- Use case: Large-scale annotation projects

### Performance Summary

**Logistic Regression (Primary Model):**

| Dataset | ROC-AUC | PR-AUC | ECE | Balanced Policy |
|---------|---------|--------|-----|-----------------|
| IMDb    | 89.4%   | 99.2%  | 0.6% | 95.7% acc @ 99.4% cov |
| Jigsaw  | 97.4%   | 98.9%  | 0.3% | 97.3% acc @ 77.0% cov |
| FEVER   | 73.6%   | 73.7%  | 4.0% | 71.5% acc @ 55.9% cov |

**Random Forest (Comparison):**

| Dataset | ROC-AUC | PR-AUC | ECE | Balanced Policy |
|---------|---------|--------|-----|-----------------|
| IMDb    | 94.6%   | 99.5%  | 0.9% | 97.1% acc @ 97.1% cov |
| Jigsaw  | 97.8%   | 99.0%  | 0.5% | 97.3% acc @ 77.1% cov |
| FEVER   | 75.8%   | 76.0%  | 5.6% | 70.4% acc @ 59.3% cov |

**Key Findings:**
- LR provides better calibration quality (lower ECE) than RF
- LR is more stable, especially on multi-class tasks (FEVER)
- Both models achieve similar discriminative performance (AUC)
- Isotonic calibration reduces ECE by 82-97% for LR

### Predictions Format

Each predictions CSV file contains per-example trust scores and decisions:

- `example_id`: Example identifier
- `text`: Input text
- `gold_label`: Ground truth label
- `llm_label`: LLM prediction
- `llm_confidence`: LLM confidence level
- `deberta_probs_{class}`: DeBERTa probability for each class
- `trust_score_raw`: Raw trust score before calibration
- `trust_score_calibrated`: Calibrated trust score [0, 1]
- `policy_decision_high_precision`: Accept/review decision (high-precision)
- `policy_decision_balanced`: Accept/review decision (balanced)
- `policy_decision_high_coverage`: Accept/review decision (high-coverage)

### Usage Examples

**Loading a trained model:**

```python
import joblib
import json

# Load model and config
pipeline = joblib.load('trust_scorer/models/imdb_lr_pipeline.joblib')
with open('trust_scorer/models/imdb_lr_config.json') as f:
    config = json.load(f)

# Predict trust score
trust_score = pipeline.predict_proba(features)[:, 1]

# Apply policy
threshold = config['policy_thresholds']['balanced']
decision = "ACCEPT" if trust_score >= threshold else "REVIEW"
```

**Loading predictions:**

```python
import pandas as pd

# Load predictions
predictions = pd.read_csv('trust_scorer/imdb_lr_predictions.csv')

# Filter low-trust cases for review
review_cases = predictions[
    predictions['policy_decision_balanced'] == 'review'
]

# Filter by trust score
low_trust = predictions[predictions['trust_score_calibrated'] < 0.5]
```

### Quality Assurance

The trust scorer has been validated through multiple quality checks:

- **Negative Control Test**: Shuffled labels produce AUC ~ 0.50 (no data leakage)
- **Threshold Stability**: Balanced policy thresholds stable across dev set variations
- **Monotonicity**: Coverage-accuracy curves show expected monotonic relationship
- **Calibration**: ECE significantly reduced after isotonic calibration

### Production Package

The `final_export/` directory contains production-ready artifacts:
- Pre-trained LR pipelines for all datasets
- Policy threshold configurations
- Comprehensive performance report
---

## Reproducing Results

### DeBERTa Predictions

```bash
python scripts/save_deberta_predictions.py
```

Requires trained models in `outputs/{dataset}/deberta-v3-base/run1/best_model_calibrated/`

### LLM Annotations

```bash
# All datasets
python scripts/run_llm_annotations_7b.py --datasets all

# Specific dataset
python scripts/run_llm_annotations_7b.py --datasets imdb
```

### Trust Scorer

```bash
# Train all models
python scripts/train_trust_scorer.py --dataset all --model all

# Train specific dataset and model
python scripts/train_trust_scorer.py --dataset imdb --model lr
```

---

## Model Configuration

### DeBERTa Baseline

- **Model**: microsoft/deberta-v3-base
- **Training**: Fine-tuned on each dataset
- **Calibration**: Temperature scaling
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 3

### LLM Annotations

- **Model**: Qwen2.5-7B-Instruct (7B parameters)
- **Inference**: Local with transformers or vLLM
- **Batch Size**: 32 samples
- **Temperature**: 0.1 (near-deterministic)
- **Max Tokens**: 12-24 (task-dependent)
- **Output Format**: Structured JSON with validation

### Trust Scorer

- **Logistic Regression**:
  - Solver: lbfgs
  - C: 1.0
  - Class Weight: balanced
  - Calibration: Isotonic regression

- **Random Forest**:
  - N Estimators: 400
  - Max Depth: None
  - Min Samples Leaf: 5
  - Max Features: sqrt
  - Class Weight: balanced

- **Training Split**: 70% train / 15% dev / 15% test (stratified)
- **Random Seed**: 42 (all experiments)

---

## Dataset Summary

| Dataset | Source | Task | Classes | Samples | Train | Dev | Test |
|---------|--------|------|---------|---------|-------|-----|------|
| IMDb | Sentiment | Binary | 2 | 12,500 | 8,750 | 1,875 | 1,875 |
| Jigsaw | Toxicity | Binary | 2 | 20,000* | 14,000 | 3,000 | 3,000 |
| FEVER | Fact Check | Multi | 3 | 9,999 | 6,999 | 1,500 | 1,500 |

*Jigsaw: Stratified sample from full 97k dataset
