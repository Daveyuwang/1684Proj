"""
Trust score prediction for LLM annotation reliability.

This module implements classifiers to predict whether LLM annotations
are trustworthy based on features from LLM confidence, supervised model
disagreement, text characteristics, and more.
"""

import logging
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

from models.text_features import TextFeatureExtractor, create_feature_extractor
import config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_joined_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and join LLM annotations with DeBERTa predictions.
    
    Args:
        dataset_name: Name of dataset (imdb, jigsaw, fever)
        
    Returns:
        Tuple of (train_df, dev_df, test_df) with joined data
    """
    logger.info(f"Loading data for {dataset_name}...")
    
    # Load original dataset to get split information
    from data.dataset_loader import IMDBLoader, JigsawLoader, FEVERLoader
    
    if dataset_name == 'imdb':
        loader = IMDBLoader(dataset_name)
    elif dataset_name == 'jigsaw':
        loader = JigsawLoader(dataset_name)
    elif dataset_name == 'fever':
        loader = FEVERLoader(dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    original_data = loader.load_data()
    
    # Create mapping from row position to split
    # Determine the size of each split to map row_ids to splits
    train_size = len(original_data['train'])
    dev_size = len(original_data['dev'])
    test_size = len(original_data['test'])
    
    def get_split_from_row_id(row_id):
        """Map row_id to split based on position."""
        if row_id < train_size:
            return 'train'
        elif row_id < train_size + dev_size:
            return 'dev'
        else:
            return 'test'
    
    # Load LLM annotations
    llm_annotations_path = config.RESULTS_DIR / f"llm_annotations_7b" / dataset_name / f"{dataset_name}_llm_annotations.json"
    with open(llm_annotations_path, 'r') as f:
        llm_data = json.load(f)
    llm_df = pd.DataFrame(llm_data)
    
    # Deduplicate LLM annotations (keep last occurrence)
    n_before = len(llm_df)
    llm_df = llm_df.drop_duplicates(subset='row_id', keep='last')
    n_after = len(llm_df)
    if n_before != n_after:
        logger.info(f"Removed {n_before - n_after} duplicate LLM annotations (kept last occurrence)")
    
    # Add split information to LLM data
    llm_df['split'] = llm_df['row_id'].apply(get_split_from_row_id)
    
    # Load DeBERTa predictions
    deberta_pred_path = config.RESULTS_DIR / "deberta_predictions" / f"{dataset_name}_deberta_predictions.json"
    with open(deberta_pred_path, 'r') as f:
        deberta_data = json.load(f)
    
    # Convert DeBERTa data to DataFrame
    deberta_df = pd.DataFrame({
        'example_id': deberta_data['example_id'],
        'deberta_predicted_label': deberta_data['predicted_label']
    })
    
    # Add split information to DeBERTa data
    deberta_df['split'] = deberta_df['example_id'].apply(get_split_from_row_id)
    
    # Add probability columns
    probs_array = np.array(deberta_data['probabilities'])
    num_classes = probs_array.shape[1]
    for i in range(num_classes):
        deberta_df[f'prob_class{i}'] = probs_array[:, i]
    
    # Add text if available in DeBERTa data
    if 'text' in deberta_data:
        deberta_df['text'] = deberta_data['text']
    
    # For Jigsaw, filter to 20k sample if available
    if dataset_name == 'jigsaw':
        # Try 20k first, then fall back to 10k
        sample_ids_path = config.RESULTS_DIR / "llm_annotations_7b" / dataset_name / "jigsaw_sample" / "jigsaw_sample20k_ids.csv"
        if not sample_ids_path.exists():
            sample_ids_path = config.RESULTS_DIR / "llm_annotations_7b" / dataset_name / "jigsaw_sample" / "jigsaw_sample10k_ids.csv"
        
        if sample_ids_path.exists():
            sample_ids_df = pd.read_csv(sample_ids_path)
            # The column might be 'example_id' or 'original_index'
            id_col = 'example_id' if 'example_id' in sample_ids_df.columns else 'original_index'
            sample_ids = set(sample_ids_df[id_col].values)
            # For Jigsaw, LLM annotations use 'example_id' already (not 'row_id')
            llm_id_col = 'example_id' if 'example_id' in llm_df.columns else 'row_id'
            llm_df = llm_df[llm_df[llm_id_col].isin(sample_ids)].copy()
            deberta_df = deberta_df[deberta_df['example_id'].isin(sample_ids)].copy()
            sample_size = "20k" if "20k" in str(sample_ids_path) else "10k"
            logger.info(f"Filtered to Jigsaw {sample_size} sample: {len(llm_df)} examples")
    
    # Join on example IDs (row_id in LLM data, example_id in DeBERTa data)
    # For Jigsaw, 'example_id' already exists; for others, rename 'row_id' to 'example_id'
    if 'row_id' in llm_df.columns and 'example_id' not in llm_df.columns:
        llm_df = llm_df.rename(columns={'row_id': 'example_id'})
    joined_df = llm_df.merge(deberta_df, on='example_id', how='inner', suffixes=('', '_deberta'))
    
    # Validate join
    assert len(joined_df) > 0, f"No matching examples after join for {dataset_name}"
    assert joined_df['example_id'].nunique() == len(joined_df), f"Duplicate example IDs after join for {dataset_name}"
    
    # Log join statistics
    logger.info(f"Join stats for {dataset_name}:")
    logger.info(f"  LLM annotations: {len(llm_df)}")
    logger.info(f"  DeBERTa predictions: {len(deberta_df)}")
    logger.info(f"  Joined: {len(joined_df)}")
    logger.info(f"  Dropped LLM: {len(llm_df) - len(joined_df)}")
    logger.info(f"  Dropped DeBERTa: {len(deberta_df) - len(joined_df)}")
    
    # Create train/dev/test splits for trust scorer from the joined data
    # Use stratified split based on trust label (LLM correct or not)
    joined_df['trust_label_temp'] = (joined_df['llm_label'] == joined_df['gold_label']).astype(int)
    
    # Split: 70% train, 15% dev, 15% test
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        joined_df, test_size=0.3, random_state=42, stratify=joined_df['trust_label_temp']
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['trust_label_temp']
    )
    
    # Remove temporary column
    train_df = train_df.drop(columns=['trust_label_temp'])
    dev_df = dev_df.drop(columns=['trust_label_temp'])
    test_df = test_df.drop(columns=['trust_label_temp'])
    
    # Add split column
    train_df['split_trust'] = 'train'
    dev_df['split_trust'] = 'dev'
    test_df['split_trust'] = 'test'
    
    logger.info(f"Trust scorer splits - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    return train_df, dev_df, test_df


def extract_features(df: pd.DataFrame, dev_percentiles: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Extract features for trust score prediction.
    
    Args:
        df: DataFrame with joined LLM and DeBERTa data
        dev_percentiles: Pre-computed percentiles from dev set (for test set)
        
    Returns:
        Tuple of (DataFrame with features, percentiles dict)
    """
    features_df = df.copy()
    
    # LLM confidence feature (already numeric in data)
    # llm_confidence_numeric is already in the data (0.3, 0.6, 0.9)
    
    # DeBERTa features
    # Get probability columns
    prob_cols = [col for col in df.columns if col.startswith('prob_class')]
    probs = df[prob_cols].values
    
    # Max probability
    features_df['deberta_p_max'] = probs.max(axis=1)
    
    # Margin (top1 - top2)
    sorted_probs = np.sort(probs, axis=1)
    features_df['deberta_margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # Entropy
    features_df['deberta_entropy'] = np.array([entropy(p) for p in probs])
    
    # Disagreement feature
    features_df['llm_deberta_disagree'] = (features_df['llm_label'] != features_df['deberta_predicted_label']).astype(int)
    
    # Data-driven confidence clash features
    if dev_percentiles is None:
        # Compute percentiles on current data (dev set)
        p_max_25 = features_df['deberta_p_max'].quantile(0.25)
        p_max_75 = features_df['deberta_p_max'].quantile(0.75)
        percentiles = {'p_max_25': p_max_25, 'p_max_75': p_max_75}
    else:
        # Use pre-computed percentiles (for test set)
        p_max_25 = dev_percentiles['p_max_25']
        p_max_75 = dev_percentiles['p_max_75']
        percentiles = dev_percentiles
    
    # Confidence clash flags
    llm_high_conf = features_df['llm_confidence_numeric'] >= 0.9
    llm_low_conf = features_df['llm_confidence_numeric'] <= 0.3
    deberta_low_conf = features_df['deberta_p_max'] <= p_max_25
    deberta_high_conf = features_df['deberta_p_max'] >= p_max_75
    
    features_df['confidence_clash_high_low'] = (llm_high_conf & deberta_low_conf).astype(int)
    features_df['confidence_clash_low_high'] = (llm_low_conf & deberta_high_conf).astype(int)
    
    # Text features (lightweight)
    # Token count
    if 'text' in features_df.columns:
        features_df['token_count'] = features_df['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Punctuation density
        def punct_density(text):
            if pd.isna(text) or len(str(text)) == 0:
                return 0.0
            text_str = str(text)
            punct_count = len(re.findall(r'[^\w\s]', text_str))
            return punct_count / len(text_str)
        
        features_df['punctuation_density'] = features_df['text'].apply(punct_density)
        
        # Negation present (binary)
        negation_pattern = r'\b(no|not|never|nothing|none|nobody|nowhere|neither|nor|can\'t|cannot|won\'t|wouldn\'t|shouldn\'t|couldn\'t)\b'
        features_df['negation_present'] = features_df['text'].apply(
            lambda x: 1 if pd.notna(x) and re.search(negation_pattern, str(x).lower()) else 0
        )
    elif 'text_len' in features_df.columns:
        # Approximate token count from text length
        features_df['token_count'] = features_df['text_len'] / 5.0  # rough approximation
        features_df['punctuation_density'] = 0.0
        features_df['negation_present'] = 0
    else:
        features_df['token_count'] = 0
        features_df['punctuation_density'] = 0.0
        features_df['negation_present'] = 0
    
    # Target variable: trust label (1 if LLM correct, 0 if incorrect)
    features_df['trust_label'] = (features_df['llm_label'] == features_df['gold_label']).astype(int)
    
    return features_df, percentiles


class TrustScoreClassifier:
    """Classifier to predict trustworthiness of LLM annotations."""
    
    def __init__(self, dataset_name: str, model_type: str = "lr"):
        """
        Initialize trust score classifier.
        
        Args:
            dataset_name: Name of dataset (imdb, jigsaw, fever)
            model_type: Type of classifier ('lr' or 'rf')
        """
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.pipeline = None
        self.feature_columns = []
        self.continuous_features = []
        self.discrete_features = []
        self.dev_percentiles = None
        self.policy_thresholds = {}
        self.trained = False
        
        logger.info(f"Initialized {model_type} classifier for {dataset_name}")
    
    def _define_feature_columns(self):
        """Define which features to use for training."""
        self.feature_columns = [
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
        
        # Define continuous vs discrete features
        self.continuous_features = [
            'deberta_p_max',
            'deberta_margin',
            'deberta_entropy',
            'token_count',
            'punctuation_density'
        ]
        
        self.discrete_features = [
            'llm_confidence_numeric',
            'llm_deberta_disagree',
            'confidence_clash_high_low',
            'confidence_clash_low_high',
            'negation_present'
        ]
    
    def prepare_training_data(self, annotations_df: pd.DataFrame, gold_labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with features and trust labels.
        
        Args:
            annotations_df: DataFrame with LLM annotations
            gold_labels: Ground truth labels for trust calculation
            
        Returns:
            Tuple of (features, trust_labels)
            
        TODO: Implement training data preparation
        """
        # TODO: Implement training data preparation
        # 1. Extract features from annotations
        # 2. Calculate trust labels (LLM correct = 1, incorrect = 0)
        # 3. Handle missing values
        # 4. Split features and labels
        
        logger.info(f"TODO: Prepare training data for {len(annotations_df)} samples")
        
        # Extract features
        features_df = self.extract_features(annotations_df)
        
        # TODO: Calculate trust labels based on agreement with gold labels
        # trust_labels = (features_df['llm_label'] == gold_labels).astype(int)
        trust_labels = np.ones(len(annotations_df))  # Placeholder
        
        # Get feature matrix
        X = features_df[self.feature_columns].values
        
        return X, trust_labels
    
    def train(self, annotations_df: pd.DataFrame, gold_labels: pd.Series, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the trust score classifier.
        
        Args:
            annotations_df: Training annotations
            gold_labels: Ground truth labels
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics dictionary
            
        TODO: Implement model training
        """
        # TODO: Implement model training
        # 1. Prepare training data
        # 2. Split into train/validation
        # 3. Train classifier
        # 4. Validate performance
        # 5. Optimize threshold
        # 6. Return metrics
        
        logger.info(f"TODO: Train {self.model_type} trust classifier")
        
        # Prepare data
        X, y = self.prepare_training_data(annotations_df, gold_labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # TODO: Initialize and train classifier
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=42)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # TODO: Train model
        # self.model.fit(X_train, y_train)
        
        # TODO: Validate and optimize threshold
        # y_val_proba = self.model.predict_proba(X_val)[:, 1]
        # precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
        # self.threshold = self._optimize_threshold(y_val, y_val_proba)
        
        self.trained = True
        
        # Placeholder metrics
        metrics = {
            "train_accuracy": 0.85,
            "val_accuracy": 0.82,
            "val_auc": 0.88,
            "optimal_threshold": 0.65
        }
        
        logger.info(f"Training completed with metrics: {metrics}")
        return metrics
    
    def predict_trust_scores(self, annotations_df: pd.DataFrame) -> np.ndarray:
        """
        Predict trust scores for annotations.
        
        Args:
            annotations_df: DataFrame with annotations to score
            
        Returns:
            Array of trust scores (0-1)
            
        TODO: Implement trust score prediction
        """
        # TODO: Implement trust score prediction
        # 1. Extract features
        # 2. Predict probabilities
        # 3. Return trust scores
        
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        logger.info(f"TODO: Predict trust scores for {len(annotations_df)} annotations")
        
        # Extract features
        features_df = self.extract_features(annotations_df)
        X = features_df[self.feature_columns].values
        
        # TODO: Predict probabilities
        # trust_scores = self.model.predict_proba(X)[:, 1]
        trust_scores = np.random.random(len(annotations_df))  # Placeholder
        
        return trust_scores
    
    def predict_trust_decisions(self, annotations_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trust decisions (accept/review) for annotations.
        
        Args:
            annotations_df: DataFrame with annotations
            
        Returns:
            Tuple of (trust_scores, trust_decisions)
            
        TODO: Implement trust decision prediction
        """
        # TODO: Implement trust decision prediction
        # 1. Get trust scores
        # 2. Apply threshold
        # 3. Return decisions
        
        trust_scores = self.predict_trust_scores(annotations_df)
        trust_decisions = (trust_scores >= self.threshold).astype(int)
        
        return trust_scores, trust_decisions
    
    def _optimize_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Optimize threshold for trust decisions.
        
        Args:
            y_true: True trust labels
            y_scores: Predicted trust scores
            
        Returns:
            Optimal threshold value
            
        TODO: Implement threshold optimization
        """
        # TODO: Implement threshold optimization
        # 1. Calculate precision-recall curve
        # 2. Find optimal threshold (e.g., max F1-score)
        # 3. Return threshold
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        
        return thresholds[optimal_idx]
    
    def evaluate(self, annotations_df: pd.DataFrame, gold_labels: pd.Series) -> Dict[str, float]:
        """
        Evaluate trust classifier performance.
        
        Args:
            annotations_df: Test annotations
            gold_labels: Ground truth labels
            
        Returns:
            Evaluation metrics dictionary
            
        TODO: Implement evaluation
        """
        # TODO: Implement comprehensive evaluation
        # 1. Predict trust scores
        # 2. Calculate trust labels
        # 3. Compute metrics (accuracy, precision, recall, F1, AUC)
        # 4. Return metrics
        
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"TODO: Evaluate trust classifier on {len(annotations_df)} samples")
        
        # Predict trust scores
        trust_scores = self.predict_trust_scores(annotations_df)
        
        # Calculate true trust labels
        true_trust = (annotations_df['llm_label'] == gold_labels).astype(int)
        
        # TODO: Calculate comprehensive metrics
        # accuracy = accuracy_score(true_trust, (trust_scores >= self.threshold).astype(int))
        # auc = roc_auc_score(true_trust, trust_scores)
        # precision, recall, f1, _ = precision_recall_fscore_support(true_trust, (trust_scores >= self.threshold).astype(int))
        
        # Placeholder metrics
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "auc": 0.90
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
            
        TODO: Implement model saving
        """
        # TODO: Implement model saving
        # 1. Save classifier
        # 2. Save feature columns
        # 3. Save threshold
        # 4. Save metadata
        
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'trained': self.trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"TODO: Save model to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
            
        TODO: Implement model loading
        """
        # TODO: Implement model loading
        # 1. Load model data
        # 2. Restore classifier
        # 3. Restore metadata
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.task_type = model_data['task_type']
        self.feature_columns = model_data['feature_columns']
        self.threshold = model_data['threshold']
        self.trained = model_data['trained']
        
        logger.info(f"TODO: Load model from {filepath}")


def create_trust_scorer(model_type: str = "logistic_regression", task_type: str = "sentiment") -> TrustScoreClassifier:
    """
    Factory function to create trust score classifier.
    
    Args:
        model_type: Type of classifier
        task_type: Type of annotation task
        
    Returns:
        TrustScoreClassifier instance
        
    TODO: Implement factory function
    """
    # TODO: Implement factory function
    # 1. Create TrustScoreClassifier
    # 2. Initialize with correct parameters
    # 3. Return ready-to-use classifier
    
    classifier = TrustScoreClassifier(model_type, task_type)
    return classifier


if __name__ == "__main__":
    # Test the trust scorer
    print("Testing TrustScoreClassifier...")
    
    # Create test classifier
    trust_scorer = create_trust_scorer("logistic_regression", "sentiment")
    
    # Test feature extraction (placeholder)
    print("TODO: Implement trust scorer training and evaluation")
    
    print("TODO: Implement actual trust scoring logic")
