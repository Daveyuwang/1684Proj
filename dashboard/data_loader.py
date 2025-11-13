"""
Data loading module for dashboard.

Loads and preprocesses annotation data, metrics, trust scores, and DeBERTa predictions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import config


class DashboardDataLoader:
    """Load and preprocess data for dashboard visualization."""
    
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        self.data_cache = {}
        
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets with LLM annotations."""
        llm_dir = self.results_dir / "llm_annotations_7b"
        if not llm_dir.exists():
            return []
        
        datasets = []
        for item in llm_dir.iterdir():
            if item.is_dir() and (item / f"{item.name}_llm_annotations.json").exists():
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def load_llm_annotations(self, dataset_name: str) -> pd.DataFrame:
        """Load LLM annotations for a dataset."""
        cache_key = f"{dataset_name}_annotations"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        annotations_path = (
            self.results_dir / "llm_annotations_7b" / dataset_name / 
            f"{dataset_name}_llm_annotations.json"
        )
        
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Add agreement column
        df['is_correct'] = (df['llm_label'] == df['gold_label']).astype(int)
        df['agreement_status'] = df['is_correct'].map({1: 'correct', 0: 'incorrect'})
        
        # Map confidence to readable labels
        confidence_map = {0.3: 'low', 0.6: 'medium', 0.9: 'high'}
        df['confidence_level'] = df['llm_confidence_numeric'].map(confidence_map)
        
        self.data_cache[cache_key] = df
        return df
    
    def load_agreement_metrics(self, dataset_name: str) -> Dict[str, Any]:
        """Load agreement metrics for a dataset."""
        cache_key = f"{dataset_name}_metrics"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        metrics_path = (
            self.results_dir / "llm_annotations_7b" / dataset_name / 
            f"{dataset_name}_agreement_metrics.json"
        )
        
        if not metrics_path.exists():
            return {}
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        self.data_cache[cache_key] = metrics
        return metrics
    
    def load_hard_cases(self, dataset_name: str) -> pd.DataFrame:
        """Load hard cases for a dataset."""
        cache_key = f"{dataset_name}_hard_cases"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        hard_cases_path = (
            self.results_dir / "llm_annotations_7b" / dataset_name / 
            f"{dataset_name}_hard_cases.json"
        )
        
        if not hard_cases_path.exists():
            return pd.DataFrame()
        
        with open(hard_cases_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        self.data_cache[cache_key] = df
        return df
    
    def load_trust_scores(self, dataset_name: str, model_type: str = "lr") -> pd.DataFrame:
        """Load trust score predictions for a dataset."""
        cache_key = f"{dataset_name}_trust_{model_type}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        trust_path = (
            self.results_dir / "trust_scorer" / 
            f"{dataset_name}_{model_type}_predictions.csv"
        )
        
        if not trust_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(trust_path)
        self.data_cache[cache_key] = df
        return df
    
    def load_trust_metrics(self, dataset_name: str, model_type: str = "lr") -> Dict[str, Any]:
        """Load trust score metrics for a dataset."""
        cache_key = f"{dataset_name}_trust_metrics_{model_type}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        metrics_path = (
            self.results_dir / "trust_scorer" / 
            f"{dataset_name}_{model_type}_metrics.json"
        )
        
        if not metrics_path.exists():
            return {}
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        self.data_cache[cache_key] = metrics
        return metrics
    
    def load_trust_config(self, dataset_name: str) -> Dict[str, Any]:
        """Load trust scorer configuration including policy thresholds."""
        cache_key = f"{dataset_name}_trust_config"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        config_path = (
            self.results_dir / "trust_scorer" / "final_export" / 
            f"{dataset_name}_config.json"
        )
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        self.data_cache[cache_key] = config_data
        return config_data
    
    def load_deberta_predictions(self, dataset_name: str) -> pd.DataFrame:
        """Load DeBERTa predictions for a dataset."""
        cache_key = f"{dataset_name}_deberta"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        deberta_path = (
            self.results_dir / "deberta_predictions" / 
            f"{dataset_name}_deberta_predictions.json"
        )
        
        if not deberta_path.exists():
            return pd.DataFrame()
        
        with open(deberta_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        self.data_cache[cache_key] = df
        return df
    
    def get_joined_data(self, dataset_name: str, include_trust: bool = True, 
                       model_type: str = "lr") -> pd.DataFrame:
        """Get joined data with LLM annotations, DeBERTa predictions, and trust scores."""
        # Load LLM annotations
        llm_df = self.load_llm_annotations(dataset_name)
        
        # Load DeBERTa predictions
        deberta_df = self.load_deberta_predictions(dataset_name)
        
        if not deberta_df.empty:
            # Merge on example_id or row_id
            merge_col = 'example_id' if 'example_id' in deberta_df.columns else 'row_id'
            if merge_col in llm_df.columns and merge_col in deberta_df.columns:
                llm_df = llm_df.merge(
                    deberta_df, 
                    on=merge_col, 
                    how='left',
                    suffixes=('', '_deberta')
                )
        
        # Load trust scores if requested
        if include_trust:
            trust_df = self.load_trust_scores(dataset_name, model_type)
            if not trust_df.empty:
                merge_col = 'example_id' if 'example_id' in trust_df.columns else 'row_id'
                if merge_col in llm_df.columns and merge_col in trust_df.columns:
                    llm_df = llm_df.merge(
                        trust_df,
                        on=merge_col,
                        how='left',
                        suffixes=('', '_trust')
                    )
        
        return llm_df
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {
            'name': dataset_name,
            'display_name': config.DATASETS.get(dataset_name, {}).get('name', dataset_name),
            'task': config.DATASETS.get(dataset_name, {}).get('task', 'unknown'),
            'classes': config.DATASETS.get(dataset_name, {}).get('classes', []),
        }
        
        # Load metrics
        metrics = self.load_agreement_metrics(dataset_name)
        if metrics:
            info['metrics'] = {
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0),
                'cohen_kappa': metrics.get('cohen_kappa', 0),
                'average_confidence': metrics.get('average_confidence', 0),
                'num_samples': metrics.get('num_samples', 0),
            }
        
        # Load annotations to get sample count
        annotations_df = self.load_llm_annotations(dataset_name)
        info['num_samples'] = len(annotations_df)
        
        return info
    
    def clear_cache(self):
        """Clear the data cache."""
        self.data_cache.clear()


# Global instance
_data_loader = None

def get_data_loader() -> DashboardDataLoader:
    """Get or create the global data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader()
    return _data_loader

