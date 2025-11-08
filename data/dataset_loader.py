"""
Dataset loaders for IMDb, Jigsaw, and CrisisBench datasets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import requests
import zipfile
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
import config

class DatasetLoader:
    """Base class for dataset loading and preprocessing."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.config = config.DATASETS[dataset_name]
        self.data_dir = config.DATA_DIR / dataset_name
        self.data_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and split dataset into train/dev/test."""
        raise NotImplementedError
    
    def _load_from_local_csv(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Try to load dataset from local CSV files first."""
        train_path = self.data_dir / "train.csv"
        dev_path = self.data_dir / "dev.csv"
        test_path = self.data_dir / "test.csv"
        
        # Check if all required files exist
        if train_path.exists() and dev_path.exists() and test_path.exists():
            try:
                print(f"Loading {self.dataset_name} from local CSV files...")
                train_df = pd.read_csv(train_path)
                dev_df = pd.read_csv(dev_path)
                test_df = pd.read_csv(test_path)
                
                # Validate required columns
                for split_name, df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
                    if 'text' not in df.columns or 'label' not in df.columns:
                        print(f"Warning: Local CSV missing required columns for {split_name}, will load from HuggingFace")
                        return None
                
                print(f"Successfully loaded from local files: {self.data_dir}")
                return {
                    'train': train_df,
                    'dev': dev_df,
                    'test': test_df
                }
            except Exception as e:
                print(f"Warning: Failed to load from local CSV: {e}")
                print("Will attempt to load from HuggingFace instead...")
                return None
        
        return None
    
    def _create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create stratified train/dev/test splits."""
        # Stratified split to maintain class distribution
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            stratify=df['label'], 
            random_state=config.EVALUATION['random_seed']
        )
        
        dev_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df['label'], 
            random_state=config.EVALUATION['random_seed']
        )
        
        return {
            'train': train_df.reset_index(drop=True),
            'dev': dev_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }

class IMDBLoader(DatasetLoader):
    """IMDb sentiment analysis dataset loader."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load IMDb dataset from local CSV or HuggingFace (stanfordnlp/imdb)."""
        # Try loading from local CSV first
        local_data = self._load_from_local_csv()
        if local_data is not None:
            return local_data
        
        # Otherwise load from HuggingFace
        try:
            if not HAS_DATASETS:
                print("datasets library not available, using sample data")
                return self._create_sample_data()

            print("Loading IMDb from HuggingFace...")
            # Use the stanfordnlp variant as requested
            dataset = load_dataset("stanfordnlp/imdb")

            # Convert to pandas DataFrames for train/test; create dev from test
            splits: Dict[str, pd.DataFrame] = {}
            for split_name in ['train', 'test']:
                if split_name in dataset:
                    df = dataset[split_name].to_pandas()
                    # Normalize expected columns
                    if 'text' not in df.columns and 'review' in df.columns:
                        df = df.rename(columns={'review': 'text'})
                    df = df.rename(columns={'label': 'label'})
                    df = df[['text', 'label']].copy()
                    splits[split_name] = df

            # Create dev split from test (50/50 of provided test split)
            dev_df, test_df = train_test_split(
                splits['test'],
                test_size=0.5,
                stratify=splits['test']['label'],
                random_state=config.EVALUATION['random_seed']
            )

            splits['dev'] = dev_df.reset_index(drop=True)
            splits['test'] = test_df.reset_index(drop=True)

            return splits

        except Exception as e:
            print(f"Error loading IMDb dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample data if dataset loading fails."""
        sample_texts = [
            "This movie was absolutely fantastic!",
            "Terrible acting and poor plot.",
            "Great cinematography and direction.",
            "Boring and predictable storyline.",
            "Amazing performances by all actors."
        ]
        sample_labels = [1, 0, 1, 0, 1]
        
        df = pd.DataFrame({
            'text': sample_texts * 100,  # Repeat for more data
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

class JigsawLoader(DatasetLoader):
    """Jigsaw toxicity classification dataset loader."""
    
    def _load_from_sampled_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Try to load from stratified sample if available."""
        # Check for sampled data in results directory
        results_dir = Path("results/llm_annotations_7b/jigsaw/jigsaw_sample")
        
        # Try 20k sample first, then fall back to 10k
        sample_path = results_dir / "jigsaw_sample20k.csv"
        if not sample_path.exists():
            sample_path = results_dir / "jigsaw_sample10k.csv"
        
        if sample_path.exists():
            try:
                print(f"Loading Jigsaw from stratified sample: {sample_path}")
                sample_df = pd.read_csv(sample_path)
                
                # Validate required columns
                if 'text' not in sample_df.columns or 'label' not in sample_df.columns:
                    print("Warning: Sampled data missing required columns")
                    return None
                
                # Use the ENTIRE sample as test set for LLM annotation and DeBERTa prediction
                # Create minimal dummy train/dev splits for compatibility
                # (Trust scorer will create its own splits from the joined data)
                n_samples = len(sample_df)
                
                # Create minimal dummy train/dev (just 1 sample each for compatibility)
                train_df = sample_df.iloc[:1].copy()  # Minimal train
                dev_df = sample_df.iloc[:1].copy()    # Minimal dev  
                test_df = sample_df.copy()            # FULL sample as test
                
                # Use original_index as the actual index if available
                if 'original_index' in train_df.columns:
                    train_df = train_df.set_index('original_index')
                    dev_df = dev_df.set_index('original_index')
                    test_df = test_df.set_index('original_index')
                else:
                    train_df = train_df.reset_index(drop=True)
                    dev_df = dev_df.reset_index(drop=True)
                    test_df = test_df.reset_index(drop=True)
                
                return {
                    'train': train_df,
                    'dev': dev_df,
                    'test': test_df
                }
                
            except Exception as e:
                print(f"Error loading sampled data: {e}")
                return None
        
        return None
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load Jigsaw dataset from local CSV or HuggingFace (civil_comments)."""
        # Try loading from sampled data first (if exists)
        sampled_data = self._load_from_sampled_data()
        if sampled_data is not None:
            return sampled_data
            
        # Try loading from local CSV
        local_data = self._load_from_local_csv()
        if local_data is not None:
            return local_data
        
        # Otherwise load from HuggingFace
        try:
            if not HAS_DATASETS:
                print("datasets library not available, using sample data")
                return self._create_sample_data()

            print("Loading Jigsaw from HuggingFace...")
            # Use civil_comments dataset as primary source (more reliable than affahrizain version)
            dataset = load_dataset("civil_comments")

            def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
                # Determine text column
                text_col = None
                for candidate in ['text', 'comment_text', 'comment']:
                    if candidate in df.columns:
                        text_col = candidate
                        break
                if text_col is None:
                    # If no obvious text column, create from first string column
                    string_cols = [c for c in df.columns if df[c].dtype == object]
                    text_col = string_cols[0] if string_cols else df.columns[0]

                # Determine label column(s)
                label: Optional[pd.Series] = None
                if 'label' in df.columns:
                    label = df['label']
                elif 'toxic' in df.columns:
                    label = (df['toxic'] > 0).astype(int)
                else:
                    # Multi-label Jigsaw columns
                    multi_cols = [
                        c for c in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                        if c in df.columns
                    ]
                    if multi_cols:
                        label = (df[multi_cols].sum(axis=1) > 0).astype(int)
                    elif 'target' in df.columns:
                        # Unintended Bias in Toxicity dataset style
                        label = (df['target'] >= 0.5).astype(int)
                    else:
                        raise ValueError("Could not determine label column for Jigsaw dataset")

                out = pd.DataFrame({'text': df[text_col], 'label': label})
                return out

            splits: Dict[str, pd.DataFrame] = {}
            for split_name in dataset.keys():
                df = dataset[split_name].to_pandas()
                splits[split_name] = normalize_df(df)

            # Ensure we have train/dev/test
            result: Dict[str, pd.DataFrame] = {}
            if 'train' in splits:
                result['train'] = splits['train']
                if 'validation' in splits:
                    result['dev'] = splits['validation'].reset_index(drop=True)
                else:
                    # Create dev from train (15% of train)
                    train_df, dev_df = train_test_split(
                        result['train'],
                        test_size=0.15,
                        stratify=result['train']['label'],
                        random_state=config.EVALUATION['random_seed']
                    )
                    result['train'] = train_df.reset_index(drop=True)
                    result['dev'] = dev_df.reset_index(drop=True)
            else:
                # Single split present
                return self._create_splits(next(iter(splits.values())))

            if 'test' in splits:
                result['test'] = splits['test'].reset_index(drop=True)
            else:
                # If no test provided, create from dev
                dev_df, test_df = train_test_split(
                    result['dev'],
                    test_size=0.5,
                    stratify=result['dev']['label'],
                    random_state=config.EVALUATION['random_seed']
                )
                result['dev'] = dev_df.reset_index(drop=True)
                result['test'] = test_df.reset_index(drop=True)

            return result
        except Exception as e:
            print(f"Error loading Jigsaw dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample toxicity data."""
        sample_texts = [
            "This is a normal comment.",
            "You are an idiot!",
            "Great work on this project.",
            "I hate this so much.",
            "Thanks for sharing this information."
        ]
        sample_labels = [0, 1, 0, 1, 0]  # 0: non-toxic, 1: toxic
        
        df = pd.DataFrame({
            'text': sample_texts * 100,
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

class CrisisBenchLoader(DatasetLoader):
    """CrisisBench English dataset loader with subsets."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load CrisisBench-english with 'humanitarian' or 'informativeness' subset."""
        # Try loading from local CSV first
        local_data = self._load_from_local_csv()
        if local_data is not None:
            return local_data
        
        # Otherwise load from HuggingFace
        try:
            if not HAS_DATASETS:
                print("datasets library not available, using sample data")
                return self._create_sample_data()

            # Determine subset from dataset name
            if 'humanitarian' in self.dataset_name:
                subset = 'humanitarian'
            elif 'informativeness' in self.dataset_name:
                subset = 'informativeness'
            else:
                # default to humanitarian if not specified
                subset = 'humanitarian'

            print(f"Loading CrisisBench ({subset}) from HuggingFace...")
            dataset = load_dataset("QCRI/CrisisBench-english", subset)

            def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
                # Text field can be 'tweet_text' or 'text'
                text_col = 'text' if 'text' in df.columns else (
                    'tweet_text' if 'tweet_text' in df.columns else None
                )
                if text_col is None:
                    # Fall back to first object column
                    string_cols = [c for c in df.columns if df[c].dtype == object]
                    text_col = string_cols[0] if string_cols else df.columns[0]

                # Label field may be 'label', 'labels', or 'class_label'
                if 'label' in df.columns:
                    label_series = df['label']
                elif 'labels' in df.columns:
                    label_series = df['labels']
                elif 'class_label' in df.columns:
                    label_series = df['class_label']
                else:
                    raise ValueError("Could not find label column in CrisisBench dataset")

                # If labels are strings, factorize to integers
                if label_series.dtype == object:
                    label_series, _ = pd.factorize(label_series)

                out = pd.DataFrame({'text': df[text_col], 'label': label_series.astype(int)})
                return out

            splits: Dict[str, pd.DataFrame] = {}
            for split_name in dataset.keys():
                df = dataset[split_name].to_pandas()
                splits[split_name] = normalize_df(df)

            result: Dict[str, pd.DataFrame] = {}
            # Prefer provided splits; otherwise create
            if 'train' in splits:
                result['train'] = splits['train'].reset_index(drop=True)
                if 'validation' in splits:
                    result['dev'] = splits['validation'].reset_index(drop=True)
                else:
                    train_df, dev_df = train_test_split(
                        result['train'],
                        test_size=0.15,
                        stratify=result['train']['label'],
                        random_state=config.EVALUATION['random_seed']
                    )
                    result['train'] = train_df.reset_index(drop=True)
                    result['dev'] = dev_df.reset_index(drop=True)
            else:
                # If no train split, synthesize from any available split
                return self._create_splits(next(iter(splits.values())))

            if 'test' in splits:
                result['test'] = splits['test'].reset_index(drop=True)
            else:
                dev_df, test_df = train_test_split(
                    result['dev'],
                    test_size=0.5,
                    stratify=result['dev']['label'],
                    random_state=config.EVALUATION['random_seed']
                )
                result['dev'] = dev_df.reset_index(drop=True)
                result['test'] = test_df.reset_index(drop=True)

            return result
        except Exception as e:
            print(f"Error loading CrisisBench dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample crisis data."""
        sample_texts = [
            "Just had lunch with friends.",
            "Earthquake hit our city, need help!",
            "Beautiful sunset today.",
            "Flooding in downtown area, stay safe!",
            "Coffee with colleagues this morning."
        ]
        sample_labels = [0, 1, 0, 1, 0]  # 0: not crisis, 1: crisis
        
        df = pd.DataFrame({
            'text': sample_texts * 100,
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

class FEVERLoader(DatasetLoader):
    """FEVER fact verification dataset loader."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load FEVER dataset from local JSONL files or create sample data."""
        # Try loading from local JSONL files first
        local_data = self._load_from_local_jsonl()
        if local_data is not None:
            return local_data
        
        # Fallback to sample data
        print("FEVER data not found locally, using sample data")
        return self._create_sample_data()
    
    def _load_from_local_jsonl(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load FEVER from local JSONL files."""
        import json
        from pathlib import Path
        
        # self.data_dir is already config.DATA_DIR / "fever"
        train_file = self.data_dir / "train.jsonl"
        dev_file = self.data_dir / "paper_dev.jsonl"
        test_file = self.data_dir / "paper_test.jsonl"
        
        if not all([train_file.exists(), dev_file.exists(), test_file.exists()]):
            return None
        
        print(f"Loading FEVER from local JSONL files...")
        
        # Label mapping: SUPPORTS -> 2, REFUTES -> 0, NOT ENOUGH INFO -> 1
        label_map = {
            "REFUTES": 0,
            "NOT ENOUGH INFO": 1,
            "SUPPORTS": 2
        }
        
        def load_jsonl(file_path: Path) -> pd.DataFrame:
            """Load JSONL file and extract claim and label."""
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Use the 'claim' as text
                    text = item.get('claim', '')
                    # Map label to integer
                    label_str = item.get('label', '')
                    if label_str in label_map:
                        label = label_map[label_str]
                        data.append({'text': text, 'label': label})
            return pd.DataFrame(data)
        
        train_df = load_jsonl(train_file)
        dev_df = load_jsonl(dev_file)
        test_df = load_jsonl(test_file)
        
        print(f"Successfully loaded FEVER: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")
        
        return {
            'train': train_df,
            'dev': dev_df,
            'test': test_df
        }
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample fact verification data."""
        sample_texts = [
            "The sky is blue.",
            "Cats can fly.",
            "Water boils at 100°C.",
            "The moon is made of cheese.",
            "Paris is the capital of France."
        ]
        sample_labels = [2, 0, 2, 0, 2]  # 2: supports, 0: refutes, 1: not enough info
        
        df = pd.DataFrame({
            'text': sample_texts * 100,
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

def load_dataset_by_name(dataset_name: str) -> Dict[str, pd.DataFrame]:
    """Factory function to load dataset by name."""
    loaders = {
        'imdb': IMDBLoader,
        'jigsaw': JigsawLoader,
        'crisisbench': CrisisBenchLoader,  # backward compatibility if used
        'crisisbench_humanitarian': CrisisBenchLoader,
        'crisisbench_informativeness': CrisisBenchLoader,
        'fever': FEVERLoader
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = loaders[dataset_name](dataset_name)
    return loader.load_data()

if __name__ == "__main__":
    # Test loading all datasets
    for dataset_name in config.DATASETS.keys():
        print(f"Loading {dataset_name}...")
        data = load_dataset_by_name(dataset_name)
        print(f"Loaded {dataset_name}: {[(k, len(v)) for k, v in data.items()]}")
