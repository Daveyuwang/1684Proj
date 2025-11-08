# Datasets

This directory contains three benchmark datasets for annotation reliability analysis.

---

## IMDb

**Task**: Sentiment analysis (binary classification)  
**Classes**: Positive, Negative  
**Source**: [Hugging Face - stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

**Dataset Split**:
- Training: 25,000 samples
- Development: 12,500 samples
- Test: 12,500 samples
- Total: 50,000 samples

---

## Jigsaw Toxic Comment Classification

**Task**: Toxicity detection (binary classification)  
**Classes**: Toxic / Non-toxic  
**Source**: [Kaggle - Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)

**Note**: This project uses only the binary toxicity labels (toxic/non-toxic). Identity labels available in the original dataset are not used.

**Dataset Split**:
- Full dataset (used for DeBERTa training):
  - Training: 1,804,874 samples
  - Development: 97,320 samples
  - Test: 97,320 samples
  - Total: 1,999,514 samples
- Stratified 20k sample (used for LLM annotations): 20,000 samples sampled from test set

**Sample File**: `jigsaw/jigsaw_sample20k.csv` (stratified subset for LLM annotations)
---

## FEVER

**Task**: Fact verification (3-class classification)  
**Classes**: SUPPORTS, REFUTES, NOT ENOUGH INFO  
**Source**: [FEVER Dataset](https://fever.ai/dataset/fever.html)

**Files Used**:
- Training Dataset: 145,449 samples
- Paper Development Dataset: 9,999 samples
- Paper Test Dataset: 9,999 samples

**Format**: JSONL files (`train.jsonl`, `paper_dev.jsonl`, `paper_test.jsonl`)

**Data Structure**:
- `id`: Claim identifier
- `label`: One of `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`
- `claim`: Text of the claim to verify
- `evidence`: Supporting evidence sentences (when applicable)

---

## Dataset Statistics

| Dataset | Task | Classes | Train | Dev | Test | Total |
|---------|------|---------|-------|-----|------|-------|
| IMDb | Sentiment | 2 | 25,000 | 12,500 | 12,500 | 50,000 |
| Jigsaw | Toxicity | 2 | 1,804,874 | 97,320 | 97,320 | 1,999,514 |
| FEVER | Fact Check | 3 | 145,449 | 9,999 | 9,999 | 165,447 |

**Notes on Usage**:
- **IMDb**: Full dataset used for both DeBERTa training and LLM annotations
- **Jigsaw**: Full 2M dataset for DeBERTa training; 20k stratified sample from test set for LLM annotations
- **FEVER**: Full 145k training set for DeBERTa; paper dev/test datasets (9,999 each) for LLM annotations

---

## Notes

**Label Mappings**:

*IMDb*:
- `negative` → 0
- `positive` → 1

*Jigsaw*:
- `non-toxic` → 0
- `toxic` → 1

*FEVER*:
- `SUPPORTS` → 0
- `REFUTES` → 1
- `NOT ENOUGH INFO` → 2


