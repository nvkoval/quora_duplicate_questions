# Quora Question Pairs Duplicate Detection

A comprehensive machine learning project for identifying duplicate question pairs using the [Quora Question Pairs dataset](https://www.kaggle.com/competitions/quora-question-pairs) from Kaggle. The goal is to improve search and recommendation systems by detecting duplicate questions.This project explores multiple approaches — from traditional ML to transformer models with a focus on BERT fine-tuning and interpretability.


## Project Overview
This is a **binary classification** task: determine whether two given questions are duplicates. Applications include search engines, Q&A platforms, and content deduplication systems.

## 📊 Results Summary

| Model | Train Log Loss | Validation Log Loss |
|-------|----------------|-------------------|
| **BERT Fine-Tuning** | **0.22866** | **0.27312** |
| SentenceTransformer with Cosine Similarity | 0.44307 | 0.44029 |
| Logistic Regression (TF-IDF)| 0.45201 | 0.46431 |
| Random Forest (TF-IDF)| 0.46324 | 0.48274 |
| LGBMClassifier (TF-IDF)| 0.47579 | 0.48881 |
| Logistic Regression with TF-IDF on matching words | 0.51774 | 0.53856 |
| XGBClassifier (TF-IDF)| 0.53803 | 0.53968 |
| GloVe Embeddings with Logistic Regression | 0.56615 | 0.56536 |
| **Baseline: DummyClassifier (uniform)** | **0.69315** | **0.69315** |

**🏆 Best Performance**: BERT Fine-Tuning achieved the lowest validation log loss of 0.27312, significantly outperforming all other methods.

## Project Structure

```
.
├── data/
│   ├── processed/                    # Processed datasets and results
│   └── raw/                          # Original dataset
├── models/
│   └── bert_quora_model/             # Fine-tuned BERT model (not included due to size)
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Data preprocessing and cleaning
│   ├── 03_baseline.ipynb             # Simple baseline models
│   ├── 04_traditional_ML.ipynb       # Traditional ML approaches
│   ├── 05_embeddings_with_cosine_similarity.ipynb
│   ├── 06_embeddings_approach.ipynb  # Advanced embedding techniques
│   ├── 07_bert_finetuning.ipynb      # BERT model fine-tuning
│   ├── 08_bert_interpreting.ipynb    # Interpret BERT predictions using Captum
│   └── 09_model_comparison.ipynb     # Final model comparison
├── src/
│   └── utils.py                      # Utility functions for the project
├── requirements.txt
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis
- Dataset overview and statistics
- Question length distributions
- Duplicate ratio analysis
- Text similarity patterns

### 2. Data Preprocessing
- Text cleaning, stopword removal, lemmatization
- Feature engineering for traditional ML

### 3. Modeling Approaches

#### Traditional Machine Learning (TF-IDF Features)
- Logistic Regression, Random Forest, XGBoost, LightGBM
- Additional experiment: TF-IDF on **matching words only**

#### Embedding-Based Approaches
- **GloVe Embeddings** with logistic regression
- **SentenceTransformer**: semantic similarity with cosine distance

#### Deep Learning
- **BERT Fine-Tuning**: Transformer model fine-tuned on the task

## Key Findings

1. **BERT dominates**: Fine-tuned BERT significantly outperforms all other approaches
2. **SentenceTransformer** provides a great balance between performance and simplicity
3. **Logistic Regression** serves as a solid baseline
4. **Clear performance hierarchy**: Deep learning > Sentence embeddings > TF-IDF + traditional ML > Word embeddings > Random baseline

## 📈 Performance Metrics

The models are evaluated using **log loss**, which is particularly suitable for:
- Binary classification problems
- Penalizing incorrect confident predictions
- Measuring probability calibration

Lower log loss indicates better performance, with perfect predictions achieving 0.

## 📦 Pretrained BERT Model Download
Due to GitHub file size limits, the fine-tuned BERT model is **stored externally on Google Drive**.

**To use locally:**
1. Download model from shared [link](https://drive.google.com/file/d/1LMdECszFOCzrs6AbSnBwS2lzwTwOG8Y9/view?usp=drive_link)
2. Place it in `models/bert_quora_model/`

## Interpreting BERT with Captum

In `08_Interpreting_BERT.ipynb`, we use the [Captum](https://captum.ai/) library to interpret BERT model predictions. Using **Integrated Gradients**, we compute **token-level attributions** that show how much each token contributed to the model’s decision.

**Key features**:
- Attribution visualization with `captum.attr.visualize_text()`
- Randomly selected test examples
- Side-by-side comparison of question pairs and predicted

## 🔍 Next Steps

- Improve preprocessing and normalization
- Experiment with custom neural networks (e.g., LSTM, Siamese BiLSTM)
- Try other transformer architectures (RoBERTa, DeBERTa)
- Add cross-validation for more robust evaluation
- ✅ **[Done]** Deploy model as a REST API with FastAPI and Docker

## 🌍 Deployed API (FastAPI + Docker)

We built a lightweight **REST API** for duplicate question detection using the fine-tuned BERT model.

**Live Demo:**
[https://api-duplicate-questions.onrender.com](https://api-duplicate-questions.onrender.com/docs)

### Features
- FastAPI + Pydantic-based RESTful backend
- Loads BERT model from Google Drive on first run
- Accepts question pairs and returns:
  - Binary classification (`duplicate`/`not duplicate`)
  - Model confidence score

### Example request:
```json
POST /predict/
{
  "question1": "Why is beef banned in India and not pork as well?",
  "question2": "Is beef banned in India?"
}
```

### Example response:
```json
{
  "class_name": "not duplicate",
  "confidence": 0.9058
}
```

*For detailed implementation and analysis, please refer to the individual notebooks in the `notebooks/` directory.*
