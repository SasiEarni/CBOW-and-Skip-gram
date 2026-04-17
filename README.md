# Analysis of CBOW and Skip-gram Embeddings with Dimensionality Reduction for Text Classification

##  Project Overview

This project focuses on analysing the performance of Word2Vec embedding models (CBOW and Skip-gram) for text classification, and evaluating the impact of dimensionality reduction using Principal Component Analysis (PCA).

The study compares:

* Traditional Machine Learning (TF-IDF + Logistic Regression)
* Deep Learning (LSTM)
* Word Embeddings (CBOW and Skip-gram)
* With and without PCA (300 → 50 dimensions)

---

##  Objectives

* Generate word embeddings using Word2Vec (CBOW & Skip-gram)
* Apply PCA to reduce embedding dimensionality
* Evaluate the impact on classification performance
* Compare results with a baseline model (TF-IDF + Logistic Regression)

---

##  Methodology

### Pipeline:

```id="pipe01"
Text Data  
↓  
Preprocessing (cleaning, tokenization, lemmatization)  
↓  
Word2Vec (CBOW / Skip-gram)  
↓  
PCA (optional: 300D → 50D)  
↓  
Embedding Matrix  
↓  
LSTM Model  
↓  
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
```

---

##  Models Used

### 1. Baseline Model

* TF-IDF Vectorizer
* Logistic Regression

### 2. Deep Learning Models

* LSTM with CBOW embeddings (300D)
* LSTM with CBOW + PCA (50D)
* LSTM with Skip-gram embeddings (300D)
* LSTM with Skip-gram + PCA (50D)

---

##  Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* ROC-AUC Curve

---

##  Dimensionality Reduction Analysis

* Explained Variance Ratio
* Cosine Similarity Preservation
* Top-K Nearest Neighbour Overlap

---

##  Dataset

* Fake.csv
* True.csv

Each dataset contains:

* Title
* Text
* Label (Fake / Real)

---

##  Prerequisites

Make sure you have Python installed (version 3.8 or above).

### Install Jupyter Notebook

If Jupyter Notebook is not installed, you can install it using:

pip install notebook


Alternatively, you can install Anaconda, which includes Jupyter Notebook:
https://www.anaconda.com/

---

### Verify Installation

Run the following command to start Jupyter:

jupyter notebook

-----

##  How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Download NLTK resources

Open Python or a Jupyter Notebook and run the following:

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

### 3. Run the project

jupyter notebook CBOW_vs_Skipgram_Text_Classification.ipynb
---

##  Key Findings

* LSTM outperforms Logistic Regression in classification tasks
* PCA reduces dimensionality with a small loss in accuracy
* CBOW embeddings are more robust to PCA than Skip-gram

---

##  Notes

* PCA is applied to Word2Vec embeddings before feeding them into the LSTM
* Embedding dimension reduced from 300 to 50
* Early stopping is used to prevent overfitting

---

##  Author

Sasi Earni

---

##  Future Work

* Hyperparameter tuning for LSTM
* Use of pre-trained embeddings (e.g., GloVe, BERT)
* Exploring other dimensionality reduction techniques
