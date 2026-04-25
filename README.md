# Threshold-Tuned Gradient Boosting for Fetal Health Classification

> Achieving **100% Pathological Recall** on Cardiotocography Data — Zero Missed Sick Cases

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Authors

| Name | Roll Number | Email |
|------|-------------|-------|
| Deepanshu | 2210990253 | deepanshu253.be22@chitkara.edu.in |
| Harshit Kukreja | 2210990387 | harshit387.be22@chitkara.edu.in |
| Arsh Sudan | 2210990167 | arsh167.be22@chitkara.edu.in |
| Inder Goswami | 2210991677 | inder1677.be22@chitkara.edu.in |

**Supervisor:** Rajat Takkar, Chitkara University, Punjab

---

## Problem Statement

Classifying fetal health from Cardiotocography (CTG) recordings is critical for early detection of fetal distress. Manual CTG interpretation varies widely between clinicians, and delays can prove fatal. This project develops a machine learning pipeline that classifies CTG records into **Normal**, **Suspect**, or **Pathological** classes with a focus on **never missing a truly sick baby**.

## Key Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **96.22%** |
| Pathological Recall | **100.00%** (zero missed cases) |
| Suspect Recall | **81.03%** |
| Pathological Precision | **92.11%** (only 3 false alarms) |

## Approach

### Two-Stage Optimization

```
CTG Data (2,113 records) → Dedup & EDA → 80/20 Stratified Split
                                              ↓
                          ┌─── Stage 1: Sample Weighting (1:5:10) ───┐
                          │    Penalize Pathological errors 10×       │
                          └─── Stage 2: Threshold Tuning ────────────┘
                                 P(Patho) ≥ 0.10 | P(Suspect) ≥ 0.30
                                              ↓
                              96.22% Accuracy | 100% Pathological Recall
```

**Stage 1 — Cost-Aware Sample Weighting:** Assigns training weights Normal=1, Suspect=5, Pathological=10 to force the Gradient Boosting loss function to penalize minority class errors more heavily.

**Stage 2 — Probability Threshold Tuning:** Lowers the classification threshold for Pathological from default (~33%) to 10%, catching every sick case at the cost of only 3 additional false alarms.

### Why Not Standard Preprocessing?

Our ablation experiments show that popular preprocessing steps **actually hurt** the best tree-based models:

| Preprocessing Step | Effect on Accuracy |
|---|---|
| Outlier Capping | ↓ Removed extreme values that distinguish Pathological |
| Log Transform | ↓ Doesn't affect tree split order, degrades threshold learning |
| Feature Engineering (+9 features) | ↓ Caused importance dilution across correlated columns |
| SMOTE Oversampling | ↓ Synthetic points fell in ambiguous class boundaries |

**Lesson:** Preprocessing should be validated empirically per model family, not applied as a default step.

---

## Dataset

- **Source:** [UCI Fetal Health CTG Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) (Kaggle)
- **Records:** 2,126 → 2,113 after deduplication
- **Features:** 21 extracted from FHR and uterine contraction signals
- **Classes:** Normal (77.9%), Suspect (13.8%), Pathological (8.3%)
- **Null values:** 0

---

## Project Structure

```
├── Fetal_Health_Classification.ipynb   # Main notebook (well-commented, viva-ready)
├── fetal_health.csv                    # Dataset (download from Kaggle)
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## How to Run

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/fetal-health-classification.git
cd fetal-health-classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset

Download `fetal_health.csv` from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) and place it in the project root directory.

### Step 4: Run the Notebook

```bash
jupyter notebook Fetal_Health_Classification.ipynb
```

Run all cells sequentially (Cell → Run All). The notebook will:
1. Load and clean the data
2. Perform exploratory data analysis with visualizations
3. Train 11 model configurations
4. Display per-model evaluation with confusion matrices
5. Print the final comparison table and recommendation

**Expected runtime:** ~2–3 minutes on a standard laptop.

---

## Models Benchmarked

| # | Model | Accuracy | Recall (Pathological) |
|---|-------|----------|----------------------|
| 1 | **GB + Threshold Tuning** | **96.22%** | **100.00%** |
| 2 | **GB + Sample Weighting (1:5:10)** | **96.22%** | **97.14%** |
| 3 | Gradient Boosting (plain) | 95.98% | 94.29% |
| 4 | Voting Ensemble (GB+RF+ET) | 95.74% | 91.43% |
| 5 | Random Forest | 95.04% | 91.43% |
| 6 | Extra Trees | 93.85% | 88.57% |
| 7 | Decision Tree | 93.14% | 88.57% |
| 8 | KNN | 91.96% | 77.14% |
| 9 | Neural Network (MLP) | 91.96% | 77.14% |
| 10 | SVM | 90.31% | 77.14% |
| 11 | Logistic Regression | 88.89% | 82.86% |

---

## References

1. Şahin, H., & Subasi, A. (2015). Classification of the cardiotocogram data for anticipation of fetal risks using machine learning techniques. *Applied Soft Computing*, 33, 231–238.
2. Rahmayanti, N., et al. (2022). Comparison of machine learning algorithms to classify fetal health using cardiotocogram data. *Procedia Computer Science*, 197, 162–171.
3. Subasi, A., et al. (2020). Classification of the cardiotocogram data using bagging ensemble classifier. *Procedia Computer Science*, 168, 34–39.
4. Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository.
5. Cömert, Z. & Kocamaz, A. (2017). Comparison of machine learning techniques for fetal heart rate classification. *Acta Physica Polonica A*, 132, 451–454.
6. Cömert, Z., et al. (2019). Prediction of intrapartum fetal hypoxia considering feature selection algorithms and machine learning models. *Health Information Science and Systems*, 7, 1–9.

---

## License

This project is for academic purposes as part of the Co-Op Project at Chitkara University, Punjab.
