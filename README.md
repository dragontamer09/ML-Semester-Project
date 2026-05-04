# FraudShield — ML Fraud Detection

A machine learning pipeline that compares a **Decision Tree** and a **Linear SVM** for detecting fraudulent financial transactions. Built for COMP4330 – Intro to Machine Learning, Northwest Nazarene University.

---

## Project Structure

```
ML-Semester-Project/
├── src/
│   ├── train.py              # Main entry point — trains and evaluates a classifier
│   ├── evaluate.py           # Scoring and metrics
│   ├── load_data.py          # CSV ingestion and tuple conversion
│   ├── preprocess.py         # Feature engineering and sklearn preprocessing
│   ├── feature_metadata.py   # Single source of truth for all feature definitions
│   └── dist/
│       └── train.exe         # Standalone Windows executable (no Python needed)
├── data/
│   └── raw/
│       └── FraudShield_Banking_Data.csv
└── outputs/
    └── models/               # Saved model and preprocessor files (.joblib)
```

---

## Requirements

### Running from Source

- Python 3.11+
- Install dependencies:

```bash
pip install pandas scikit-learn imbalanced-learn joblib openpyxl
```

### Running the Executable

- Windows only
- No Python installation required
- `train.exe` is self-contained

---

## Running the Code

### From Source

Run from the **project root** (`ML-Semester-Project/`), not from inside `src/`.

**Train the Decision Tree:**
```bash
python src/train.py data/raw/FraudShield_Banking_Data.csv --model decision_tree
```

**Train the SVM:**
```bash
python src/train.py data/raw/FraudShield_Banking_Data.csv --model svm
```

The `--model` flag defaults to `decision_tree` if omitted.

---

### From the Executable

Navigate to the `src/dist/` folder or run it directly from the terminal:

**Train the Decision Tree:**
```bash
./train.exe "C:\path\to\FraudShield_Banking_Data.csv" --model decision_tree
```

**Train the SVM:**
```bash
./train.exe "C:\path\to\FraudShield_Banking_Data.csv" --model svm
```

> **Note:** Use the full path to the CSV file when running the executable, since it runs from a temporary directory and may not resolve relative paths correctly.

---

### Re-evaluating a Saved Model

If you've already trained a model and want to re-score it without retraining:

```bash
python src/evaluate.py outputs/models/decision_tree.joblib outputs/models/decision_tree_preprocessor.joblib data/raw/FraudShield_Banking_Data.csv
```

The argument order is: `model file`, `preprocessor file`, `data file`.
Though I am fairly certain this breaks. Attempt at your own risk.

> **Important:** Always load the model and preprocessor together as a matched pair. The preprocessor was fitted on the same training data the model was trained on — using a mismatched preprocessor will produce incorrect results.

---

## What Happens When You Run It

The pipeline runs five steps automatically:

1. **Load** — reads the CSV and splits rows into labeled (training) and unlabeled (inference) tuples
2. **Preprocess** — parses time/date columns, drops identifiers, encodes binary flags, scales numerics, one-hot encodes categoricals
3. **Split** — divides labeled data into 80% training / 20% test using a stratified split (preserves the fraud/normal ratio in both halves)
4. **SMOTE** — synthesizes additional fraud examples in the training set to balance the 95/5 class imbalance
5. **Train and evaluate** — fits the chosen classifier and prints results to the terminal

Trained models are saved to `outputs/models/` as `.joblib` files.

---

## Interpreting the Results

When training completes, the terminal prints a results block that looks like this:

```
=======================================================
  Results: decision_tree
=======================================================
  Accuracy : 0.8886
  ROC-AUC  : 0.5012

  Confusion matrix:
               Predicted Normal  Predicted Fraud
  Actual Normal      8851               664
  Actual Fraud        450                35

  Classification report:
               precision    recall  f1-score   support
      Normal       0.95      0.93      0.94      9515
       Fraud       0.05      0.07      0.06       485
```

### Metrics

| Metric | What it measures | What to look for |
|---|---|---|
| **Accuracy** | Fraction of all predictions that were correct | Not reliable here — a model predicting everything as Normal gets 95% accuracy while catching zero fraud |
| **ROC-AUC** | How well the model separates fraud from normal based on its internal probability scores | 0.5 = random, 1.0 = perfect. Anything below 0.6 is poor |
| **Recall (Fraud)** | Of all actual fraud cases, how many did the model catch | The most important metric for fraud detection — missing fraud is more costly than a false alarm |
| **Precision (Fraud)** | Of all fraud predictions, how many were actually fraud | Lower precision = more false alarms flagged for review |
| **F1-Score (Fraud)** | Harmonic mean of precision and recall | Useful single number combining both — only meaningful when both are reasonable |

### Confusion Matrix

```
                  Predicted Normal    Predicted Fraud
Actual Normal          TN                  FP
Actual Fraud           FN                  TP
```

| Cell | Name | Meaning |
|---|---|---|
| Top-left | True Negative (TN) | Correctly identified as normal — want this high |
| Top-right | False Positive (FP) | Flagged as fraud but was normal — false alarm |
| Bottom-left | False Negative (FN) | Missed fraud — predicted normal but was fraud — most costly error |
| Bottom-right | True Positive (TP) | Correctly caught fraud — want this high |

### Which Metric Matters Most

For fraud detection, **recall is the priority**. Missing a fraudulent transaction (False Negative) causes real financial harm. Generating a false alarm (False Positive) just means a legitimate transaction gets reviewed by a human — inconvenient, but far less damaging.

A model with 49% recall and low precision is more useful in this context than a model with 95% accuracy that catches zero fraud.

---

## Results Summary

| | Decision Tree (no SMOTE) | Linear SVM (no SMOTE) | Decision Tree (SMOTE) | Linear SVM (SMOTE) |
|---|---|---|---|---|
| Accuracy | 0.9095 | 0.9515 | 0.8886 | 0.6115 |
| ROC-AUC | 0.5043 | 0.5963 | 0.5012 | 0.5952 |
| Fraud Recall | 0.06 | 0.00 | 0.07 | **0.49** |
| Fraud F1 | 0.06 | 0.00 | 0.06 | 0.11 |

The **Linear SVM with SMOTE** is the most effective fraud detector based on recall, catching 239 of 485 fraud cases in the test set. The Decision Tree is more conservative and generates fewer false alarms but misses the majority of actual fraud.

---

## Notes

- The dataset has a significant class imbalance (4.8% fraud). SMOTE is applied automatically to the training set to address this.
- The random seed is fixed at `42` throughout, so results are reproducible across runs.
- Model and preprocessor files are saved as a matched pair. Never swap the preprocessor from one model's output into another model's evaluation.
