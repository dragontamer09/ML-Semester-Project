import joblib
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "classifier",
) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], zero_division=0)

    # Pull individual confusion matrix cells for clarity

    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*55}")
    print(f"  Results: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"               Predicted Normal  Predicted Fraud")
    print(f"  Actual Normal      {tn:<18} {fp}")
    print(f"  Actual Fraud       {fn:<18} {tp}")
    print(f"\n  Classification report:")
    print(report)

    return {
        "model_name": model_name,
        "accuracy":   accuracy,
        "roc_auc":    roc_auc,
        "cm":         cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

def load_and_evaluate(
    model_path: str | Path,
    prep_path: str | Path,
    data_path: str | Path,
    model_name: str = "classifier",
) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from load_data import load_pipeline
    from preprocess import prepare_X_y
    from sklearn.model_selection import train_test_split

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    labeled, _ = load_pipeline(data_path)
    X, y, _ = prepare_X_y(labeled, preprocessor=preprocessor, fit=False)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return evaluate(model, X_test, y_test, model_name)

if __name__ == "__main__":
        if len(sys.argv) < 4:
            print("Usage: python evaluate.py <model.joblib> <preprocessor.joblib> <data.csv>")
            sys.exit(1)
 
        load_and_evaluate(
            model_path=sys.argv[1],
            prep_path=sys.argv[2],
            data_path=sys.argv[3],
            model_name=Path(sys.argv[1]).stem,
        )







