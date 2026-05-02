import time
import joblib
import argparse
import sys
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

# bring in the pipeline
# handles both normal execution and PyInstaller bundles
if getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS)
else:
    base_path = Path(__file__).resolve().parent

sys.path.insert(0, str(base_path))
from load_data import load_pipeline
from preprocess import prepare_X_y

def build_classifier(model_name: str, random_state: int):
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "svm":
        base_svm = LinearSVC(
        class_weight="balanced",
        max_iter=5000,      # was 2000, give it more iterations
        tol=1e-3,           # relaxed tolerance — stops when close enough
        dual=False,
        random_state=random_state,
        )   
        return CalibratedClassifierCV(base_svm, cv=5)
    elif model_name == "svm_rbf":
        from sklearn.svm import SVC
        base_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,    # enables predict_proba() natively, no calibration wrapper needed
        random_state=random_state,
        )
        return base_svm
    else:
        supported = ["decision_tree", "svm"]
        raise ValueError(
            f"Unknown model '{model_name}'. Supported options: {supported}"
        )
    

def train(
        data_path: str | Path,
        output_dir: str | Path = "outputs/models",
        model_name: str = "decision_tree",
        test_size: float = 0.2,
        random_state: int = 42,
) -> dict:
    # 1. Load
    print("Loading data...")
    labeled, _ = load_pipeline(data_path)
    print (f"Loaded {len(labeled):,} labeled entries.")

    # 2. Preprocess
    print("Preprocessing data...")
    X, y, preprocessor = prepare_X_y(labeled, fit=True)
    print(f" Feature Matrix : {X.shape}")
    print(f" Class Balance : {int((y == 0).sum())} normal / {int((y == 1).sum())} fraud")

    # 3. Train/Test Split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f" Train set : {X_train.shape[0]:,} samples")
    print(f" Test set  : {X_test.shape[0]:,} samples")

    # 4. Apply SMOTE to training data
    sm = SMOTE(random_state=random_state)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f" After SMOTE : {int((y_train==0).sum())} normal / {int((y_train==1).sum())} fraud")
    # 5. Build Classifier
    model = build_classifier(model_name, random_state)
    print(f"Training {model_name} ...")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"  Done. Training time: {elapsed:.2f} seconds.")

    # 5. Save Model and Preprocessor
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}_model.joblib"
    preprocessor_path = output_dir / f"{model_name}_preprocessor.joblib"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Saved model to {model_path}")
    print(f"Saved preprocessor to {preprocessor_path}")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

#Run the training pipeline and evaluate on the test set

if __name__ == "__main__":
 
    from evaluate import evaluate
 
    parser = argparse.ArgumentParser(description="Train a fraud detection classifier")
    parser.add_argument("data_path", help="Path to the raw CSV file")
    parser.add_argument(
        "--model",
        default="decision_tree",
        choices=["decision_tree", "svm", "svm_rbf"],
        help="Which classifier to train (default: decision_tree)",
    )
    args = parser.parse_args()
 
    results = train(args.data_path, model_name=args.model)
 
    print(f"\n--- Test set evaluation: {args.model} ---")
    evaluate(
        model=results["model"],
        X_test=results["X_test"],
        y_test=results["y_test"],
        model_name=args.model,
    )

