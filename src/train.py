import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# bring in the pipeline
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_pipeline
from preprocess import prepare_X_y

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

    # 4. Train Model
    print(f"Training {model_name} model...")
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=random_state
    )

    print("Fitting model to training data...")
    model.fit(X_train, y_train)
    print("Model training complete.")

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
 
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/home/dragon/Documents/Repos/ML-Semester-Project/src/FraudShield_Banking_Data.csv"
 
    train(data_path)
