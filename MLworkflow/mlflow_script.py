import os
import sys
import mlflow
import mlflow.sklearn

# ✅ Fix import path for src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingest import load_data
from preprocess import preprocess_train, preprocess_test

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ✅ ✅ CRITICAL FIX (WORKS IN AZURE + GITHUB ACTIONS)
# Force MLflow to use local directory instead of restricted cloud paths

TRACKING_DIR = os.path.abspath("./mlruns")
os.makedirs(TRACKING_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")


def run_mlflow():

    # ✅ Load dataset
    df = load_data()

    # ✅ Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    # ✅ Preprocess
    X_train, y_train, encoders = preprocess_train(train_df)
    X_test, y_test = preprocess_test(test_df, encoders)

    # ✅ Start MLflow run
    with mlflow.start_run():

        # ✅ Train model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # ✅ Predict
        preds = model.predict(X_test)

        # ✅ Accuracy
        acc = accuracy_score(y_test, preds)

        # ✅ Log parameters
        mlflow.log_param("model", "NaiveBayes")

        # ✅ Log metrics
        mlflow.log_metric("accuracy", acc)

        # ✅ SAVE MODEL (UPDATED SYNTAX — NO DEPRECATION ISSUE)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"   # ✅ IMPORTANT (instead of artifact_path)
        )

        print("✅ MLflow run logged successfully")
        print(f"✅ Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_mlflow()
