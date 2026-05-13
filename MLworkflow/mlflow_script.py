import os
import sys
import mlflow
import mlflow.sklearn

# ✅ Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingest import load_data
from preprocess import preprocess_train, preprocess_test

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ✅ FIX 1: Safe tracking directory
TRACKING_DIR = os.path.abspath("./mlruns")
os.makedirs(TRACKING_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")

# ✅ ✅ FIX 2: CREATE OR SET EXPERIMENT (CRITICAL)
mlflow.set_experiment("Loan-Default-Experiment")


def run_mlflow():

    df = load_data()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    X_train, y_train, encoders = preprocess_train(train_df)
    X_test, y_test = preprocess_test(test_df, encoders)

    with mlflow.start_run():

        model = GaussianNB()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", "NaiveBayes")
        mlflow.log_metric("accuracy", acc)

        # ✅ Save model (modern syntax)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        print("✅ MLflow run logged successfully")
        print(f"✅ Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_mlflow()