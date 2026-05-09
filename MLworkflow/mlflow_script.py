import os
import sys
import mlflow
import mlflow.sklearn

# ✅ Fix path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingest import load_data
from preprocess import preprocess_train, preprocess_test

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ✅ CRITICAL FIX: Use local folder (no permission issue)
mlflow.set_tracking_uri("file:./mlruns")


def run_mlflow():

    # ✅ Load data
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

        # ✅ Calculate accuracy
        acc = accuracy_score(y_test, preds)

        # ✅ Log parameters
        mlflow.log_param("model", "NaiveBayes")

        # ✅ Log metrics
        mlflow.log_metric("accuracy", acc)

        # ✅ SAVE MODEL (FIXED — no permission issue)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("✅ MLflow run logged successfully")
        print(f"✅ Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_mlflow()