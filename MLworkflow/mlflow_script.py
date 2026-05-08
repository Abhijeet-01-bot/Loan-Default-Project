import sys
import os

# ✅ Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import mlflow
import mlflow.sklearn

from ingest import load_data
from preprocess import preprocess_train, preprocess_test

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ✅ FORCE MLflow to use correct storage folder
mlflow.set_tracking_uri("file:./mlruns")


def run_mlflow():

    # ✅ Load dataset
    df = load_data()

    # ✅ Split raw data FIRST (no leakage)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    # ✅ Preprocess correctly
    X_train, y_train, encoders = preprocess_train(train_df)
    X_test, y_test = preprocess_test(test_df, encoders)

    # ✅ Start MLflow run
    with mlflow.start_run():

        # ✅ Use your BEST MODEL (Naive Bayes)
        model = GaussianNB()
        model.fit(X_train, y_train)

        # ✅ Predictions
        preds = model.predict(X_test)

        # ✅ Accuracy
        acc = accuracy_score(y_test, preds)

        # ✅ LOG EVERYTHING TO MLFLOW
        mlflow.log_param("model", "NaiveBayes")
        mlflow.log_metric("accuracy", acc)

        # ✅ SAVE MODEL ARTIFACT
        mlflow.sklearn.log_model(model, "model")

        print("✅ MLflow run logged successfully")
        print(f"✅ Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_mlflow()
