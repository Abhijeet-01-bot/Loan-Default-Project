import joblib
from ingest import load_data
from preprocess import preprocess_train, preprocess_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def evaluate():
    df = load_data()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    X_train, y_train, encoders = preprocess_train(train_df)
    X_test, y_test = preprocess_test(test_df, encoders)

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = joblib.load("models/model.pkl")

    y_pred = model.predict(X_test)

    print("\n📊 FINAL MODEL EVALUATION:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()

