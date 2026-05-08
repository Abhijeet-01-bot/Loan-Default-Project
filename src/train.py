from ingest import load_data
from preprocess import preprocess_train, preprocess_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def train():
    df = load_data()

    # ✅ Split raw data FIRST
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # ✅ Preprocess
    X_train, y_train, encoders = preprocess_train(train_df)
    X_test, y_test = preprocess_test(test_df, encoders)

    # ✅ SMOTE
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ✅ Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ✅ Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    best_model = None
    best_score = 0

    print("\n📊 MODEL PERFORMANCE:\n")

    for name, model in models.items():

        if name in ["Logistic Regression", "KNN"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 50)

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print("\n✅ Best Model:", best_name)
    print("✅ Best Accuracy:", best_score)

    # ✅ Save everything
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    print("\n✅ Model + scaler + encoders saved")

if __name__ == "__main__":
    train()
