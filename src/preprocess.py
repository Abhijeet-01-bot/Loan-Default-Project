from sklearn.preprocessing import LabelEncoder

def preprocess_train(df):
    encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop("Status", axis=1)
    y = df["Status"]

    return X, y, encoders


def preprocess_test(df, encoders):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = encoders[col].transform(df[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop("Status", axis=1)
    y = df["Status"]

    return X, y
