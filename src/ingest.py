import pandas as pd

def load_data(path="data/Loan_Default.csv"):
    return pd.read_csv(path)
