import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):

    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    if ' Label' not in data.columns:
        raise KeyError("Column ' Label' not found in data")
    x = data.drop(columns=['Timestamp'], errors='ignore')
    y = data[' Label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    for column in x.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x[column] = le.fit_transform(x[column].astype(str))
        x.fillna(x.mean(), inplace=True)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.fillna(x.mean(), inplace=True)
    return x, y_encoded