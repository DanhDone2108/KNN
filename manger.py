import numpy as np  # Data processing
import pandas as pd  # Data reading
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns


# Function to load data from CSV files in a directory
def load_data(file_path):
    data_files = os.listdir(file_path)
    dataframes = []

    for file in data_files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(file_path, file))
            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True) if dataframes else None


# Preprocess the dataset
def preprocess_data(data):
    # Check if the 'Label' column exists in the DataFrame
    if ' Label' not in data.columns:
        raise KeyError("Column ' Label' not found in data")

    # Handle missing values and infinite values
    data.fillna(0, inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)

    # Separate features and labels
    x = data.drop(columns=['Timestamp'], errors='ignore')
    y = data[' Label']  # Create y containing labels from the 'Label' column

    le = LabelEncoder()  # Initialize LabelEncoder to encode labels
    y_encoded = le.fit_transform(y)

    # Encode categorical features as strings
    for column in x.select_dtypes(include=['object']).columns:
        x[column] = le.fit_transform(x[column].astype(str))

    # Replace infinite values with NaN and then with column mean
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.fillna(x.mean(), inplace=True)

    # Apply Min-Max Scaling to features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, y_encoded  # Return scaled features and encoded labels


# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Function to visualize data using various plots
def visualize_data(data):
    plot_histogram(data, ' Flow Duration')  # Example: Histogram of 'Flow Duration'

    # Example: Scatter plot between 'Total Fwd Packets' and 'Total Backward Packets'
    plot_scatter(data, ' Total Fwd Packets', ' Total Backward Packets')

    # Example: Boxplot for the column 'Flow Duration'
    plot_boxplot(data, ' Flow Duration')


# Function to plot histogram for a specific column
def plot_histogram(data, column):
    plt.figure(figsize=(15, 10), dpi=100)  # Image size 15x10 inches with DPI 100
    sns.histplot(data[column], bins=30, kde=True)  # KDE for distribution curve
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# Function to plot scatter plot between two specific columns
def plot_scatter(data, x_column, y_column):
    plt.figure(figsize=(15, 10), dpi=100)  # Image size 15x10 inches with DPI 100
    sns.scatterplot(x=data[x_column], y=data[y_column])
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


# Function to plot boxplot for a specific column
def plot_boxplot(data, column):
    plt.figure(figsize=(15, 10), dpi=100)  # Image size 15x10 inches with DPI 100
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()