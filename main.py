import os
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting

from analysis import train_model
from manger import load_data, preprocess_data, visualize_data, plot_confusion_matrix


def main():
    # Path to the directory containing CSV files
    file_path = "/Users/danh/Python/KNN /MachineLearningCVE"

    try:
        # Load the data from CSV files in the specified directory
        data = load_data(file_path)

        if data is not None:
            print("Các cột trong DataFrame:")
            print(data.columns)

            print("\nDữ liệu đầu vào (5 dòng đầu):")
            print(data.head())

            x_scaled, y_encoded = preprocess_data(data)

            visualize_data(data)

            accuracy, report, cm = train_model(x_scaled, y_encoded)

            print("Precision level:", accuracy)
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", cm)

            plot_confusion_matrix(cm)

            plt.show()
        else:
            print("No CSV files found in the specified directory.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()