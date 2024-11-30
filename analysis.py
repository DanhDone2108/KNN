import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to train the KNN model
def train_model(x, y, k=3):
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize KDTree with training data
    tree = KDTree(x_train)

    # Predict on the test set using KDTree to find nearest neighbors
    dist, ind = tree.query(x_test, k=k)  # Find k nearest neighbors

    # Collect predictions based on nearest neighbors' votes
    y_pred = []
    for neighbors in ind:
        votes = np.bincount(y_train[neighbors])  # Count votes for each class
        y_pred.append(np.argmax(votes))  # Select the class with the highest votes

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100.0))

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, report, cm