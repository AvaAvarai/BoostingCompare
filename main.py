import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.plotting import parallel_coordinates
import pandas as pd

# Define a function to map numeric class labels to class names
def map_class_names(labels):
    class_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return [class_map[label] for label in labels]

if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Get number of runs from user
    n_runs = int(input("Enter the number of runs: "))
    accuracies = []
    classifiers = []

    for _ in tqdm(range(n_runs), desc="Running iterations"):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create and train the AdaBoost classifier with SAMME algorithm
        clf = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
        clf.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        classifiers.append((clf, X_train, y_train, X_test, y_test))

    # Find best and worst runs
    best_run = np.argmax(accuracies)
    worst_run = np.argmin(accuracies)

    # Train classifier on all data and print confusion matrix
    clf_all = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
    clf_all.fit(X, y)
    y_pred_all = clf_all.predict(X)
    cm = confusion_matrix(y, y_pred_all)
    print("Confusion Matrix for Classifier trained on all data:")
    print(cm)

    # Plot parallel coordinates for the best run
    plt.figure(figsize=(14, 7))

    best_clf, best_X_train, best_y_train, best_X_test, best_y_test = classifiers[best_run]
    best_y_named = map_class_names(best_y_train)  # Map class labels to names

    # Create DataFrame for parallel coordinates for best classifier
    df_best = pd.DataFrame(best_X_train, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    df_best['class'] = best_y_named

    plt.subplot(121)
    parallel_coordinates(df_best, 'class', color=('#1f77b4', '#ff7f0e', '#2ca02c'))  # Fixed class colors
    plt.title('Parallel Coordinates: Best Decision Boundary')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot parallel coordinates for the worst run
    worst_clf, worst_X, worst_y, _, _ = classifiers[worst_run]
    worst_y_named = map_class_names(worst_y)  # Map class labels to names

    # Create DataFrame for parallel coordinates for worst classifier
    df_worst = pd.DataFrame(worst_X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    df_worst['class'] = worst_y_named

    plt.subplot(122)
    parallel_coordinates(df_worst, 'class', color=('#1f77b4', '#ff7f0e', '#2ca02c'))  # Same class colors
    plt.title('Parallel Coordinates: Worst Decision Boundary')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()
