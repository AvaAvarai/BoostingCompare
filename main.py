import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_runs, shuffle=True, random_state=42)

    for train_index, test_index in tqdm(skf.split(X, y), total=n_runs, desc="Running iterations"):
        # Split the data using the indices from StratifiedKFold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Print the number of cases per class for train and test sets
        print("\nNumber of cases per class:")
        print("Train set:")
        for class_name, count in zip(['setosa', 'versicolor', 'virginica'], np.bincount(y_train)):
            print(f"{class_name}: {count}")
        print("Test set:")
        for class_name, count in zip(['setosa', 'versicolor', 'virginica'], np.bincount(y_test)):
            print(f"{class_name}: {count}")

        # Create and train the AdaBoost classifier with SAMME algorithm
        clf = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
        clf.fit(X_train, y_train)

        # Print the weak classifiers used
        print("\nWeak classifiers used:")
        for i, estimator in enumerate(clf.estimators_):
            print(f"Estimator {i + 1}: {estimator}")

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

    # Predict on training data to identify misclassified cases
    best_y_pred = best_clf.predict(best_X_train)
    df_best['misclassified'] = best_y_pred != best_y_train

    plt.subplot(121)
    for class_name in ['setosa', 'versicolor', 'virginica']:
        class_data = df_best[df_best['class'] == class_name]
        correct = class_data[~class_data['misclassified']]
        incorrect = class_data[class_data['misclassified']]
        
        parallel_coordinates(correct, 'class', color={'setosa': '#1f77b4', 'versicolor': '#ff7f0e', 'virginica': '#2ca02c'}[class_name], alpha=0.7)
        if not incorrect.empty:
            parallel_coordinates(incorrect, 'class', color='yellow', alpha=1.0)

    plt.title('Parallel Coordinates: Best Decision Boundary')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot parallel coordinates for the worst run
    worst_clf, worst_X, worst_y, _, _ = classifiers[worst_run]
    worst_y_named = map_class_names(worst_y)  # Map class labels to names

    # Create DataFrame for parallel coordinates for worst classifier
    df_worst = pd.DataFrame(worst_X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    df_worst['class'] = worst_y_named

    # Predict on training data to identify misclassified cases
    worst_y_pred = worst_clf.predict(worst_X)
    df_worst['misclassified'] = worst_y_pred != worst_y

    plt.subplot(122)
    for class_name in ['setosa', 'versicolor', 'virginica']:
        class_data = df_worst[df_worst['class'] == class_name]
        correct = class_data[~class_data['misclassified']]
        incorrect = class_data[class_data['misclassified']]
        
        parallel_coordinates(correct, 'class', color={'setosa': '#1f77b4', 'versicolor': '#ff7f0e', 'virginica': '#2ca02c'}[class_name], alpha=0.7)
        if not incorrect.empty:
            parallel_coordinates(incorrect, 'class', color='yellow', alpha=1.0)

    plt.title('Parallel Coordinates: Worst Decision Boundary')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()
