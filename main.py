import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Get number of runs from user
    n_runs = int(input("Enter the number of runs: "))
    accuracies = []
    roc_curves = []

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

        # Calculate ROC curve and AUC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        y_score = clf.predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        roc_curves.append((fpr, tpr, roc_auc))

    # Calculate statistics
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # Find best and worst ROC curves
    best_run = np.argmax(accuracies)
    worst_run = np.argmin(accuracies)

    # Print results
    print(f"Min Accuracy: {min_accuracy:.4f}")
    print(f"Max Accuracy: {max_accuracy:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Standard Deviation of Accuracies: {std_accuracy:.4f}")

    # Plot best and worst ROC curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    for i in range(3):
        plt.plot(roc_curves[best_run][0][i], roc_curves[best_run][1][i], 
                 label=f'ROC curve (class {i}) (AUC = {roc_curves[best_run][2][i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best ROC Curve')
    plt.legend(loc="lower right")

    plt.subplot(122)
    for i in range(3):
        plt.plot(roc_curves[worst_run][0][i], roc_curves[worst_run][1][i], 
                 label=f'ROC curve (class {i}) (AUC = {roc_curves[worst_run][2][i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Worst ROC Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
