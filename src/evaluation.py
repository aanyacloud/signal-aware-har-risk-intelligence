import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_model_comparison(results):
    models = list(results.keys())
    accuracies = list(results.values())

    plt.figure()
    plt.bar(models, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()


def print_classification_report(y_test, y_pred):
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


def plot_decision_distribution(decisions):
    plt.figure()
    plt.hist(decisions, bins=3)
    plt.title("Decision Distribution")
    plt.xlabel("Decision Level")
    plt.ylabel("Frequency")
    plt.savefig("outputs/decision_distribution.png")
    plt.show()