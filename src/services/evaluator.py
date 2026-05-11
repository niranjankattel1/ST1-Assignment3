from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import EDA_OUTPUT_DIR

class Evaluator:
    """Evaluate model performance and generate reports."""

    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred) -> dict:
        """Compute evaluation metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

    def plot_confusion_matrix(self, y_true, y_pred, labels: list, filename: str = 'confusion_matrix.png'):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        output_path = EDA_OUTPUT_DIR / filename
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")

    def save_metrics(self, metrics: dict, filename: str = 'metrics.json'):
        """Save metrics to a JSON file."""
        import json
        output_path = EDA_OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_path}")