import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import METRICS_DIR, FIGURES_DIR


def evaluate_model(model, X_train, y_train, X_valid, y_valid, model_name, experiment_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_valid, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)

    results = {
        "experiment": experiment_name,
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

    print(f"\n===== {model_name} | {experiment_name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    save_classification_report(y_valid, y_pred, model_name, experiment_name)
    save_confusion_matrix(y_valid, y_pred, model_name, experiment_name)

    return results


def save_classification_report(y_true, y_pred, model_name, experiment_name):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    file_name = f"{experiment_name}_{model_name.lower().replace(' ', '_')}_report.csv"
    file_path = os.path.join(METRICS_DIR, file_name)
    report_df.to_csv(file_path, index=True)


def save_confusion_matrix(y_true, y_pred, model_name, experiment_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name} ({experiment_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    file_name = f"{experiment_name}_{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    file_path = os.path.join(FIGURES_DIR, file_name)
    plt.savefig(file_path, dpi=300)
    plt.close()


def save_results_summary(results_list, experiment_name):
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="macro_f1", ascending=False)

    file_name = f"{experiment_name}_summary.csv"
    file_path = os.path.join(METRICS_DIR, file_name)
    results_df.to_csv(file_path, index=False)

    print(f"\nSaved summary to: {file_path}")
    print("\nResults Summary:")
    print(results_df)

    return results_df