import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, METRICS_DIR, FIGURES_DIR


def main():
    model_path = MODELS_DIR / "final_random_forest.joblib"
    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top5_df = importance_df.head(5).copy()
    top5_df["feature"] = top5_df["feature"].str.replace("num__", "", regex=False)
    top5_df["feature"] = top5_df["feature"].str.replace("cat__", "", regex=False)

    csv_path = METRICS_DIR / "top5_feature_importance.csv"
    top5_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(top5_df["feature"], top5_df["importance"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Importance")
    plt.title("Top 5 Feature Importances - Final Random Forest")
    plt.tight_layout()

    fig_path = FIGURES_DIR / "top5_feature_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Top 5 feature importances:")
    print(top5_df)
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()