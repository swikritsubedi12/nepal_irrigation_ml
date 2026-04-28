import pandas as pd
from src.config import METRICS_DIR


def main():
    baseline_path = METRICS_DIR / "baseline_summary.csv"
    smote_path = METRICS_DIR / "smote_summary.csv"

    baseline_df = pd.read_csv(baseline_path)
    smote_df = pd.read_csv(smote_path)

    combined_df = pd.concat([baseline_df, smote_df], ignore_index=True)
    combined_df = combined_df.sort_values(by="macro_f1", ascending=False)

    output_path = METRICS_DIR / "all_model_comparison.csv"
    combined_df.to_csv(output_path, index=False)

    print("Combined comparison saved to:", output_path)
    print("\nFinal Comparison Table:")
    print(combined_df)


if __name__ == "__main__":
    main()