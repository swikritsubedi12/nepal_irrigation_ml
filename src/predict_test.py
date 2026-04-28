import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, MODELS_DIR, PREDICTIONS_DIR
from src.data_utils import load_train_data, load_test_data, get_test_ids
from src.preprocessing import add_features, build_pipeline
from src.evaluate import evaluate_model


def main():
    # Load data
    train_df = load_train_data()
    test_df = load_test_data()

    # Prepare training data
    train_df = add_features(train_df)
    X_train = train_df.drop(columns=["Irrigation_Need", "id"], errors="ignore")
    y_train = train_df["Irrigation_Need"]

    # Prepare test data
    test_ids = get_test_ids(test_df)
    test_df = add_features(test_df)
    X_test = test_df.drop(columns=["id"], errors="ignore")

    # Final selected model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = build_pipeline(model)

    # Train on full training data
    pipeline.fit(X_train, y_train)

    # Save model
    model_path = MODELS_DIR / "final_random_forest.joblib"
    joblib.dump(pipeline, model_path)

    # Predict test set
    test_predictions = pipeline.predict(X_test)

    # Save predictions
    submission = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": test_predictions
    })

    prediction_path = PREDICTIONS_DIR / "final_test_predictions.csv"
    submission.to_csv(prediction_path, index=False)

    print(f"Final model saved to: {model_path}")
    print(f"Predictions saved to: {prediction_path}")
    print("\nPrediction preview:")
    print(submission.head())


if __name__ == "__main__":
    main()