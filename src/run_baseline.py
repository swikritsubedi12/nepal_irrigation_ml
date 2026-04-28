from src.data_utils import load_train_data
from src.preprocessing import split_data, build_pipeline
from src.models import get_models
from src.evaluate import evaluate_model, save_results_summary


def main():
    train_df = load_train_data()

    X_train, X_valid, y_train, y_valid = split_data(train_df)
    models = get_models()

    results = []

    for model_name, model in models.items():
        pipeline = build_pipeline(model)

        result = evaluate_model(
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            model_name=model_name,
            experiment_name="baseline"
        )

        results.append(result)

    save_results_summary(results, experiment_name="baseline")


if __name__ == "__main__":
    main()