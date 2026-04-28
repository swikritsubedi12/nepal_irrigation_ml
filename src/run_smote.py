from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import RANDOM_STATE
from src.data_utils import load_train_data
from src.preprocessing import split_data, get_feature_lists
from src.evaluate import evaluate_model, save_results_summary


def get_smote_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }
    return models


def build_smote_pipeline(model):
    categorical_features, numerical_features = get_feature_lists()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ]
    )

    categorical_indices = list(
        range(len(numerical_features), len(numerical_features) + len(categorical_features))
    )

    smote = SMOTENC(
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("model", model),
        ]
    )

    return pipeline


def main():
    train_df = load_train_data()
    X_train, X_valid, y_train, y_valid = split_data(train_df)

    models = get_smote_models()
    results = []

    for model_name, model in models.items():
        pipeline = build_smote_pipeline(model)

        result = evaluate_model(
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            model_name=model_name,
            experiment_name="smote"
        )

        results.append(result)

    save_results_summary(results, experiment_name="smote")


if __name__ == "__main__":
    main()