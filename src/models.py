from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    return models