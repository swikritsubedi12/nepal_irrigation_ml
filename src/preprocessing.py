import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET, ID_COL, TEST_SIZE, RANDOM_STATE


CATEGORICAL_FEATURES = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

NUMERICAL_FEATURES = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]


def add_features(df):
    df = df.copy()

    df["Moisture_Deficit"] = df["Humidity"] - df["Soil_Moisture"]
    df["Soil_Stress_Score"] = abs(df["Soil_pH"] - 7) + df["Electrical_Conductivity"]
    df["Temp_Rainfall_Ratio"] = df["Temperature_C"] / (df["Rainfall_mm"] + 1)
    df["Irrigation_Per_Hectare"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 1)

    return df


def get_feature_lists():
    categorical = CATEGORICAL_FEATURES.copy()
    numerical = NUMERICAL_FEATURES.copy()

    numerical.extend([
        "Moisture_Deficit",
        "Soil_Stress_Score",
        "Temp_Rainfall_Ratio",
        "Irrigation_Per_Hectare",
    ])

    return categorical, numerical


def split_data(df):
    df = add_features(df)

    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_valid, y_train, y_valid


def build_preprocessor():
    categorical, numerical = get_feature_lists()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return preprocessor


def build_pipeline(model):
    preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def prepare_test_data(df):
    df = add_features(df)
    ids = df[ID_COL].copy()
    X_test = df.drop(columns=[ID_COL], errors="ignore")
    return ids, X_test