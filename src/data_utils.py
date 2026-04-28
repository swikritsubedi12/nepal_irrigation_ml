import pandas as pd
from src.config import TRAIN_FILE, TEST_FILE, TARGET, ID_COL


def load_train_data():
    return pd.read_csv(TRAIN_FILE)


def load_test_data():
    return pd.read_csv(TEST_FILE)


def check_data(df, name="data"):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    if TARGET in df.columns:
        print("\nTarget distribution:")
        print(df[TARGET].value_counts())

        print("\nTarget distribution (%):")
        print((df[TARGET].value_counts(normalize=True) * 100).round(2))


def split_features_target(df):
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]
    return X, y


def get_test_ids(df):
    return df[ID_COL]


if __name__ == "__main__":
    train_df = load_train_data()
    test_df = load_test_data()

    check_data(train_df, "train")
    check_data(test_df, "test")