import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/{name}.csv", index_col='id')


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    return data


def split_data(data: pd.DataFrame, targets_name: list, random_state: int):
    features = data.drop(columns=targets_name)
    targets = data[targets_name]
    return train_test_split(features, targets, 0.2, random_state=random_state)


if __name__ == "__main__":
    train = load_data('train')
    test = load_data('test')
