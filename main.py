import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

def load_data(name: str) -> pd.DataFrame:
    print(f"Load Data : {name}")
    return pd.read_csv(f"data/{name}", index_col='id')


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    return data


def val_score(model, features, targets, metrics=None) -> None:
    scores = cross_val_score(
        model,
        X=features,
        y=targets,
        scoring=metrics,
        cv=4,
        n_jobs=-1,
        verbose=20
        )
    print(f'mean {metrics} : {np.mean(scores):.5f} ± {np.std(scores):.5f}')


def split_features_targets(data: pd.DataFrame, targets_name: list)-> tuple:
    features = data.drop(columns=targets_name)
    targets = data[targets_name]
    return features, targets


if __name__ == "__main__":
    TARGETS = [
        'Pastry',
        'Z_Scratch',
        'K_Scatch',
        'Stains',
        'Dirtiness',
        'Bumps',
        'Other_Faults'
        ]
    SEED = 42
    SPLITS = 5
    train = load_data('train.csv')
    test = load_data('test.csv')
    submission = load_data('sample_submission.csv')
    model = XGBClassifier()
    X, Y = split_features_targets(train, TARGETS)
    
    val_score(model, X, Y, 'roc_auc')
