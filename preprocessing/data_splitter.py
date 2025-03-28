import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

def split_data(X: np.ndarray, df: pd.DataFrame):
    y = df.y.to_numpy()
    y_series = pd.Series(y)

    good_y_value = y_series.value_counts(
    )[y_series.value_counts() >= 3].index

    if len(good_y_value) < 1:
        print("None of the class have more than 3 records: Skipping ...")
        X_train = None
        return None

    y_good = y[y_series.isin(good_y_value)]
    X_good = X[y_series.isin(good_y_value)]

    new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good)
    y = y_good
    classes = good_y_value
    embeddings = X
    return (X_train, X_test, y_train, y_test, y, classes, embeddings)