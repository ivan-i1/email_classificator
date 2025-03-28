from models.randomforest import RandomForest
from models.adaboost import AdaBoost
from models.sgd import SGD
from models.base import BaseModel
import numpy as np


def get_model(model_name: str, embeddings: np.ndarray, y: np.ndarray) -> BaseModel:
    model_name = model_name.lower()

    if model_name == "rf":
        return RandomForest(model_name="RandomForest", embeddings=embeddings, y=y)

    elif model_name == "ada":
        return AdaBoost(model_name="AdaBoost", embeddings=embeddings, y=y)

    elif model_name == "sgd":
        return SGD(model_name="SGD", embeddings=embeddings, y=y)

    else:
        print(
            f"Model {model_name} not recognized. Using RandomForest as default.")
        return RandomForest(model_name="RandomForest", embeddings=embeddings, y=y)
