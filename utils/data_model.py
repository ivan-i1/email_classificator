import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)


class DataBundle():
    def __init__(self,
                X_train, X_test, y_train, y_test, y, classes, embeddings):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y = y
        self.classes = classes
        self.embeddings = embeddings

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df

    def get_X_DL_test(self):
        return self.X_DL_test

    def get_X_DL_train(self):
        return self.X_DL_train
