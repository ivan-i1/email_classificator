from sklearn.base import clone


class ChainedClassifier:
    def __init__(self, base_model):
        self.model_1 = clone(base_model)
        self.model_2 = clone(base_model)
        self.model_3 = clone(base_model)

    def fit(self, X, df):
        self.model_1.fit(X, df["type_2"])
        self.model_2.fit(X, df["type_2_3"])
        self.model_3.fit(X, df["type_2_3_4"])

    def predict_all(self, X):
        return {
            "type_2": self.model_1.predict(X),
            "type_2_3": self.model_2.predict(X),
            "type_2_3_4": self.model_3.predict(X)
        }
