from sklearn.base import clone
 
class HierarchicalModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.type2_model = clone(base_model)
        self.type3_models = {}  # key: type_2
        self.type4_models = {}  # key: (type_2, type_3)
 
    def fit(self, X, df):
        self.type2_model.fit(X, df["type_2"])
        for t2 in df["type_2"].unique():
            mask2 = df["type_2"] == t2
            self.type3_models[t2] = clone(self.base_model)
            self.type3_models[t2].fit(X[mask2], df.loc[mask2, "type_3"])
            for t3 in df.loc[mask2, "type_3"].unique():
                mask3 = (df["type_2"] == t2) & (df["type_3"] == t3)
                y_type4 = df.loc[mask3, "type_4"]
                if len(y_type4.unique()) <= 1:
                    print(f"Skipping type_4 model for ({t2}, {t3}) — only one class found.")
                    continue
                self.type4_models[(t2, t3)] = clone(self.base_model)
                self.type4_models[(t2, t3)].fit(X[mask3], y_type4)
 
    def predict(self, X):
        pred2 = self.type2_model.predict(X)
        pred3, pred4 = [], []
 
        for i, t2 in enumerate(pred2):
            x_row = X[i].reshape(1, -1)
            t3 = self.type3_models[t2].predict(x_row)[0] if t2 in self.type3_models else "unknown"
            pred3.append(t3)
 
            key = (t2, t3)
            t4 = self.type4_models[key].predict(x_row)[0] if key in self.type4_models else "unknown"
            pred4.append(t4)
 
        return {"type_2": pred2, "type_3": pred3, "type_4": pred4}