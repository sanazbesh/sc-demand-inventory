import lightgbm as lgb
import numpy as np
import pandas as pd

class QuantileGBM:
    def __init__(self, quantiles=(0.1, 0.5, 0.9), params=None, num_boost_round=400):
        self.quantiles = quantiles
        self.models = {}
        self.params = params or dict(objective="quantile", learning_rate=0.05,
                                     num_leaves=63, min_data_in_leaf=50)
        self.num_boost_round = num_boost_round

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for q in self.quantiles:
            p = self.params.copy()
            p["alpha"] = q
            dtrain = lgb.Dataset(X, label=y)
            self.models[q] = lgb.train(p, dtrain, num_boost_round=self.num_boost_round, verbose_eval=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = {q: self.models[q].predict(X) for q in self.quantiles}
        return np.column_stack([preds[q] for q in self.quantiles])
