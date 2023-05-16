import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class RFRegressor(BaseRegressor):
    def __init__(self):
        #self.z_samples = z_samples
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        #self.m = RandomForestRegressor(n_estimators=int(y.shape[0]//5))
        self.m = RandomForestRegressor(n_estimators=200,max_depth=100)
        self.m.fit(X, y)
        self.X = X
        self.y = y


    def predict(self, X):
        #y_mean = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        y_mean = np.asarray(self.m.predict(X))
        preds = []
        for pred in self.m.estimators_:
            preds.append(pred.predict(X))
        preds = np.asarray(preds)
        y_stdev = np.std(preds,axis=0)
        y_mean = (y_mean.reshape(1,-1))
        y_stdev = (y_stdev.reshape(1,-1))
        return (y_mean, y_stdev)

