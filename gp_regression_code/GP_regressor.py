from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import GPy
import numpy as np


class GPRegressor:
    def __init__(self, kernel, save_model_path=None, save_predictions=None, num_restarts=1):
        self.kernel = kernel
        self.save_model_path = save_model_path
        self.num_restarts = num_restarts
        self.save_predictions = save_predictions

    def optimization(self, X, Y):
        noise_var = Y.var() * 0.01
        m = GPy.models.GPRegression(X, Y, self.kernel, noise_var=noise_var)
        m.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        m.optimize_restarts(num_restarts=self.num_restarts)

        if self.save_model_path is not None:
            m.save_model(self.save_model_path)

        return m

    def predict(self, m, x_test):

        y_pred = m.predict(x_test, False, include_likelihood=False)  # ???


        return y_pred[0], y_pred[1]

    def criterion(self, pred, target):
        mse_error = mean_squared_error(target, pred)
        evs_error = explained_variance_score(target, pred)
        r2 = r2_score(target, pred)

        return {'mse': mse_error,
                'evs_error': evs_error,
                'r2': r2}
