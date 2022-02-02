from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import GPy


class GPRegressor:
    def __init__(self, kernel, save_model_path=None, save_predictions=None, num_restarts=1):
        '''
        :param kernel: kernel that is used for Gaussian processes
        :param save_model_path: where model's parameters should be saved
        :param save_predictions:
        :param num_restarts: number of resturts to approximate function
        '''
        self.kernel = kernel
        self.save_model_path = save_model_path
        self.num_restarts = num_restarts
        self.save_predictions = save_predictions

    def optimization(self, X, Y):
        '''
        :param X: data
        :param Y: targets
        :return: GP approximation model
        '''
        m = GPy.models.GPRegression(X, Y, self.kernel)
        m.Gaussian_noise.variance = 1e-12
        m.Gaussian_noise.variance.constrain_fixed() # fixe noise as now we use noiseless evaluation
        m.optimize_restarts(num_restarts=self.num_restarts)
        if self.save_model_path is not None:
            m.save_model(self.save_model_path)
        return m

    def predict(self, m, x_test):
        '''
        :param m: model - parameters of GP
        :param x_test: data to predict
        :return: mean value for prediction, variance for the prediction
        '''
        y_pred = m.predict(x_test, include_likelihood=False)  # This doesn't includes the likelihood variance added to the predicted underlying function
        return y_pred

    def criterion(self, pred, target):
        '''
        :param pred: prediction from the model
        :param target: real value
        :return: dict of 3 main metrics: mse, r2, explained_variance_score
        '''
        mse_error = mean_squared_error(target, pred)
        evs_error = explained_variance_score(target, pred)
        r2 = r2_score(target, pred)

        return {'mse': mse_error,
                'evs_error': evs_error,
                'r2': r2}
