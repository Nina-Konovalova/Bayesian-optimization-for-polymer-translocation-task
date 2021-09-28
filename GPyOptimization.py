import GPy
import GPyOpt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from numpy.random import seed
import pandas as pd
from numpy import exp, linspace, random
from scipy.optimize import curve_fit
import subprocess
import math
import matplotlib.pyplot as plt

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(12354)


class BayesianOptimization:
    def __init__(self, model_type, X_true, exp_number=1, kernel=None):
        self.model_type = model_type
        self.exp_number = exp_number
        max_amp = np.max(abs(X_true)) * 100
        self.space = [{'name': 'var_1', 'type': 'continuous', 'domain': (-max_amp, max_amp)},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, 100)},  # 2
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, 10)},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (-max_amp, max_amp)},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (0, 100)},  # 2
                      {'name': 'var_6', 'type': 'continuous', 'domain': (0, 10)},
                      {'name': 'var_7', 'type': 'continuous', 'domain': (-max_amp, max_amp)},
                      {'name': 'var_8', 'type': 'continuous', 'domain': (0, 100)},  # 2
                      {'name': 'var_9', 'type': 'continuous', 'domain': (0, 10)},
                      {'name': 'var_10', 'type': 'continuous', 'domain': (-max_amp, max_amp)},
                      {'name': 'var_11', 'type': 'continuous', 'domain': (0, 100)},  # 2
                      {'name': 'var_12', 'type': 'continuous', 'domain': (0, 10)}]

        if self.model_type == 'GP':
            if kernel is not None:
                self.model = GPyOpt.models.GPModel(kernel, optimize_restarts=3, exact_feval=True)
            else:
                self.model = GPyOpt.models.GPModel(optimize_restarts=3, exact_feval=True)
        elif self.model_type == 'GP_MCMC':
            if kernel is not None:
                self.model = GPyOpt.models.GPModel_MCMC(kernel, exact_feval=True)
            else:
                self.model = GPyOpt.models.GPModel_MCMC(exact_feval=True)
        elif self.model_type == 'InputWarpedGP':
            if kernel is not None:
                print('fuck')
                self.model = GPyOpt.models.WarpedGPModel(GPyOpt.core.task.space.Design_space(self.space), kernel,
                                                         exact_feval=True)  # нужно разобраться со спейсом тут
            else:
                self.model = GPyOpt.models.WarpedGPModel(GPyOpt.core.task.space.Design_space(self.space),
                                                         exact_feval=True)  # нужно разобраться со спейсом тут
        self.points_polymer = np.arange(51)
        self.X_true = X_true
        self.X_param_true = self.fitting_curves(X_true)
        self.Y_pos_real, self.Y_neg_real, self.rate_real = self.probabilities_from_init_distributions(self.X_param_true)

    @staticmethod
    def gaussian(x, amp, cen, wid, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3):
        gauss = amp * 1 / (np.sqrt(wid * 2 * np.pi)) * exp(-(x - cen) ** 2 / (2 * wid)) + \
                amp1 * 1 / (np.sqrt(wid1 * 2 * np.pi)) * exp(-(x - cen1) ** 2 / (2 * wid1)) + \
                amp2 * 1 / (np.sqrt(wid2 * 2 * np.pi)) * exp(-(x - cen2) ** 2 / (2 * wid2)) + \
                amp3 * 1 / (np.sqrt(wid3 * 2 * np.pi)) * exp(-(x - cen3) ** 2 / (2 * wid3))

        return gauss
    # def gaussian(x, params):
    #     amp = []
    #     mu = []
    #     sigma = []
    #     for i in range (len(params)//3):
    #         amp.append(params[3*i])
    #         mu.append(params[3*i+1])
    #         sigma.append(params[3*i+2])
    #     gauss = 0
    #     for i in range(len(amp)):
    #         gauss += amp * 1 / (np.sqrt(sigma[i] * 2 * np.pi)) * exp(-(x - mu[i]) ** 2 / (2 * sigma[i]))
    #
    #     return gauss


    def fitting_curves(self, y, function='gauss'):
        x = linspace(0, 51, 51)

        if function == 'gauss':
            guess = [0.4, x[np.argmax(np.abs(y))], 1, 0.4, x[np.argmax(np.abs(y))], 1, 0.4, x[np.argmax(np.abs(y))], 1,
                     0.4, x[np.argmax(np.abs(y))], 1]
            best_vals, _ = curve_fit(self.gaussian, x, y, maxfev=100000, method='trf', p0=guess)

        return best_vals

    def probabilities_from_init_distributions(self, x_end):
        best_vals = x_end
        y = self.gaussian(self.points_polymer, best_vals[0], best_vals[1], best_vals[2],
                               best_vals[3], best_vals[4], best_vals[5],
                               best_vals[6], best_vals[7], best_vals[8],
                               best_vals[9], best_vals[10], best_vals[11])
        self.make_input_file(y)
        plt.plot(y, 'go--', linewidth=4, markersize=2,
                 color='red', label='parametrized X')

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])
        # saving new output data
        rate, y_pos_new, y_neg_new = self.read_data()

        return y_pos_new, y_neg_new, rate

    @staticmethod
    def function_from_der(y, der):
        y1 = np.copy(y)
        y1[1] = y[0] + der[0]
        for i in range(1, len(y) - 1):
            y1[i + 1] = y1[i - 1] + 2 * der[i]
        return y1

    @staticmethod
    def make_input_file(curv, N=51, t=1, num1=50000):
        f = open('./new_input.txt', 'w')
        f.write(
            str(N - 1) + '\t' + str(t) + '\n' + str(num1) + '\t' + str(t) + '\t' + str(10000) + '\t' + str(t) + '\n')
        for i in range(N):
            f.write(str(i) + '\t' + str(curv[i]) + '\n')
        f.close()

    @staticmethod
    def read_derives(path_to_file='./der_output.txt'):
        derives_dat = pd.read_csv(path_to_file, sep=' ', header=None)
        derives_dat[2][derives_dat[2] == 0] = -1
        for i in range(derives_dat.shape[0]):
            if str(derives_dat[1][i]).find('E') == -1 \
                    and str(derives_dat[1][i]).find('e') == -1 \
                    and str(derives_dat[1][i]).find('-') != -1 \
                    and str(derives_dat[1][i]).find('nan') == -1:
                derives_dat[1][i] = (str(derives_dat[1][i]).split('-')[0]) + 'E' + \
                                    '-' + (str(derives_dat[1][i]).split('-')[1])
        derives = np.array(np.array(derives_dat[1]).astype('float32') * derives_dat[2]).astype('float32')
        return derives

    @staticmethod
    def read_data(path_to_file='./new_output.txt'):
        dat = pd.read_csv(path_to_file, sep=' ', skiprows=[0, 1, 2], header=None)
        dat.drop(dat.columns[0], axis=1, inplace=True)
        dat.fillna(1e+20, inplace=True)
        r = pd.read_csv('./new_output.txt', sep=' ', nrows=1, header=None)
        r.fillna(1e+20, inplace=True)
        rate = np.array(r)[0][11]
        rate = float(rate)

        y_pos_new = np.array(dat[1][:], dtype=float)
        y_neg_new = np.array(dat[2][:], dtype=float)

        return rate, y_pos_new, y_neg_new


    def fokker_plank_eq(self, x_end):
        eps = 10e-18
        best_vals = x_end[0]
        #print(best_vals)
        new_curv = self.gaussian(self.points_polymer, best_vals[0], best_vals[1], best_vals[2] + eps,
                                                      best_vals[3], best_vals[4], best_vals[5] + eps,
                                                      best_vals[6], best_vals[7], best_vals[8] + eps,
                                                      best_vals[9], best_vals[10], best_vals[11]+ eps)
        # print(new_curv[0])
        # if round(new_curv[0], 3) != 0:
        #     print('moving a little bit')
        #     new_curv = new_curv - new_curv[0]
        #     x_end = self.fitting_curves(new_curv, function='gauss')

        self.make_input_file(new_curv)

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])

        # checking derives of function
        old_derives = self.read_derives()
        new_der = np.clip(old_derives, -4, 4)
        new_curv_1 = self.function_from_der(new_curv, new_der)
        #print(new_curv)
        if not np.allclose(new_curv, new_curv_1):
            x_end = self.fitting_curves(new_curv_1, function='gauss')
            self.make_input_file(new_curv_1)
            subprocess.check_output(["./outputic"])
        rate, y_pos_new, y_neg_new = self.read_data()

        if rate == 1e20:
            print('fuckup')
            return 1e20

        # mse for minimization
        loss_true = mse((y_pos_new[:]), (self.Y_pos_real[:]))
        loss_false = mse((y_neg_new[:]), (self.Y_neg_real[:]))
        loss_rate = abs(rate - self.rate_real)
        loss_rate *= 10 ** (-int(math.log((2 * loss_rate) / (loss_true + loss_false), 10)))
        if rate > 1:
            print('fuckup', x_end)
        #print(rate, self.rate_real)
        diff_new = loss_false + loss_true + loss_rate
        return diff_new

    def optimization_step(self, x_parametr_pol, y_train, num_steps, path_for_save, acquisition_type='MPI',
                          normalize=True, num_cores=-1, evaluator_type='lbfgs'):

        myBopt = GPyOpt.methods.BayesianOptimization(f=self.fokker_plank_eq,  # function to optimize
                                                     domain=self.space,  # box-constraints of the problem
                                                     model=self.model,
                                                     Model_type=self.model_type,
                                                     X=x_parametr_pol,
                                                     verbosity=False,
                                                     normalize_Y=normalize,
                                                     evaluator_type=evaluator_type,
                                                     num_cores=num_cores,
                                                     acquisition_type=acquisition_type)
        # print('model', myBopt.model.model)
        print('optimization starts')
        myBopt.run_optimization(num_steps, report_file = path_for_save + 'report_file_' + str(self.exp_number) + '.txt',
                                models_file = path_for_save + 'model_params_' + str(self.exp_number) + '.txt')

        best_vals = myBopt.x_opt
        print(myBopt.model.model)
        plt.figure(figsize=(16, 12))
        plt.plot(self.points_polymer, self.X_true, label='real X')
        plt.plot(self.points_polymer, self.gaussian(self.points_polymer, best_vals[0], best_vals[1], best_vals[2],
                                  best_vals[3], best_vals[4], best_vals[5],
                                  best_vals[6], best_vals[7], best_vals[8],
                                  best_vals[9], best_vals[10], best_vals[11]), 'go--', linewidth=4, markersize=2,
                 color='red', label='predicted')
        plt.legend()
        path_for_save = path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(path_for_save + '.png')
        # npz.save('model.npz', model = myBopt.model)
        return best_vals