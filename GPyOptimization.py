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
subprocess.call(["gfortran","-o","outputic","F.f90"])
seed(12354)


class BayesianOptimization:
    def __init__(self, model_type, X_true, kernel=None):

        self.model_type = model_type
        max_amp = np.max(abs(X_true))
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

        self.X_param_true = self.fitting_curves(X_true)
        self.Y_pos_real, self.Y_neg_real, self.rate_real = self.probabilities_from_init_distributions(self.X_param_true)

    def gaussian(self, x, amp, cen, wid, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3):
        return amp * exp(-(x - cen) ** 2 / wid) + amp1 * exp(-(x - cen1) ** 2 / wid1) + amp2 * exp(
            -(x - cen2) ** 2 / wid2) + amp3 * exp(-(x - cen3) ** 2 / wid3)

    def fitting_curves(self, y, function='gauss'):
        x = linspace(0, 51, 51)

        if function == 'gauss':
            guess = [0.4, x[np.argmax(np.abs(y))], 1, 0.4, x[np.argmax(np.abs(y))], 1, 0.4, x[np.argmax(np.abs(y))], 1,
                     0.4, x[np.argmax(np.abs(y))], 1]
            best_vals, _ = curve_fit(self.gaussian, x, y, maxfev=100000, method='trf', p0=guess)

        return best_vals

    def probabilities_from_init_distributions(self, x_end):
        f = open('./new_input.txt', 'w')
        N = 51
        t = 1
        num1 = 50000
        f.write(
            str(N - 1) + '\t' + str(t) + '\n' + str(num1) + '\t' + str(t) + '\t' + str(10000) + '\t' + str(t) + '\n')
        x = []
        y = []
        best_vals = x_end
        for i in range(N):
            f.write(str(i) + '\t' + str(self.gaussian(i, best_vals[0], best_vals[1], best_vals[2],
                                                      best_vals[3], best_vals[4], best_vals[5],
                                                      best_vals[6], best_vals[7], best_vals[8],
                                                      best_vals[9], best_vals[10], best_vals[11])) + '\n')
        f.close()

        plt.plot(self.gaussian(np.arange(51), best_vals[0], best_vals[1], best_vals[2],
                               best_vals[3], best_vals[4], best_vals[5],
                               best_vals[6], best_vals[7], best_vals[8],
                               best_vals[9], best_vals[10], best_vals[11]), 'go--', linewidth=4, markersize=2,
                 color='red', label='parametrized X')

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])
        # saving new output data
        dat = pd.read_csv('./new_output.txt', sep=' ', skiprows=[0, 1, 2], header=None)
        dat.drop(dat.columns[0], axis=1, inplace=True)
        rate = float(np.array(pd.read_csv('./new_output.txt', sep=' ', nrows=1, header=None).fillna(1e+20)[11]))

        Y_pos_new = np.array(dat[1][:], dtype=float)
        Y_neg_new = np.array(dat[2][:], dtype=float)

        return Y_pos_new, Y_neg_new, rate

    def Fokker_plank_eq(self, x_end):
        N = 51
        t = 1
        num1 = 50000
        f = open('./new_input.txt', 'w')
        f.write(
            str(N - 1) + '\t' + str(t) + '\n' + str(num1) + '\t' + str(t) + '\t' + str(10000) + '\t' + str(t) + '\n')
        x = []
        y = []
        best_vals = x_end[0]
        for i in range(N):
            f.write(str(i) + '\t' + str(self.gaussian(i, best_vals[0], best_vals[1], best_vals[2],
                                                      best_vals[3], best_vals[4], best_vals[5],
                                                      best_vals[6], best_vals[7], best_vals[8],
                                                      best_vals[9], best_vals[10], best_vals[11])) + '\n')
        f.close()

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])
        # saving new output data
        dat = pd.read_csv('./new_output.txt', sep=' ', skiprows=[0, 1, 2], header=None)
        dat.drop(dat.columns[0], axis=1, inplace=True)
        dat.fillna(1e+20, inplace=True)
        r = pd.read_csv('./new_output.txt', sep=' ', nrows=1, header=None)
        r.fillna(1e+20, inplace=True)
        rate = np.array(r)[0][11]
        rate = float(rate)

        rate = float(rate)

        Y_pos_new = np.array(dat[1][:], dtype=float)
        Y_neg_new = np.array(dat[2][:], dtype=float)

        if rate == 1e20:
            return 1e20

        if rate < 1e-7:
            return 1e20

        # mse for minimization
        loss_true = mse((Y_pos_new[:]), (self.Y_pos_real[:]))
        loss_false = mse((Y_neg_new[:]), (self.Y_neg_real[:]))
        loss_rate = abs(rate - self.rate_real)
        loss_rate *= 10 ** (-int(math.log((2 * loss_rate) / (loss_true + loss_false), 10)))
        print(rate, self.rate_real)
        diff_new = loss_false + loss_true + loss_rate
        return diff_new

    def optimization_step(self, x_parametr_pol, y_train, num_steps, path_for_save, acquisition_type='MPI',
                          normalize=True, num_cores=1, evaluator_type='lbfgs'):

        myBopt = GPyOpt.methods.BayesianOptimization(f=self.Fokker_plank_eq,  # function to optimize
                                                     domain=self.space,  # box-constraints of the problem
                                                     model=self.model,
                                                     Model_type='WarpedGP',
                                                     X=x_parametr_pol,
                                                     Y=y_train,
                                                     verbosity=False,
                                                     normalize_Y=normalize,
                                                     evaluator_type=evaluator_type,
                                                     num_cores=num_cores,
                                                     acquisition_type=acquisition_type)
        # print('model', myBopt.model.model)
        myBopt.run_optimization(num_steps)

        best_vals = myBopt.x_opt
        print(myBopt.model.model)
        plt.figure(figsize=(16, 12))
        plt.plot(np.arange(51), X_e[i], label='real X')
        plt.plot(x, self.gaussian(x, best_vals[0], best_vals[1], best_vals[2],
                                  best_vals[3], best_vals[4], best_vals[5],
                                  best_vals[6], best_vals[7], best_vals[8],
                                  best_vals[9], best_vals[10], best_vals[11]), 'go--', linewidth=4, markersize=2,
                 color='red', label='predicted')
        plt.legend()

        plt.savefig(path_for_save + '.png')
        # npz.save('model.npz', model = myBopt.model)
        return best_vals