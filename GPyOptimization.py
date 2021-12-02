import GPy
import GPyOpt
from sklearn.metrics import mean_squared_error as mse
from numpy.random import seed
from utils.gauss_fit import *
from utils.data_frotran_utils import *
import subprocess
import math
import matplotlib.pyplot as plt
import Configurations.Config as CFG

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(12354)


class BayesianOptimization:
    def __init__(self, model_type, X_true, exp_number, kernel):
        self.model_type = model_type
        self.points_polymer = np.arange(CFG.MONOMERS)
        self.exp_number = exp_number
        self.space = CFG.SPACE
        self.constraints = CFG.CONSTRAINTS
        kernel = kernel

        if kernel == 'Matern52':
            kernel = GPy.kern.Matern52(1)
        elif kernel == 'Matern32':
            kernel = GPy.kern.Matern32(1)
        elif kernel == 'RBF':
            kernel = GPy.kern.RBF(1)
        elif kernel == 'ExpQuad':
            kernel == GPy.kern.ExpQuad(1)
        elif kernel == 'RatQuad':
            kernel = GPy.kern.RatQuad(1)
        else:
            raise KeyError('Invalid kernel type')

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
                self.model = GPyOpt.models.WarpedGPModel(GPyOpt.core.task.space.Design_space(self.space), kernel,
                                                         exact_feval=True)  # нужно разобраться со спейсом тут
            else:
                self.model = GPyOpt.models.WarpedGPModel(GPyOpt.core.task.space.Design_space(self.space),
                                                         exact_feval=True)  # нужно разобраться со спейсом тут

        self.X_true = X_true
        self.X_param_true = fitting_curves(X_true)

        self.Y_pos_real, self.Y_neg_real, self.rate_real = self.probabilities_from_init_distributions(self.X_param_true)

    def probabilities_from_init_distributions(self, x_end):
        best_vals = x_end
        y = gaussian(self.points_polymer, *best_vals)
        make_input_file(y)
        plt.plot(y, 'go--', linewidth=4, markersize=2,
                 color='red', label='parametrized X')

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])
        # saving new output data
        rate, y_pos_new, y_neg_new = read_data()
        return y_pos_new, y_neg_new, rate

    @staticmethod
    def function_from_der(y, der):
        y1 = np.copy(y)
        y1[1] = y[0] + der[0]
        for i in range(1, len(y) - 1):
            y1[i + 1] = y1[i - 1] + 2 * der[i]
        return y1

    def fokker_plank_eq(self, x_end):
        problem = False
        eps = 10e-18
        best_vals = x_end[0]
        new_curv = gaussian(self.points_polymer, *best_vals)

        make_input_file(new_curv)

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])

        # checking derives of function
        change = "not_change"
        i = 0

        old_der = read_derives()
        if (abs(old_der) > 4.).sum() != 0:
            problem = True
            np.save(self.path_for_save + "out_of_space/" + str(x_end[0][0]) + '.npy', x_end)
            plt.figure(figsize=(16, 12))
            plt.plot(self.points_polymer, gaussian(self.points_polymer, *best_vals), 'go--',
                     linewidth=4,
                     markersize=2,
                     color='red', label='predicted')
            plt.legend()
            path_for_save = self.path_for_save + "out_of_space/" + str(x_end[0][0])
            plt.savefig(path_for_save + '.png')

        rate, y_pos_new, y_neg_new = read_data()

        if rate == 1e20:
            print('Out of space')
            problem = True
            old_der = read_derives()
            print((old_der > 4).sum())
            np.save(self.path_for_save + "big_rate/" + str(x_end[0][0]) + '.npy', x_end)
            plt.figure(figsize=(16, 12))
            plt.plot(self.points_polymer, gaussian(self.points_polymer, *best_vals), 'go--',
                     linewidth=4,
                     markersize=2,
                     color='red', label='predicted')
            plt.legend()
            path_for_save = self.path_for_save + "big_rate/" + str(x_end[0][0])
            plt.savefig(path_for_save + '.png')

            return 1e20, x_end, problem

        if rate > 1:
            print('Rate bigger 1', x_end)
            problem = True
            np.save(self.path_for_save + "bigger_1/" + "bigger_1_" + change + "_" + str(x_end[0][0]) + '.npy',
                    x_end)
            plt.figure(figsize=(16, 12))
            plt.plot(self.points_polymer, gaussian(self.points_polymer, *best_vals), 'go--',
                     linewidth=4,
                     markersize=2,
                     color='red', label='predicted')
            plt.legend()
            path_for_save = self.path_for_save + "bigger_1/" + str(x_end[0][0])
            plt.savefig(path_for_save + '.png')

        # mse for minimization
        loss_true = mse((y_pos_new[:]), (self.Y_pos_real[:]))
        loss_false = mse((y_neg_new[:]), (self.Y_neg_real[:]))
        loss_rate = abs(rate - self.rate_real)
        loss_rate *= 10 ** (-int(math.log((2 * loss_rate) / (loss_true + loss_false), 10)))

        # print(rate, self.rate_real)
        diff_new = loss_false + loss_true + loss_rate
        return diff_new, x_end, problem

    def optimization_step(self, x_parametr_pol, num_steps, path_for_save, acquisition_type='MPI',
                          normalize=False, num_cores=-1, evaluator_type='lbfgs'):
        self.path_for_save = path_for_save
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.fokker_plank_eq,  # function to optimize
                                                     domain=self.space,
                                                     constraints=self.constraints,  # box-constraints of the problem
                                                     model=self.model,
                                                     Model_type=self.model_type,
                                                     X=x_parametr_pol,
                                                     verbosity=False,
                                                     normalize_Y=normalize,
                                                     evaluator_type=evaluator_type,
                                                     num_cores=1,
                                                     acquisition_type=acquisition_type)

        # print('model', myBopt.model.model)
        print('optimization starts')
        myBopt.run_optimization(num_steps, report_file=path_for_save + 'report_file_' + str(self.exp_number) + '.txt',
                                models_file=path_for_save + 'model_params_' + str(self.exp_number) + '.txt')

        best_vals = myBopt.x_opt
        print(myBopt.model.model)
        plt.figure(figsize=(16, 12))
        plt.plot(self.points_polymer, self.X_true, label='real X')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, *best_vals), 'go--', linewidth=4,
                 markersize=2,
                 color='red', label='predicted')
        plt.legend()
        path_for_save = path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(path_for_save + '.png')
        return best_vals
