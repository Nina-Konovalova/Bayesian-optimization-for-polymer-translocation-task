import GPy
import GPyOpt
from sklearn.metrics import mean_squared_error as mse
from numpy.random import seed
from utils.gauss_fit import *
from utils.data_frotran_utils import *
import subprocess
import matplotlib.pyplot as plt
import Configurations.Config as CFG
import os
import json

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(42)


class BayesianOptimization:
    def __init__(self, model_type, x_e, exp_number, kernel):
        self.model_type = model_type
        self.points_polymer = np.arange(CFG.MONOMERS)
        self.exp_number = exp_number
        self.space = CFG.SPACE[CFG.NUM_GAUSS]
        self.constraints = CFG.CONSTRAINTS[CFG.NUM_GAUSS]
        self.opt = False
        kernel = kernel
        print(kernel)
        self.good_step_have_left = 0
        self.exp_number = exp_number

        self.x_real = x_e['vecs'][exp_number]
        self.X_true = gaussian(self.points_polymer, *self.x_real)

        self.rate_real = x_e['rates'][exp_number]
        self.angs_real = x_e['angs'][exp_number]

        self.ang_pos_real = x_e['angs'][exp_number][0]
        self.ang_neg_real = x_e['angs'][exp_number][1]

        self.Y_pos_real = x_e['y_pos'][exp_number]
        self.Y_neg_real = x_e['y_neg'][exp_number]

        self.opt_steps = {'vecs': [],
                          'loss': []}

        print("Rate, angs_pos, ang_neg", self.rate_real, self.ang_pos_real, self.ang_neg_real)

        if kernel == 'Matern52':
            kernel = GPy.kern.Matern52(CFG.NUM_GAUSS)
        elif kernel == 'Matern32':
            kernel = GPy.kern.Matern32(CFG.NUM_GAUSS)
        elif kernel == 'RBF':
            kernel = GPy.kern.RBF(CFG.NUM_GAUSS)
        elif kernel == 'ExpQuad':
            kernel == GPy.kern.ExpQuad(CFG.NUM_GAUSS)
        elif kernel == 'RatQuad':
            kernel = GPy.kern.RatQuad(CFG.NUM_GAUSS)
        else:
            raise KeyError('Invalid kernel type')

        if self.model_type == 'GP':
            if kernel is not None:
                self.model = GPyOpt.models.GPModel(kernel, optimize_restarts=1, exact_feval=True)
            else:
                self.model = GPyOpt.models.GPModel(optimize_restarts=1, exact_feval=True)
        elif self.model_type == 'GP_MCMC':
            if kernel is not None:
                self.model = GPyOpt.models.GPModel_MCMC(kernel, exact_feval=True)
            else:
                self.model = GPyOpt.models.GPModel_MCMC(exact_feval=True)


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
    def function(rate_real, angs_real, y_pos_real, y_neg_real, rate_new, angs_new, y_pos_new, y_neg_new):
        loss_true = mse((y_pos_new[:]), (y_pos_real))
        loss_false = mse((y_neg_new[:]), (y_neg_real[:]))
        loss_mse = loss_true + loss_false
        loss_rate = abs(rate_new - rate_real)
        loss_angs = (abs(angs_new - angs_real)).sum()
        diff_new = loss_angs + loss_rate * CFG.ALPHA

        return diff_new, loss_rate, loss_angs, loss_mse

    def make_data(self, rate_real, angs_real, y_pos_real, y_neg_real, rate, angs, y_pos, y_neg):
        f = []
        for i in (range(len(rate))):
            f.append(self.function(rate_real, angs_real, y_pos_real, y_neg_real, rate[i], angs[i], y_pos[i], y_neg[i]))
        return np.array(f)

    @staticmethod
    def angle(y_pos_new, y_neg_new):
        xx = np.arange(len(y_pos_new))
        t_pos = np.polyfit(xx, np.log(y_pos_new), 1)
        t_neg = np.polyfit(xx, np.log(y_neg_new), 1)
        return [t_pos[0], t_neg[0]]

    def fokker_plank_eq(self, x_end):
        problem = False
        eps = 10e-18
        best_vals = x_end[0]
        new_curv = gaussian(self.points_polymer, *best_vals)
        make_input_file(new_curv)
        subprocess.check_output(["./outputic"])
        i = 0
        old_der = read_derives()
        if (abs(old_der) > 4.).sum() != 0:
            print('too big derives')
            problem = True
            try:
                os.mkdir(self.path_for_save + "out_of_space/")
            except:
                pass
            np.save(self.path_for_save + "out_of_space/" + str(x_end[0][0]) + '.npy', x_end)

        rate, y_pos_new, y_neg_new = read_data()

        if rate == 1e20 or rate == 0:
            print('Out of space')
            problem = True
            old_der = read_derives()
            print((old_der > 4).sum())
            try:
                os.mkdir(self.path_for_save + "big_rate/")
            except:
                pass
            np.save(self.path_for_save + "big_rate/" + str(x_end[0][0]) + '.npy', x_end)


        if rate > 1:
            print('Rate bigger 1', x_end)
            problem = True
            try:
                os.mkdir(self.path_for_save + "bigger_1/")
            except:
                pass
            np.save(self.path_for_save + "bigger_1/" + "bigger_1_" + change + "_" + str(x_end[0][0]) + '.npy',
                    x_end)

        angs = self.angle(y_pos_new, y_neg_new)
        diff_new, loss_rate, loss_angs, loss_mse = self.function(self.rate_real, self.angs_real,
                                                                 self.y_pos_real, self.y_neg_real, rate,
                                                                 angs, y_pos_new, y_neg_new)
        print('rate_new', rate, 'rate_real', self.rate_real, 'differ', diff_new, 'loss angs', loss_angs)
        res = { 'func_val': diff_new,
                'loss_rate': loss_rate,
                'mse_loss': loss_mse,
                'angle_pos_diff': abs(angs[0] - self.ang_pos_real),
                'angle_neg_diff': abs(angs[1] - self.ang_neg_real)
        }
        if not problem:
            if self.opt:
                self.good_step_have_left += 1

                self.opt_steps['vecs'].append(x_end.tolist())
                self.opt_steps['loss'].append(diff_new)
                plt.figure(figsize=(16, 12))
                plt.plot(new_curv)
                try:
                    os.mkdir(self.path_for_save + 'optimization_pics/')
                except:
                    pass
                plt.savefig(self.path_for_save + 'optimization_pics/' + 'pic' + str(self.good_step_have_left) + '.jpg')
                self.data_out = self.data_out.append(res, ignore_index=True)
                self.data_out.to_csv(self.path_for_save + 'results_compare_' + str(self.rate_real) + '.csv', index=False)
        print('good steps have left', self.good_step_have_left, 'from', CFG.NUM_STEPS)
        return diff_new, x_end, problem


    def optimization_step(self, x_data_path, num_steps, path_for_save, acquisition_type='EI',
                          normalize=False, num_cores=-1, evaluator_type='lbfgs'):

        d_train = np.load(x_data_path)
        vecs_t, rates_t, angs_t, y_pos_t, y_neg_t = d_train['vecs'], d_train['rates'], d_train['angs'], \
                                                    d_train['y_pos'], d_train['y_neg']
        y_train = self.make_data(self.rate_real, self.angs_real, self.Y_pos_real,
                                 self.Y_neg_real, rates_t, angs_t, y_pos_t, y_neg_t)
        self.opt = False
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.fokker_plank_eq,  # function to optimize
                                                     domain=self.space,
                                                     constraints=self.constraints,  # box-constraints of the problem
                                                     model=self.model,
                                                     Model_type=self.model_type,
                                                     X=vecs_t,
                                                     Y=y_train.reshape(-1,1),
                                                     verbosity=False,
                                                     normalize_Y=False,
                                                     num_cores=1,
                                                     acquisition_type=acquisition_type)

        # print('model', myBopt.model.model)
        print('optimization starts')
        self.path_for_save = path_for_save
        self.data_out = pd.DataFrame(columns=['rate_rate', 'loss_rate',
                                              'mse_pos', 'mse_neg',
                                              'angle_pos_rate', 'angle_neg_rate'
                                              'angle_pos_diff', 'angle_neg_diff'])
        self.opt = True
        myBopt.run_optimization(num_steps, report_file=path_for_save + 'report_file_' + str(self.exp_number) + '.txt',
                                models_file=path_for_save + 'model_params_' + str(self.exp_number) + '.txt')

        best_vals = myBopt.x_opt
        best_vals_opt = np.array(self.opt_steps['vecs'])[np.argmin(np.array(self.opt_steps['loss']))]
        with open(path_for_save + 'predicted_data.json', 'w',
                  encoding='utf-8') as f:
            json.dump({'predictions_best': best_vals.tolist(), 'prediction_from_opt': best_vals_opt.tolist(),
                       'real': self.x_real.tolist(), 'all_way': self.opt_steps}, f, sort_keys=True, indent=4)
        print(myBopt.model.model)
        plt.figure(figsize=(16, 12))
        plt.plot(self.points_polymer, self.X_true, label='real X')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, *best_vals), 'go--', linewidth=4,
                 markersize=2,
                 color='red', label='predicted')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, *best_vals_opt[0]), 'go--', linewidth=4,
                 markersize=2,
                 color='green', label='predicted only from opt')
        plt.legend()
        path_for_save = path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(path_for_save + '.png')
        return best_vals
