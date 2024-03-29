import os
import sys

import GPy.util.warping_functions
from tqdm import tqdm
import GPyOpt
from GPyOpt.core.task.space import Design_space
import Configurations.config_mass as cfg_mass
import Configurations.Config as CFG
from scipy.stats import gamma
sys.path.append('/')
from utils.gauss_fit import *
from utils.data_frotran_utils import *
# from utils.help_functions import *
import subprocess
import matplotlib.pyplot as plt
from utils.utils_mass import *
import json

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(42)


class BayesianOptimizationMass:
    def __init__(self, model_type, x_e, exp_number, path_for_save, save_opt_plots, num_gamma=2):

        self.good_steps_have_left = 0
        self.exp_number = exp_number
        self.save_opt_plots = save_opt_plots

        # save csv with info
        self.path_for_save = path_for_save
        self.data_out = pd.DataFrame(columns=['shape', 'scale',
                                              'all_samples_distributions_sum_real', 'all_samples_distributions_sum_pred',
                                              'diff'])
        if self.save_opt_plots:
            try:
                os.mkdir(self.path_for_save + 'optimization_pics/')
                print(f'make dir {self.path_for_save + "optimization_pics/"} to save opt plots')
            except:
                pass

        # data
        if num_gamma == 1:
            self.x_real = np.array([x_e['shape'][exp_number], x_e['scale'][exp_number]])
            self.x_true = gamma.pdf(cfg_mass.X, self.x_real[0], scale=self.x_real[1])
        elif num_gamma == 2:
            self.x_real = np.array([x_e['shape'][0][exp_number], x_e['scale'][0][exp_number],
                                    x_e['shape'][1][exp_number], x_e['scale'][1][exp_number]])
            self.x_true = gamma.pdf(cfg_mass.X, x_e['shape'][0][exp_number], scale=x_e['scale'][0][exp_number]) +\
                          gamma.pdf(cfg_mass.X,  x_e['shape'][1][exp_number], scale=x_e['scale'][1][exp_number])
        elif num_gamma == 3:
            self.x_real = np.array([x_e['shape'][0][exp_number], x_e['scale'][0][exp_number],
                                    x_e['shape'][1][exp_number], x_e['scale'][1][exp_number],
                                    x_e['shape'][2][exp_number], x_e['scale'][2][exp_number]])
            self.x_true = gamma.pdf(cfg_mass.X, x_e['shape'][0][exp_number], scale=x_e['scale'][0][exp_number]) +\
                          gamma.pdf(cfg_mass.X,  x_e['shape'][1][exp_number], scale=x_e['scale'][1][exp_number]) + \
                          gamma.pdf(cfg_mass.X,  x_e['shape'][2][exp_number], scale=x_e['scale'][2][exp_number])

        self.y_real = x_e['all_samples_distributions_sum'][exp_number]
        self.num_gamma = num_gamma

        if not os.path.exists('../time_distributions/time_distributions_longer.npz'):
            prepare_distributions('something')

        distributions = np.load('time_distributions/time_distributions_longer.npz')
        self.d = distributions['time_distributions']
        if self.num_gamma == 1:
            print(f"Shape {self.x_real[0]}, scale {self.x_real[1]}")
        elif self.num_gamma == 2:
            print(f"Shape {[self.x_real[0], self.x_real[2]]}, scale {[self.x_real[1], self.x_real[3]]}")
        elif self.num_gamma == 3:
            print(f"Shape {[self.x_real[0], self.x_real[2], self.x_real[4]]}, scale {[self.x_real[1], self.x_real[3], self.x_real[5]]}")
        print(f"all samples distributions sum {self.y_real.shape}")

        # model parameters
        self.model_type = model_type
        self.exp_number = exp_number
        self.space = cfg_mass.SPACE
        # self.constraints = cfg_mass.CONSTRAINTS[cfg_mass.NUM_GAUSS]
        self.opt = False

        if self.model_type == 'GP':
            self.model = GPyOpt.models.GPModel(cfg_mass.KERNEL, optimize_restarts=1, exact_feval=True)
        elif self.model_type == 'GP_MCMC':
            self.model = GPyOpt.models.GPModel_MCMC(cfg_mass.KERNEL, exact_feval=True)
        elif self.model_type == 'InputWarpedGP':
            self.model = GPyOpt.models.input_warped_gpmodel.InputWarpedGPModel(space=Design_space(self.space),
                                                                               #warping_function=GPy.util.warping_functions.LogFunction(),
                                                                               #warping_function=GPy.util.warping_functions.TanhFunction(),
                                                                               kernel=cfg_mass.KERNEL, exact_feval=True)
        else:
            raise ValueError('no such type of model implemented')

    @staticmethod
    def plots(g, sample, path_to_save):
        shape = sample[0]
        scale = sample[1]
        plt.figure(figsize=(16, 12))
        plt.scatter(cfg_mass.X, g, marker='+', color='green')
        plt.title(f'shape {shape}, scale {scale}')
        plt.savefig(path_to_save + 'sample_' + str(round(shape, 2)) + '_' + str(round(scale, 2)) + '.jpg')
        plt.close()

    def fokker_plank_eq(self, x_end):
        x_end = x_end[0]
        problem = False
        if self.num_gamma == 1:
            shape = x_end[0]
            scale = x_end[1]
            g = gamma.pdf(cfg_mass.X, shape, scale=scale)
        elif self.num_gamma == 2:
            shape = np.array([x_end[0], x_end[2]])
            scale = np.array([x_end[1], x_end[3]])
            g = gamma.pdf(cfg_mass.X, shape[0], scale=scale[0])
            g += gamma.pdf(cfg_mass.X, shape[1], scale=scale[1])
        elif self.num_gamma == 3:
            shape = np.array([x_end[0], x_end[2], x_end[4]])
            scale = np.array([x_end[1], x_end[3], x_end[5]])
            g = gamma.pdf(cfg_mass.X, shape[0], scale=scale[0])
            g += gamma.pdf(cfg_mass.X, shape[1], scale=scale[1])
            g += gamma.pdf(cfg_mass.X, shape[2], scale=scale[2])

        final_time_d_array = []
        for i, sample in tqdm(enumerate(cfg_mass.X)):

            if np.isnan(self.d[i]).sum() != 0:
                problem = True
                break
            final_time_d_array.append(self.d[i] * g[i])

        if not problem:

            final_value = np.array(final_time_d_array).sum(axis=0)
            # if self.save_opt_plots:
            #     self.plots(g, x_end, self.path_for_save)
            diff_new = make_data(self.y_real, [final_value])[0]
            res = {'shape': shape,
                   'scale': scale,
                   'all_samples_distributions_sum_real': self.y_real,
                   'all_samples_distributions_sum_pred': final_value,
                   'diff': diff_new
                   }
            if self.opt:
                self.good_steps_have_left += 1
                self.opt_steps['vecs'].append(x_end.tolist())
                self.opt_steps['loss'].append(diff_new)



            self.data_out = self.data_out.append(res, ignore_index=True)

            print(f'step {self.good_steps_have_left}: scale - {scale}, shape - {shape}, loss - {diff_new}')
        print('good steps have left', self.good_steps_have_left, 'from', CFG.NUM_STEPS)
        return (diff_new), x_end, problem

    def optimization_step(self, x_data_path, num_steps, acquisition_type='EI',
                          normalize=False, num_cores=-1, evaluator_type='lbfgs'):

        d_train = np.load(x_data_path)
        shape_train, scale_train, all_samples_distributions_sum_train = \
            d_train['shape'], d_train['scale'], d_train['all_samples_distributions_sum_1']

        if self.num_gamma == 1:
            y_train = make_data(self.y_real, all_samples_distributions_sum_train)[:80]
            x_train = np.concatenate((shape_train.reshape(-1, 1), scale_train.reshape(-1, 1)), axis=1)[:80]
        elif self.num_gamma == 2:
            x_train = np.concatenate((shape_train[0,:].reshape(-1, 1), scale_train[0,:].reshape(-1, 1),
                                      shape_train[1,:].reshape(-1, 1), scale_train[1,:].reshape(-1, 1)), axis=1)
            y_train = make_data(self.y_real, all_samples_distributions_sum_train)
        elif self.num_gamma == 3:
            x_train = np.concatenate((shape_train[0, :].reshape(-1, 1), scale_train[0, :].reshape(-1, 1),
                                      shape_train[1, :].reshape(-1, 1), scale_train[1, :].reshape(-1, 1),
                                      shape_train[2, :].reshape(-1, 1), scale_train[2, :].reshape(-1, 1)), axis=1)
            y_train = make_data(self.y_real, all_samples_distributions_sum_train)

        self.opt = False
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.fokker_plank_eq,  # function to optimize
                                                     domain=self.space,
                                                     #constraints=self.constraints,  # box-constraints of the problem
                                                     model=self.model,
                                                     Model_type=self.model_type,
                                                     X=x_train,
                                                     Y=y_train.reshape(-1, 1),
                                                     verbosity=False,
                                                     normalize_Y=False,
                                                     num_cores=1,
                                                     acquisition_type='MPI')

        # print('model', myBopt.model.model)
        print('optimization starts')

        self.opt = True
        self.opt_steps = {'vecs': [],
                          'loss': []}
        myBopt.run_optimization(num_steps,
                                report_file=self.path_for_save + 'report_file_' + str(self.exp_number) + '.txt',
                                models_file=self.path_for_save + 'model_params_' + str(self.exp_number) + '.txt')

        best_vals = myBopt.x_opt
        print(len(self.opt_steps['vecs']))
        best_vals_opt = np.array(self.opt_steps['vecs'])[np.argmin(np.array(self.opt_steps['loss']))]
        diff_final_opt = np.min(np.array(self.opt_steps['loss']))
        diff_final = myBopt.fx_opt

        with open(self.path_for_save + 'predicted_data.json', 'w',
                  encoding='utf-8') as f:
            json.dump({'predictions_best': {'vec': best_vals.tolist(), 'loss': diff_final},
                       'prediction_from_opt': {'vec': best_vals_opt.tolist(), 'loss': diff_final_opt},
                       'real': self.x_real.tolist(), 'all_way': self.opt_steps}, f, indent=4)

        print(myBopt.model.model)
        plt.figure(figsize=(16, 12))
        plt.plot(cfg_mass.X, self.x_true, label='real X')
        if self.num_gamma == 1:
            plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals[0], scale=best_vals[1]), 'go--', linewidth=4,
                     markersize=2,
                     color='red', label='predicted')
            plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals_opt[0], scale=best_vals_opt[1]), 'go--', linewidth=4,
                     markersize=2,
                     color='green', label='predicted only from opt')
        elif self.num_gamma == 2:
            plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals[0], scale=best_vals[1]) +
                                 gamma.pdf(cfg_mass.X, best_vals[2], scale=best_vals[3]),
                     'go--', linewidth=4, markersize=2,
                     color='red', label='predicted')
            plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals_opt[0], scale=best_vals_opt[1]) +
                                 gamma.pdf(cfg_mass.X, best_vals_opt[2], scale=best_vals_opt[3]),
                     'go--', linewidth=4, markersize=2,
                     color='green', label='predicted only from opt')
        elif self.num_gamma == 3:
            plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals[0], scale=best_vals[1]) +
                                 gamma.pdf(cfg_mass.X, best_vals[2], scale=best_vals[3]) +
                                 gamma.pdf(cfg_mass.X, best_vals[4], scale=best_vals[5]) ,
                     'go--', linewidth=4, markersize=2,
                     color='red', label='predicted')
            plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals_opt[0], scale=best_vals_opt[1]) +
                                 gamma.pdf(cfg_mass.X, best_vals_opt[2], scale=best_vals_opt[3]) +
                                 gamma.pdf(cfg_mass.X, best_vals_opt[4], scale=best_vals_opt[5]),
                     'go--', linewidth=4, markersize=2,
                     color='green', label='predicted only from opt')

        plt.legend()
        path_for_save = self.path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(path_for_save + '.png')
        return best_vals
