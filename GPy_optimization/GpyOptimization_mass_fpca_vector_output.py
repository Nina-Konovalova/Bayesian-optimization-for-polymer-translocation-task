import os
import sys
import Configurations.config_mass as cfg_mass
import Configurations.Config as CFG
from scipy.stats import gamma
sys.path.append('/')
from utils.data_frotran_utils import *
from GPy.models import GPRegression

from emukit.model_wrappers import GPyModelWrapper
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from utils.utils_mass import *
import json
from target_vector_estimation.l2_bayes_opt.acquisitions import (
    L2NegativeLowerConfidenceBound as L2_LCB,
    L2ExpectedImprovement as L2_EI)
from skfda.representation.basis import BSpline, Fourier, Monomial

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(42)


class BayesianOptimizationMassFunctionalOutput:
    def __init__(self, x_e, exp_number, path_for_save, save_opt_plots, num_gamma=2):

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
        else:
            self.x_real = np.array([x_e['shape'][exp_number], x_e['scale'][exp_number]])
            self.x_true = gamma.pdf(cfg_mass.X, self.x_real[0], scale=self.x_real[1])

        self.y_real = x_e['all_samples_distributions_sum'][exp_number]

        if not os.path.exists('../time_distributions/time_distributions.npz'):
            prepare_distributions()

        distributions = np.load('time_distributions/time_distributions.npz')
        self.d = distributions['time_distributions']

        print(f"Shape {self.x_real[0]}, scale {self.x_real[1]}")
        print(f"all samples distributions sum {self.y_real.shape}")

        # model parameters

        self.exp_number = exp_number
        self.space = cfg_mass.SPACE
        # self.constraints = cfg_mass.CONSTRAINTS[cfg_mass.NUM_GAUSS]
        self.opt = False
        self.target = np.array([0, 0, 0])

    @staticmethod
    def find_const_for_mult(all_samples_distributions_sum_real, all_samples_distributions_sum_train):
        c = []
        for i in (range(len(all_samples_distributions_sum_train))):
            f1 = mse(np.log(all_samples_distributions_sum_real)[100:],
                     np.log(all_samples_distributions_sum_train[i])[100:])
            f2 = mse(all_samples_distributions_sum_real[:100], all_samples_distributions_sum_train[i][:100])
            c.append(f1/f2)
        return np.array(c).mean()

    @staticmethod
    def plots(g, sample, path_to_save):
        shape = sample[0]
        scale = sample[1]
        plt.figure(figsize=(16, 12))
        plt.scatter(cfg_mass.X, g, marker='+', color='green')
        plt.title(f'shape {shape}, scale {scale}')
        plt.savefig(path_to_save + 'sample_' + str(round(shape, 2)) + '_' + str(round(scale, 2)) + '.jpg')
        plt.close()

    @staticmethod
    def fpca(data):
        data = FDataGrid(np.log(data), np.arange(len(np.log(data)[0])),
                         dataset_name='time_distribution',
                         argument_names=['t'],
                         coordinate_names=['p(t)'])
        #basis_fd = fd.to_basis(BSpline(n_basis=7))
        fpca_discretized = FPCA(n_components=2, components_basis=Fourier(n_basis=20),  centering=True)
        #fpca_discretized = FPCA(n_components=3, centering=True)
        fpca_discretized.fit(data)
        return fpca_discretized

    def fokker_plank_eq(self, x_end):
        x_end = x_end[0]
        problem = False
        shape = x_end[0]
        scale = x_end[1]

        g = gamma.pdf(cfg_mass.X, shape, scale=scale)
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
            self.data = np.concatenate((self.data, final_value.reshape(1, -1)), axis=0)
            self.fpca_discretized = fpca(self.data)
            diff_new = make_data_vector_fpca_output(self.data, self.fpca_discretized, self.const)[-1]
            res = {'shape': shape,
                   'scale': scale,
                   'all_samples_distributions_sum_real': self.y_real,
                   'all_samples_distributions_sum_pred': final_value,
                   'diff': diff_new
                   }
            if self.opt:
                self.good_steps_have_left += 1
                self.opt_steps['vecs'].append(x_end.tolist())
                self.opt_steps['loss'].append(diff_new.tolist())

            self.data_out = self.data_out.append(res, ignore_index=True)

            print(f'step {self.good_steps_have_left}: scale - {scale}, shape - {shape}')
        print('good steps have left', self.good_steps_have_left, 'from', CFG.NUM_STEPS)
        print(diff_new)
        return diff_new.reshape(1,3)

    @staticmethod
    def find_quad(Y):
        res = []
        for i in range(len(Y)):
            res.append(Y[i][0]**2 + Y[i][1]**2 + Y[i][2]**2)
        return np.array(res)

    def optimization_step(self, x_data_path, num_steps, acquisition_type='EI',
                          normalize=False, num_cores=-1, evaluator_type='lbfgs'):

        d_train = np.load(x_data_path)
        shape_train, scale_train, all_samples_distributions_sum_train = \
            d_train['shape'], d_train['scale'], d_train['all_samples_distributions_sum']

        self.const = 1 #self.find_const_for_mult(self.y_real, all_samples_distributions_sum_train)
        self.data = np.concatenate((self.y_real.reshape(1, -1), all_samples_distributions_sum_train), axis=0)
        self.fpca_discretized = fpca(self.data)
        y_train = make_data_vector_fpca_output(self.data, self.fpca_discretized,
                                               mult=self.const)
        x_train = np.concatenate((shape_train.reshape(-1, 1), scale_train.reshape(-1, 1)), axis=1)

        self.opt = False
        self.model = GPRegression(x_train, y_train, kernel = cfg_mass.KERNEL)
        self.model_wrapped = GPyModelWrapper(self.model)
        #acq = L2_LCB(model=self.model_wrapped, target=self.target)
        acq = L2_EI(model=self.model_wrapped, target=self.target)
        # print('model', myBopt.model.model)
        print('optimization starts')

        self.opt = True
        self.opt_steps = {'vecs': [],
                          'loss': []}

        fit_update = lambda a, b: self.model.optimize_restarts(verbose=False)
        self.bayesopt_loop = BayesianOptimizationLoop(
                                                 model=self.model_wrapped,
                                                 space=self.space,
                                                 acquisition=acq
                                                )
        self.bayesopt_loop.iteration_end_event.append(fit_update)
        self.bayesopt_loop.run_loop(self.fokker_plank_eq, num_steps)
        #result_0 = bayesopt_loop.get_results()
        y_opt = self.bayesopt_loop.loop_state.Y[x_train.shape[0]:]
        y_all = self.bayesopt_loop.loop_state.Y
        x_opt = self.bayesopt_loop.loop_state.X[x_train.shape[0]:]
        x_all = self.bayesopt_loop.loop_state.X

        y_best = min(self.find_quad(y_all))
        best_vals = x_all[np.argmin(self.find_quad(y_all))]

        y_best_opt = min(self.find_quad(y_opt))
        best_vals_opt = x_opt[np.argmin(self.find_quad(y_opt))]

        plt.figure(figsize=(16, 12))
        plt.plot(cfg_mass.X, self.x_true, label='real X')
        plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals[0], scale=best_vals[1]), 'go--', linewidth=4,
                 markersize=2,
                 color='red', label='predicted')
        plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals_opt[0], scale=best_vals_opt[1]), 'go--', linewidth=4,
                 markersize=2,
                 color='green', label='predicted only from opt')
        plt.legend()
        path_for_save = self.path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(path_for_save + '.png')

        with open(self.path_for_save + 'predicted_data.json', 'w',
                  encoding='utf-8') as f:
            json.dump({'predictions_best': {'vec': best_vals.tolist(), 'loss': y_best},
                       'prediction_from_opt': {'vec': best_vals_opt.tolist(), 'loss': y_best_opt},
                       'real': self.x_real.tolist(), 'all_way': {'X': self.bayesopt_loop.loop_state.X.tolist(),
                                                                 'Y': self.bayesopt_loop.loop_state.Y.tolist()}}, f, indent=4)

        # print(myBopt.model.model)
        # plt.figure(figsize=(16, 12))
        # plt.plot(cfg_mass.X, self.x_true, label='real X')
        # plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals[0], scale=best_vals[1]), 'go--', linewidth=4,
        #          markersize=2,
        #          color='red', label='predicted')
        # plt.plot(cfg_mass.X, gamma.pdf(cfg_mass.X, best_vals_opt[0], scale=best_vals_opt[1]), 'go--', linewidth=4,
        #          markersize=2,
        #          color='green', label='predicted only from opt')
        # plt.legend()
        # path_for_save = self.path_for_save + 'experiment_' + str(self.exp_number)
        # plt.savefig(path_for_save + '.png')
        return None
