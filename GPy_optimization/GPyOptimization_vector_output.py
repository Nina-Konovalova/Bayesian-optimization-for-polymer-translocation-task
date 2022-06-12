import sys
import Configurations.config_mass as cfg_mass
from scipy.stats import gamma
sys.path.append('/')
from GPy.models import GPRegression

from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from utils.utils_mass import *
from target_vector_estimation.l2_bayes_opt.acquisitions import (
    L2NegativeLowerConfidenceBound as L2_LCB,
    L2ExpectedImprovement as L2_EI)
from numpy.random import seed
from utils.gauss_fit import *
from utils.data_frotran_utils import *
from utils.help_functions import *
import subprocess
import matplotlib.pyplot as plt
import Configurations.Config as CFG
import os
import json

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(42)


class BayesianOptimizationVectorOutput:
    def __init__(self, x_e, exp_number, path_for_save, save_opt_plots, num_gamma=2):

        self.good_steps_have_left = 0
        self.exp_number = exp_number
        self.save_opt_plots = save_opt_plots

        # save csv with info
        self.path_for_save = path_for_save
        self.data_out = pd.DataFrame(columns=['vecs', 'func_val',
                                              'rate_real', 'rate_opt', 'loss_rate',
                                              'mse_loss',
                                              'loss_angs', 'angle_suc_real', 'angle_unsuc_real',
                                              'angle_suc_opt', 'angle_unsuc_opt', 'angle_pos_diff', 'angle_neg_diff',
                                              'loss_times', 'time_suc_real', 'time_unsuc_real',
                                              'time_suc_opt', 'time_unsuc_opt', 'time_pos_diff', 'time_neg_diff'])

        if self.save_opt_plots:
            try:
                os.mkdir(self.path_for_save + 'optimization_pics/')
                print(f'make dir {self.path_for_save + "optimization_pics/"} to save opt plots')
            except:
                pass
        self.points_polymer = np.arange(CFG.MONOMERS)
        # data
        self.x_real = x_e['vecs'][exp_number]
        self.X_true = gaussian(self.points_polymer, self.x_real)

        self.rate_real = x_e['rates'][exp_number]

        self.angs_real = x_e['angs'][exp_number]
        self.ang_pos_real = x_e['angs'][exp_number][0]
        self.ang_neg_real = x_e['angs'][exp_number][1]

        self.y_pos_real = x_e['y_pos'][exp_number]
        self.y_neg_real = x_e['y_neg'][exp_number]

        self.times_real = x_e['times'][exp_number]
        self.time_pos_real = x_e['times'][exp_number][0]
        self.time_neg_real = x_e['times'][exp_number][1]

        self.opt_steps = {'vecs': [],
                          'loss': []}

        print(f"Rate {self.rate_real}")
        print(f"angs_success {self.ang_pos_real}, ang_unsuccess {self.ang_neg_real}")
        print(f"time_success {self.time_pos_real}, time_unsuccess {self.time_neg_real}")

        # model parameters
        self.points_polymer = np.arange(CFG.MONOMERS)
        self.exp_number = exp_number
        self.space = CFG.SPACE_3_v
        #self.constraints = CFG.CONSTRAINTS[CFG.NUM_GAUSS]
        self.opt = False
        self.target = np.array([0, 0, 0, 0, 0])


    def fokker_plank_eq(self, x_end):

        problem = False
        best_vals = x_end[0]
        new_curv = gaussian(self.points_polymer, best_vals)
        make_input_file(new_curv)
        subprocess.check_output(["./outputic_1"])

        rate, times, y_pos_new, y_neg_new = read_data()


        if not problem:

            angs = angle(y_pos_new, y_neg_new)

            diff_new, loss_rate, loss_angs, loss_mse, loss_times = \
                function_vector(self.rate_real, self.angs_real, self.y_pos_real, self.y_neg_real, self.times_real,
                         rate, angs, y_pos_new, y_neg_new, times, "optimization")
            print('rate_new', rate, 'rate_real', self.rate_real, 'differ', diff_new, 'loss angs', loss_angs)

            res = {'vecs': best_vals,
                   'func_val': diff_new,
                   'rate_real': self.rate_real,
                   'rate_opt': rate,
                   'loss_rate': loss_rate,
                   'mse_loss': loss_mse,
                   'loss_angs': loss_angs,
                   'angle_suc_real': self.ang_pos_real,
                   'angle_unsuc_real': self.ang_neg_real,
                   'angle_suc_opt': angs[0],
                   'angle_unsuc_opt': angs[1],
                   'angle_pos_diff': abs(angs[0] - self.ang_pos_real),
                   'angle_neg_diff': abs(angs[1] - self.ang_neg_real),
                   'loss_times': loss_times,
                   'time_suc_real': self.time_pos_real,
                   'time_unsuc_real': self.time_neg_real,
                   'time_suc_opt': times[0],
                   'time_unsuc_opt': times[1],
                   'time_pos_diff': abs(times[0] - self.time_pos_real),
                   'time_neg_diff': abs(times[1] - self.time_neg_real),
                   }

            if self.opt:
                self.good_steps_have_left += 1
                self.opt_steps['vecs'].append(x_end.tolist())
                self.opt_steps['loss'].append(diff_new.tolist())
                if self.save_opt_plots:
                    plt.figure(figsize=(16, 12))
                plt.plot(new_curv)
                try:
                    print(f'make dir {self.path_for_save + "optimization_pics/"} to save opt plots')
                    os.mkdir(self.path_for_save + 'optimization_pics/')
                except:
                    pass
                plt.savefig(
                    self.path_for_save + 'optimization_pics/' + 'pic' + str(self.good_steps_have_left) + '.jpg')

                self.data_out = self.data_out.append(res, ignore_index=True)
                self.data_out.to_csv(self.path_for_save + 'results_compare_' + str(self.rate_real) + '.csv',
                                     index=False)
        print('good steps have left', self.good_steps_have_left, 'from', CFG.NUM_STEPS)
        print(diff_new)

        return diff_new.reshape(1,5)

    @staticmethod
    def find_quad(Y):
        res = []
        for i in range(len(Y)):
            res.append(np.sqrt(Y[i][0]**2 + Y[i][1]**2 + Y[i][2]**2 +
                       Y[i][3]**2 + Y[i][4]**2))
        return np.array(res)

    def all_losses(self, best_vals):
        print(best_vals)
        new_curv = gaussian(self.points_polymer, best_vals)
        make_input_file(new_curv)
        subprocess.check_output(["./outputic_1"])
        rate, times, y_pos_new, y_neg_new = read_data()
        angs = angle(y_pos_new, y_neg_new)
        return function_vector(self.rate_real, self.angs_real, self.y_pos_real, self.y_neg_real, self.times_real,
                        rate, angs, y_pos_new, y_neg_new, times, "optimization")

    def optimization_step(self, x_data_path, num_steps, acquisition_type='EI',
                          normalize=False, num_cores=-1, evaluator_type='lbfgs'):

        d_train = np.load(x_data_path)
        vecs_t, rates_t, angs_t, y_pos_t, y_neg_t, times_t = d_train['vecs'], d_train['rates'], d_train['angs'], \
                                                             d_train['y_pos'], d_train['y_neg'], d_train['times']
        y_train = make_data(self.rate_real, self.angs_real, self.y_pos_real, self.y_neg_real, self.times_real,
                            rates_t, angs_t, y_pos_t, y_neg_t, times_t, 'optimization', 'vector')

        self.opt = False
        self.model = GPRegression(vecs_t, y_train)
        self.model_wrapped = GPyModelWrapper(self.model)
        #acq = L2_LCB(model=self.model_wrapped, target=self.target)
        acq = L2_EI(model=self.model_wrapped, target=self.target)
        # print('model', myBopt.model.model)
        print('optimization starts')

        self.opt = True
        self.opt_steps = {'vecs': [],
                          'loss': []}

        fit_update = lambda a, b: self.model.optimize_restarts(verbose=False)
        bayesopt_loop = BayesianOptimizationLoop(
                                                 model=self.model_wrapped,
                                                 space=self.space,
                                                 acquisition=acq
                                                )
        bayesopt_loop.iteration_end_event.append(fit_update)
        bayesopt_loop.run_loop(self.fokker_plank_eq, num_steps)
        #result_0 = bayesopt_loop.get_results()
        y_opt = bayesopt_loop.loop_state.Y[vecs_t.shape[0]:]
        y_all = bayesopt_loop.loop_state.Y
        x_opt = bayesopt_loop.loop_state.X[vecs_t.shape[0]:]
        x_all = bayesopt_loop.loop_state.X

        y_best = min(self.find_quad(y_all))
        best_vals = x_all[np.argmin(self.find_quad(y_all))]

        y_best_opt = min(self.find_quad(y_opt))
        best_vals_opt = x_opt[np.argmin(self.find_quad(y_opt))]

        diff_final, loss_rate, loss_angs, loss_mse, loss_times = self.all_losses(best_vals)
        diff_final_opt, loss_rate_opt, loss_angs_opt, loss_mse_opt, loss_times_opt = self.all_losses(best_vals_opt)
        with open(self.path_for_save + 'predicted_data.json', 'w',
                  encoding='utf-8') as f:

            json.dump({'predictions_best': {'vec': best_vals.tolist(), 'loss': y_best.tolist(), 'loss_rate': loss_rate,
                                            'loss_angs': loss_angs, 'loss_mse': loss_mse, 'loss_times': loss_times},
                       'prediction_from_opt': {'vec': best_vals_opt.tolist(), 'loss': y_best_opt.tolist(),
                                               'loss_rate': loss_rate_opt, 'loss_angs': loss_angs_opt,
                                               'loss_mse': loss_mse_opt, 'loss_times': loss_times_opt},
                       'real': self.x_real.tolist(), 'all_way': self.opt_steps}, f, indent=4)

        plt.figure(figsize=(16, 12))
        plt.plot(self.points_polymer, self.X_true, label='real X')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, best_vals), 'go--', linewidth=4,
                 markersize=2,
                 color='red', label='predicted')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, best_vals_opt), 'go--', linewidth=4,
                 markersize=2,
                 color='green', label='predicted only from opt')
        plt.legend()
        path_for_save = self.path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(path_for_save + '.png')
        return None
