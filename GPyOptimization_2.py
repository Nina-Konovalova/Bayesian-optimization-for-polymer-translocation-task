import GPyOpt
from numpy.random import seed
from utils.gauss_fit import *
from utils.data_frotran_utils import *
from utils.landscape_to_distr import *
from utils.help_functions import *
import subprocess
import matplotlib.pyplot as plt
import Configurations.Config as CFG
import os
import json

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(42)


class BayesianOptimization:
    def __init__(self, model_type, x_e, exp_number, path_for_save, save_opt_plots):

        self.good_steps_have_left = 0
        self.exp_number = exp_number
        self.save_opt_plots = save_opt_plots
        self.points_polymer = np.arange(CFG.MONOMERS)

        # save scv with info
        self.path_for_save = path_for_save
        self.data_out = pd.DataFrame(columns=['vecs', 'func_val',
                                              'rate_real', 'rate_opt', 'loss_rate',
                                              'mse_loss',
                                              'loss_angs', 'angle_suc_real', 'angle_unsuc_real',
                                              'angle_suc_opt', 'angle_unsuc_opt', 'angle_pos_diff', 'angle_neg_diff',
                                              'loss_times', 'time_suc_real', 'time_unsuc_real',
                                              'time_suc_opt', 'time_unsuc_opt', 'time_pos_diff', 'time_neg_diff'])

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
        self.model_type = model_type
        self.points_polymer = np.arange(CFG.MONOMERS)
        self.exp_number = exp_number
        self.space = CFG.SPACE[CFG.NUM_GAUSS]
        self.constraints = CFG.CONSTRAINTS[CFG.NUM_GAUSS]
        self.opt = False

        if self.model_type == 'GP':
            self.model = GPyOpt.models.GPModel(CFG.KERNEL, optimize_restarts=1, exact_feval=True)
        elif self.model_type == 'GP_MCMC':
            self.model = GPyOpt.models.GPModel_MCMC(CFG.KERNEL, exact_feval=True)
        else:
            raise ValueError('no such type of model implemented')

    def all_losses(self, best_vals):
        new_curv = gaussian(self.points_polymer, best_vals)
        make_input_file(new_curv)
        subprocess.check_output(["./outputic"])
        rate, times, y_pos_new, y_neg_new = read_data()
        angs = angle(y_pos_new, y_neg_new)
        return function(self.rate_real, self.angs_real, self.y_pos_real, self.y_neg_real, self.times_real,
                        rate, angs, y_pos_new, y_neg_new, times, "optimization")

    def fokker_plank_eq(self, x_end):
        problem = False
        best_vals = x_end[0]
        new_curv = gaussian(self.points_polymer, best_vals)
        make_input_file(new_curv)
        subprocess.check_output(["./outputic"])
        old_der = read_derives()
        if (abs(old_der) > 4.).sum() != 0:
            print('too big derives')
            problem = True

        rate, times, y_pos_new, y_neg_new = read_data()

        if rate == 1e20 or rate == 0:
            print('Out of space')
            problem = True

        if rate > 1:
            print('Rate bigger 1')
            problem = True

        angs = angle(y_pos_new, y_neg_new)
        diff_new, loss_rate, loss_angs, loss_mse, loss_times = \
            function(self.rate_real, self.angs_real, self.y_pos_real, self.y_neg_real, self.times_real,
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

        if not problem:
            if self.opt:
                self.good_steps_have_left += 1
                self.opt_steps['vecs'].append(x_end.tolist())
                self.opt_steps['loss'].append(diff_new)
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
        return diff_new, x_end, problem

    def optimization_step(self, x_data_path, num_steps, acquisition_type='EI',
                          normalize=False, num_cores=-1, evaluator_type='lbfgs'):

        d_train = np.load(x_data_path)
        vecs_t, rates_t, angs_t, y_pos_t, y_neg_t, times_t = d_train['vecs'], d_train['rates'], d_train['angs'], \
                                                             d_train['y_pos'], d_train['y_neg'], d_train['times']
        y_train = make_data(self.rate_real, self.angs_real, self.y_pos_real, self.y_neg_real, self.times_real,
                            rates_t, angs_t, y_pos_t, y_neg_t, times_t, 'optimization')
        self.opt = False
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.fokker_plank_eq,  # function to optimize
                                                     domain=self.space,
                                                     constraints=self.constraints,  # box-constraints of the problem
                                                     model=self.model,
                                                     Model_type=self.model_type,
                                                     X=vecs_t,
                                                     Y=y_train.reshape(-1, 1),
                                                     verbosity=False,
                                                     normalize_Y=False,
                                                     num_cores=1,
                                                     acquisition_type=acquisition_type)

        # print('model', myBopt.model.model)
        print('optimization starts')

        self.opt = True
        myBopt.run_optimization(num_steps,
                                report_file=self.path_for_save + 'report_file_' + str(self.exp_number) + '.txt',
                                models_file=self.path_for_save + 'model_params_' + str(self.exp_number) + '.txt')

        best_vals = myBopt.x_opt
        best_vals_opt = np.array(self.opt_steps['vecs'])[np.argmin(np.array(self.opt_steps['loss']))]

        diff_final, loss_rate, loss_angs, loss_mse, loss_times = self.all_losses(best_vals)
        diff_final_opt, loss_rate_opt, loss_angs_opt, loss_mse_opt, loss_times_opt = self.all_losses(best_vals_opt[0])
        with open(self.path_for_save + 'predicted_data.json', 'w',
                  encoding='utf-8') as f:
            json.dump({'predictions_best': {'vec': best_vals.tolist(), 'loss': diff_final, 'loss_rate': loss_rate,
                                            'loss_angs': loss_angs, 'loss_mse': loss_mse, 'loss_times': loss_times},
                       'prediction_from_opt': {'vec': best_vals_opt.tolist(), 'loss': diff_final_opt,
                                               'loss_rate': loss_rate_opt, 'loss_angs': loss_angs_opt,
                                               'loss_mse': loss_mse_opt, 'loss_times': loss_times_opt},
                       'real': self.x_real.tolist(), 'all_way': self.opt_steps}, f, indent=4)
        print(myBopt.model.model)
        plt.figure(figsize=(16, 12))
        plt.plot(self.points_polymer, self.X_true, label='real X')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, best_vals), 'go--', linewidth=4,
                 markersize=2,
                 color='red', label='predicted')
        plt.plot(self.points_polymer, gaussian(self.points_polymer, best_vals_opt[0]), 'go--', linewidth=4,
                 markersize=2,
                 color='green', label='predicted only from opt')
        plt.legend()
        path_for_save = self.path_for_save + 'experiment_' + str(self.exp_number)
        plt.savefig(self.path_for_save + '.png')
        return best_vals
