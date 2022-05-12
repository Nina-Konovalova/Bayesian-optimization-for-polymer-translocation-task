import argparse
import GP_mass_config as CFG_REG
from GP_regressor import GPRegressor
import json
import time
import sys
import os
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.append('../')

from utils.gauss_fit import *
from utils.data_frotran_utils import *
from utils.utils_mass import make_data, make_data_parallel
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


class RegressionCheck:
    def __init__(self, args):
        self.d_train = np.load(CFG_REG.TRAIN_PATH, allow_pickle=True)
        self.d_test = np.load(CFG_REG.TEST_PATH, allow_pickle=True)
        self.d_exp = np.load(CFG_REG.VAL_PATH, allow_pickle=True)

        # make te - test, t - train and e - experimental data
        self.shape_train, self.scale_train, self.all_samples_distributions_sum_train = \
            self.d_train['shape'].copy(), self.d_train['scale'].copy(), self.d_train['all_samples_distributions_sum'].copy()

        self.shape_test, self.scale_test, self.all_samples_distributions_sum_test = \
            self.d_test['shape'].copy(), self.d_test['scale'].copy(), self.d_test['all_samples_distributions_sum'].copy()

        self.shape_exp, self.scale_exp, self.all_samples_distributions_sum_exp = \
            self.d_exp['shape'].copy(), self.d_exp['scale'].copy(), self.d_exp['all_samples_distributions_sum'].copy()

        del self.d_exp
        del self.d_test
        del self.d_train
        print('train:', self.shape_train.shape)
        print('test:', self.shape_test.shape)
        print('experiments:', self.shape_exp.shape)

        self.check_dirs(args.save_path)
        self.args = args



    @staticmethod
    def check_dirs(path_to_save):
        try:
            os.mkdir(path_to_save)
            print(f'dir with mame {path_to_save} created')
        except:
            print(f'dir with mame {path_to_save} already exists')

    def help(self, group_number):
        print('aaa')

    def make_regression_parallel(self):
        '''
        train data, test data, experimental data, pathes can be changed in configuration file (see Readme_regression)
        Also you can change CFG_REG.GRID - different kernels to check
        :param args:
        :return:
        '''

        for k in CFG_REG.GRID.keys():
            print(f'check kernel {k}')
            try:
                os.mkdir(self.args.save_path + k)
                print(f'dir with mame {self.args.save_path + k} created')
            except:
                print(f'dir with mame {self.args.save_path + k} already exists')

            self.k = k
            num_processes = self.args.num_processes
            self.frame_jump_unit = len(self.shape_exp) // num_processes
            with Pool(num_processes) as p:
                p.map(self.data_process, range(num_processes))
                # p.close()
                # p.join()



    def data_process(self, group_number):
        bad_samples = []
        if self.frame_jump_unit * (group_number + 1) <= len(self.scale_train):
            array = np.arange(self.frame_jump_unit * group_number, self.frame_jump_unit * (group_number + 1))
        else:
            array = np.arange(self.frame_jump_unit * group_number, len(self.scale_exp))

        for i in tqdm(array):
            metrics = {}

            save_model_path = self.args.save_path + self.k + '/' + 'exp_' + str(i)
            path_save_predictions = self.args.save_path + self.k + '/' + 'exp_' + str(i) + 'predictions.npz'
            since = time.time()

            y_test = make_data(self.all_samples_distributions_sum_exp[i], self.all_samples_distributions_sum_test,
                               self.shape_exp[i], self.scale_exp[i], self.shape_test, self.scale_test,
                               e=i, make_plots=self.args.save_plots)
            y_train = make_data(self.all_samples_distributions_sum_exp[i], self.all_samples_distributions_sum_train,
                                self.shape_exp[i], self.scale_exp[i], self.shape_train, self.scale_train,
                                e=i, make_plots=self.args.save_plots)

            x_train = np.concatenate((self.shape_train.reshape(-1, 1), self.scale_train.reshape(-1, 1)), axis=1)

            x_test = np.concatenate((self.shape_test.reshape(-1, 1), self.scale_test.reshape(-1, 1)), axis=1)

            regressor = GPRegressor(CFG_REG.GRID[self.k], save_model_path=save_model_path,
                                    save_predictions=path_save_predictions)

            m = regressor.optimization(x_train, y_train.reshape(-1, 1))

            with open(save_model_path + '.json', 'w', encoding='utf-8') as f:
                json.dump(m.to_dict(), f, indent=4)

            pred, var = regressor.predict(m, x_test)

            np.savez_compressed(path_save_predictions, mean=pred, var=var, real_vals=y_test)
            metrics[self.k] = regressor.criterion(pred, y_test)

            print('experiment', i)
            print('metrics', metrics[self.k])
            time_elapsed = time.time() - since
            print('Kernel complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('-' * 20)

            with open(self.args.save_path + self.k + '/' + 'exp_' + str(
                    i) + 'metrics.json', 'w',
                      encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)

        with open(self.args.save_path + self.k + '/' + 'bad_samples.json', 'w',
                  encoding='utf-8') as f:
            json.dump(bad_samples, f, indent=4)

    def make_regression(self):
        '''
        train data, test data, experimental data, pathes can be changed in configuration file (see Readme_regression)
        Also you can change CFG_REG.GRID - different kernels to check
        :param args:
        :return:
        '''

        for k in CFG_REG.GRID.keys():
            print(f'check kernel {k}')
            try:
                os.mkdir(self.args.save_path + k)
                print(f'dir with mame {self.args.save_path + k} created')
            except:
                print(f'dir with mame {self.args.save_path + k} already exists')

            bad_samples = []
            for i in (range(len(self.shape_exp))):
                metrics = {}

                save_model_path = self.args.save_path + k + '/' + 'exp_' + str(i)
                path_save_predictions = self.args.save_path + k + '/' + 'exp_' + str(i) + 'predictions.npz'
                since = time.time()
                if not self.args.parallel_data:
                    y_test = make_data(self.all_samples_distributions_sum_exp[i], self.all_samples_distributions_sum_test,
                                       self.shape_exp[i], self.scale_exp[i], self.shape_test, self.scale_test,
                                       e=i, make_plots=self.args.save_plots)
                    y_train = make_data(self.all_samples_distributions_sum_exp[i], self.all_samples_distributions_sum_train,
                                        self.shape_exp[i], self.scale_exp[i], self.shape_train, self.scale_train,
                                        e=i, make_plots=self.args.save_plots)
                else:

                    y_train = make_data_parallel(self.all_samples_distributions_sum_exp[i],
                                                 self.all_samples_distributions_sum_train,
                                                 4, self.shape_exp[i], self.scale_exp[i],
                                                 self.shape_train, self.scale_train,
                                                 e=i, make_plots=self.args.save_plots)
                    y_test = make_data_parallel(self.all_samples_distributions_sum_exp[i],
                                                self.all_samples_distributions_sum_test,
                                                4, self.shape_exp[i], self.scale_exp[i],
                                                self.shape_test, self.scale_test,
                                                e=i, make_plots=self.args.save_plots
                                                )

                # x_train = np.concatenate((self.shape_train.reshape(-1, 1), self.scale_train.reshape(-1, 1)), axis=1)
                #
                # x_test = np.concatenate((self.shape_test.reshape(-1, 1), self.scale_test.reshape(-1, 1)), axis=1)
                x_train = np.concatenate((self.shape_train, self.scale_train), axis=0).T

                x_test = np.concatenate((self.shape_test, self.scale_test), axis=0).T
                print(x_train.shape)

                regressor = GPRegressor(CFG_REG.GRID[k], save_model_path=save_model_path,
                                        save_predictions=path_save_predictions)

                m = regressor.optimization(x_train, y_train.reshape(-1, 1))

                with open(save_model_path + '.json', 'w', encoding='utf-8') as f:
                    json.dump(m.to_dict(), f, indent=4)

                pred, var = regressor.predict(m, x_test)

                np.savez_compressed(path_save_predictions, mean=pred, var=var, real_vals=y_test)
                metrics[k] = regressor.criterion(pred, y_test)

                print('experiment', i)
                print('metrics', metrics[k])
                time_elapsed = time.time() - since
                print('Kernel complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('-' * 20)

                with open(self.args.save_path + k + '/' + 'exp_' + str(
                        i) + 'metrics.json', 'w',
                          encoding='utf-8') as f:
                    json.dump(metrics, f, indent=4)

            with open(self.args.save_path  + k + '/' + 'bad_samples.json', 'w',
                      encoding='utf-8') as f:
                json.dump(bad_samples, f, indent=4)





def main():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    # configuration and regime
    parser.add_argument('--save_path', default='gp_regression_mass_3/', type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')
    parser.add_argument('--parallel_data', default=False, type=bool,
                        help='make data parallel')
    parser.add_argument('--parallel_reg', default=False, type=bool,
                        help='make regression parallel')
    parser.add_argument('--save_plots', default=False, type=bool,
                        help='If we want to save plots')
    parser.add_argument('--num_processes', default=4, type=int,
                        help='num of processes for parallel running')

    args = parser.parse_args()
    checker = RegressionCheck(args)
    if not args.parallel_reg:
        checker.make_regression()
    else:
        checker.make_regression_parallel()

if __name__ == '__main__':
    main()
