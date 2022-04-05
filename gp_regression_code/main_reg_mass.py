import argparse
import GP_mass_config as CFG_REG
from GP_regressor import GPRegressor
import json
import time
import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.append('../')

from utils.gauss_fit import *
from utils.data_frotran_utils import *
from utils.utils_mass import make_data


def make_regression(args):
    '''
    train data, test data, experimental data, pathes can be changed in configuration file (see Readme_regression)
    Also you can change CFG_REG.GRID - different kernels to check
    :param args:
    :return:
    '''
    d_train = np.load(CFG_REG.TRAIN_PATH)
    d_test = np.load(CFG_REG.TEST_PATH)
    d_exp = np.load(CFG_REG.VAL_PATH)

    # make te - test, t - train and e - experimental data
    shape_train, scale_train, all_samples_distributions_sum_train = \
        d_train['shape'], d_train['scale'], d_train['all_samples_distributions_sum']

    shape_test, scale_test, all_samples_distributions_sum_test = \
        d_test['shape'], d_test['scale'], d_test['all_samples_distributions_sum']

    shape_exp, scale_exp, all_samples_distributions_sum_exp = \
        d_exp['shape'], d_exp['scale'], d_exp['all_samples_distributions_sum']

    print('train:', shape_train.shape)
    print('test:', shape_test.shape)
    print('experiments:', shape_exp.shape)

    check_dirs(args.save_path)
    for k in CFG_REG.GRID.keys():
        print(f'check kernel {k}')
        try:
            os.mkdir(args.save_path + k)
            print(f'dir with mame {args.save_path + k} created')
        except:
            print(f'dir with mame {args.save_path + k} already exists')

        bad_samples = []
        for i in (range(len(shape_exp))):
            metrics = {}

            save_model_path = args.save_path + k + '/' + 'exp_' + str(i)
            path_save_predictions = args.save_path + k + '/' + 'exp_' + str(i) + 'predictions.npz'
            since = time.time()
            y_train = make_data(all_samples_distributions_sum_exp[i], all_samples_distributions_sum_train)
            x_train = np.concatenate((shape_train.reshape(-1, 1), scale_train.reshape(-1, 1)), axis=1)
            print(x_train.shape)
            y_test = make_data(all_samples_distributions_sum_exp[i], all_samples_distributions_sum_test)
            x_test = np.concatenate((shape_test.reshape(-1,1), scale_test.reshape(-1,1)), axis=1)

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

            with open(args.save_path + k + '/' + 'exp_' + str(
                    i) + 'metrics.json', 'w',
                      encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)

        with open(args.save_path  + k + '/' + 'bad_samples.json', 'w',
                  encoding='utf-8') as f:
            json.dump(bad_samples, f, indent=4)


def check_dirs(path_to_save):
    try:
        os.mkdir(path_to_save)
        print(f'dir with mame {path_to_save} created')
    except:
        print(f'dir with mame {path_to_save} already exists')


def main():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    # configuration and regime
    parser.add_argument('--save_path', default='gp_regression_mass_0/', type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')

    args = parser.parse_args()
    make_regression(args)


if __name__ == '__main__':
    main()
