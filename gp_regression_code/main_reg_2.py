import argparse
import GP_config as CFG_REG
from GP_regressor import GPRegressor
import json
import time
import sys
import os
import numpy as np

sys.path.append('../')

from utils.gauss_fit import *
from utils.data_frotran_utils import *
from utils.help_functions import function, angle, make_data


def main():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    # configuration and regime
    parser.add_argument('--save_path', default='gp_regression_results_3_gaussians_small_3/', type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')

    args = parser.parse_args()
    # train data, test data, experimental data, pathes can be changed in configuration file (see Readme_regression)
    d_train = np.load(CFG_REG.TRAIN_PATH)
    d_test = np.load(CFG_REG.TEST_PATH)
    d_exp = np.load(CFG_REG.VAL_PATH)

    # make te - test, t - train and e - experimental data
    vecs_te, rates_te, angs_te, y_pos_te, y_neg_te, times_te = \
        d_test['vecs'], d_test['rates'], d_test['angs'], \
        d_test['y_pos'], d_test['y_neg'], d_test['times']

    vecs_t, rates_t, angs_t, y_pos_t, y_neg_t, times_t = \
        d_train['vecs'], d_train['rates'], d_train['angs'], \
        d_train['y_pos'], d_train['y_neg'], d_train['times']

    vecs_e, rates_e, angs_e, y_pos_e, y_neg_e, times_e = \
        d_exp['vecs'], d_exp['rates'], d_exp['angs'], \
        d_exp['y_pos'], d_exp['y_neg'], d_exp['times']

    print('train:', vecs_t.shape)  # , rate_train.shape, angle_pos_train.shape, angle_neg_train.shape)
    print('test:', vecs_te.shape)  # , rate_test.shape, angle_pos_test.shape, angle_neg_test.shape)
    print('experiments:', vecs_te.shape)

    try:
        print(f'dir with mame {args.save_path} created')
        os.mkdir(args.save_path)
    except:
        pass

    # try:
    #     #os.mkdir(args.save_path + 'angs/')
    # except:
    #     pass

    try:
        print(f'in dir {args.save_path} create new dir {str(CFG_REG.ALPHA)}')
        os.mkdir(args.save_path + str(CFG_REG.ALPHA))
        # os.mkdir(args.save_path + 'angs/' + str(CFG_REG.ALPHA))
    except:
        pass

    print('weight alpha', CFG_REG.ALPHA)
    for k in CFG_REG.GRID.keys():
        print(k)
        try:
            print(f'in dir {args.save_path + str(CFG_REG.ALPHA)} create new dir {k}')
            os.mkdir(args.save_path + str(CFG_REG.ALPHA) + '/' + k + '/')
        except:
            pass
        bad_samples = []
        for i in range(len(vecs_e)):
            metrics = {}

            save_model_path = args.save_path + str(CFG_REG.ALPHA) + '/' + k + '/' + 'exp_' + str(i)
            path_save_predictions = args.save_path + str(CFG_REG.ALPHA) + '/' + k + '/' + 'exp_' + str(i) \
                                    + 'predictions.npz'
            since = time.time()
            y_train = make_data(rates_e[i], angs_e[i], y_pos_e[i], y_neg_e[i],
                                times_e[i],
                                rates_t, angs_t, y_pos_t, y_neg_t, times_t,
                                'approximation')

            y_test = make_data(rates_e[i], angs_e[i], y_pos_e[i], y_neg_e[i],
                               times_e[i],
                               rates_te, angs_te, y_pos_te, y_neg_te, times_te,
                               'approximation')

            regressor = GPRegressor(CFG_REG.GRID[k], save_model_path=save_model_path,
                                    save_predictions=path_save_predictions)

            m = regressor.optimization(vecs_t[:], y_train[:].reshape(-1, 1))

            with open(save_model_path + '.json', 'w', encoding='utf-8') as f:
                json.dump(m.to_dict(), f, indent=4)

            pred, var = regressor.predict(m, vecs_te[:])

            np.savez_compressed(path_save_predictions, mean=pred, var=var, real_vals=y_test)
            metrics[k] = regressor.criterion(pred, y_test)

            print('experiment', i)
            print('metrics', metrics[k])
            time_elapsed = time.time() - since
            print('Kernel complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('-' * 20)

            with open(args.save_path + str(CFG_REG.ALPHA) + '/' + k + '/' + 'exp_' + str(
                    i) + 'metrics.json', 'w',
                      encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)

        with open(args.save_path + str(CFG_REG.ALPHA) + '/' + k + '/' + 'bad_samples.json', 'w',
                  encoding='utf-8') as f:
            json.dump(bad_samples, f, indent=4)


if __name__ == '__main__':
    main()
