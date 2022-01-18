import argparse
import GP_config as cfg
import Config as CFG
from GP_regressor import GPRegressor
import json
import time
import sys
import os

sys.path.append('C:/Users/nina_/PycharmProjects/bayesian_optimization/')
from sklearn.metrics import mean_squared_error as mse
from utils.gauss_fit import *
from utils.data_frotran_utils import *
import subprocess


def funcktion(rate_real, y_pos_real, y_neg_real, rate_new, y_pos_new, y_neg_new):
    loss_true = mse((y_pos_new[:]), (y_pos_real))
    loss_false = mse((y_neg_new[:]), (y_neg_real[:]))
    loss_rate = abs(rate_new - rate_real)
    loss_rate *= 10 ** (-6) #(-int(math.log((2 * loss_rate) / (loss_true + loss_false), 10)))

    # print(rate, self.rate_real)
    diff_new = loss_false + loss_true + loss_rate
    #diff_new = loss_rate / rate_real

    return diff_new


def make_data(rate_real, y_pos_real, y_neg_real, rate, y_pos, y_neg):
    f = []
    for i in (range(len(rate))):
        f.append(funcktion(rate_real, y_pos_real, y_neg_real, rate[i], y_pos[i], y_neg[i]))
    return np.array(f)


def fokker_plank_eq(x_param):
    x = np.arange(51)
    best_vals = x_param
    new_curv = gaussian(x, *best_vals)
    make_input_file(new_curv)
    # fortran programm - making new distributions
    subprocess.check_output(["./outputic"])
    rate, y_pos_new, y_neg_new = read_data()
    return rate, y_pos_new, y_neg_new


def angle(y_pos_new, y_neg_new):
    xx = np.arange(len(y_pos_new))
    t_pos = np.polyfit(xx, np.log(y_pos_new), 1)
    t_neg = np.polyfit(xx, np.log(y_neg_new), 1)
    return [t_pos[0], t_neg[0]]


def make_dict(path):
    d = {}
    dat = np.load(path)
    d['vecs'] = dat['vecs']
    d['rates'] = dat['rates']
    d['y_pos'] = dat['y_pos']
    d['y_neg'] = dat['y_neg']
    return d

def main():
    parser = argparse.ArgumentParser(description='End-to-end inference')

    # configuration and regime
    parser.add_argument('--save_path', default='gp_regression_results_5_gaussians_new/', type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')

    args = parser.parse_args()

    d_train = np.load(CFG.TRAIN_PATH)
    d_test = np.load(CFG.TEST_PATH)
    d_exp = np.load(CFG.VAL_PATH)

    vecs_te, rates_te, angs_te, y_pos_te, y_neg_te = d_test['vecs'], d_test['rates'], d_test['angs'], d_test['y_pos'], \
                                                d_test['y_neg']
    vecs_t, rates_t, angs_t, y_pos_t, y_neg_t = d_train['vecs'], d_train['rates'], d_train['angs'], d_train['y_pos'], \
                                                d_train['y_neg']
    vecs_e, rates_e, angs_e, y_pos_e, y_neg_e = d_exp['vecs'], d_exp['rates'], d_exp['angs'], d_exp['y_pos'], \
                                                d_exp['y_neg']

    print('train:', vecs_t.shape)  # , rate_train.shape, angle_pos_train.shape, angle_neg_train.shape)
    print('test:', vecs_te.shape)  # , rate_test.shape, angle_pos_test.shape, angle_neg_test.shape)
    print('experiments:', vecs_te.shape)

    for k in cfg.GRID.keys():
        print(k)
        try:
            os.mkdir(args.save_path + k + '/')
        except:
            pass
        bad_samples = []
        for i in range(len(vecs_e)):
            try:
                metrics = {}
                save_model_path = args.save_path + k + '/' + 'exp_' + str(i)
                path_save_predictions = args.save_path + k + '/' + 'exp_' + str(i) + 'predictions.npz'
                since = time.time()
                y_train = make_data(rates_e[i], y_pos_e[i], y_neg_e[i], rates_t, y_pos_t, y_neg_t)
                y_test = make_data(rates_e[i], y_pos_e[i], y_neg_e[i], rates_te, y_pos_te, y_neg_te)
                regressor = GPRegressor(cfg.GRID[k], save_model_path=save_model_path, save_predictions=path_save_predictions)

                m = regressor.optimization(vecs_t[:], y_train[:].reshape(-1, 1))

                with open(save_model_path + '.json', 'w', encoding='utf-8') as f:
                    json.dump(m.to_dict(), f, indent=4)

                pred, var = regressor.predict(m, vecs_te[:])

                np.savez_compressed(path_save_predictions, mean=pred, var=var, real_vals=y_test)
                metrics[k] = regressor.criterion(pred, y_test)

                print('experinet', i)
                print('metrics', metrics[k])
                time_elapsed = time.time() - since
                print('Kernel complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('-' * 20)

                with open(args.save_path + k + '/' + 'exp_' + str(i) + 'metrics.json', 'w',
                      encoding='utf-8') as f:
                    json.dump(metrics, f, indent=4)
            except:
                print('fail', i)
                bad_samples.append(i)

        with open(args.save_path + k + '/' + 'bad_samples.json', 'w',
                  encoding='utf-8') as f:
            json.dump(bad_samples, f, indent=4)

if __name__ == '__main__':
    main()
