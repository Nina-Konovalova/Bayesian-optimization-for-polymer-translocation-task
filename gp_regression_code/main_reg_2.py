import argparse
import GP_config as CFG_REG
import Config_reg as cfg_reg
from GP_regressor import GPRegressor
import json
import time
import sys
import os
sys.path.append('../')
from sklearn.metrics import mean_squared_error as mse
from utils.gauss_fit import *
from utils.data_frotran_utils import *
import subprocess


def function(rate_real, angs_real, y_pos_real, y_neg_real, times_real, rate_new, angs_new,  y_pos_new, y_neg_new, times_new):
    '''
    :param rate_real: successful rate translocation for real free energy landscape (which is used in experiment)
    :param angs_real: successful angles from log time distributions (see function angle()) for real free energy landscape (which is used in experiment)
    :param y_pos_real: successful time distribution for real free energy landscape (which is used in experiment)
    :param y_neg_real: unsuccessful time distribution for real free energy landscape (which is used in experiment)
    :param times_real:
    :param rate_new: successful rate translocation for considering free energy landscape
    :param angs_new: uccessful angles from log time distributions (see function angle()) for considering free energy landscape
    :param y_pos_new: successful time distribution for considering free energy landscape
    :param y_neg_new: unsuccessful time distribution for considering free energy landscape
    :param time_new:
    :return: combinaton of losses between input variables
    '''
    loss_true = mse((y_pos_new[:]), (y_pos_real))
    loss_false = mse((y_neg_new[:]), (y_neg_real[:]))
    loss_rate = abs(rate_new - rate_real)
    loss_angs = (abs(angs_new - angs_real)).sum()
    loss_times = (abs(times_new - times_real)).sum()
    #loss_rate *= 10 ** (-6) #(-int(math.log((2 * loss_rate) / (loss_true + loss_false), 10)))
    #diff_new = loss_false + loss_true + loss_rate * CFG.ALPHA
    #diff_new = loss_rate / rate_real
    diff_new = loss_angs + loss_rate * cfg_reg.ALPHA
    return diff_new


def make_data(rate_real, angs_real, y_pos_real, y_neg_real, times_real, rate, angs, y_pos, y_neg, times):
    '''
    see all input variables in description for function()
    :return: array for loss function for each element of considered dataset
    '''
    f = []
    loss_angs = []
    loss_rate = []
    for i in (range(len(rate))):
        f.append(function(rate_real, angs_real, y_pos_real, y_neg_real, rate[i], angs[i], y_pos[i], y_neg[i])[0])
        loss_angs.append(function(rate_real, angs_real, y_pos_real, y_neg_real, rate[i], angs[i], y_pos[i], y_neg[i])[1])
        loss_rate.append(function(rate_real, angs_real, y_pos_real, y_neg_real, rate[i], angs[i], y_pos[i], y_neg[i])[2])
    return np.array(f)


# make angles form time distributions
def angle(y_pos_new, y_neg_new):
    '''
    :param y_pos_new: time distribution for successful translocation
    :param y_neg_new: time distribution for unsuccessful translocation
    :return: [tilt angle for log successful time distr.,  tilt angle for log unsuccessful time distr.]
    '''
    xx = np.arange(len(y_pos_new))
    t_pos = np.polyfit(xx, np.log(y_pos_new), 1)
    t_neg = np.polyfit(xx, np.log(y_neg_new), 1)
    return [t_pos[0], t_neg[0]]

# make data from npz file
def make_dict(path):
    d = {}
    dat = np.load(path)
    d['vecs'] = dat['vecs']
    d['rates'] = dat['rates']
    d['y_pos'] = dat['y_pos']
    d['y_neg'] = dat['y_neg']
    d['times'] = dat['times']
    return d


def main():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    # configuration and regime
    parser.add_argument('--save_path', default='gp_regression_results_3_gaussians_small_1/', type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')


    args = parser.parse_args()

    #train data, test data, experimental data, pathes can be changed in configuration file (see Readme_regression)

    d_train = np.load(cfg_reg.TRAIN_PATH)
    d_test = np.load(cfg_reg.TEST_PATH)
    d_exp = np.load(cfg_reg.VAL_PATH)

    # make te - test, t - train and e - experimental data
    vecs_te, rates_te, angs_te, y_pos_te, y_neg_te, times_te = d_test['vecs'], d_test['rates'], d_test['angs'], d_test['y_pos'], \
                                                d_test['y_neg'], d_test['times']
    vecs_t, rates_t, angs_t, y_pos_t, y_neg_t, times_t = d_train['vecs'], d_train['rates'], d_train['angs'], d_train['y_pos'], \
                                                d_train['y_neg'], d_train['times']
    vecs_e, rates_e, angs_e, y_pos_e, y_neg_e, times_e = d_exp['vecs'], d_exp['rates'], d_exp['angs'], d_exp['y_pos'], \
                                                d_exp['y_neg'], d_exp['times']

    print('train:', vecs_t.shape)  # , rate_train.shape, angle_pos_train.shape, angle_neg_train.shape)
    print('test:', vecs_te.shape)  # , rate_test.shape, angle_pos_test.shape, angle_neg_test.shape)
    print('experiments:', vecs_te.shape)

    try:
        os.mkdir(args.save_path)
    except:
        pass

    # try:
    #     #os.mkdir(args.save_path + 'angs/')
    # except:
    #     pass

    try:
        os.mkdir(args.save_path + str(cfg_reg.ALPHA))
        #os.mkdir(args.save_path + 'angs/' + str(cfg_reg.ALPHA))
    except:
        pass

    print('weight alpha', cfg_reg.ALPHA)
    for k in CFG_REG.GRID.keys():
        print(k)
        try:
            os.mkdir(args.save_path + str(cfg_reg.ALPHA) +'/' + k + '/')
            #os.mkdir(args.save_path + 'angs/' + str(cfg_reg.ALPHA) + '/' + k + '/')
        except:
            pass
        bad_samples = []
        for i in range(len(vecs_e)):
            try:
                metrics = {}
                # save_model_path = args.save_path + 'angs/' + str(cfg_reg.ALPHA) + '/' + k + '/' + 'exp_' + str(i)
                # path_save_predictions = args.save_path + 'angs/' + str(cfg_reg.ALPHA) + '/' + k + '/' + 'exp_' + str(i) \
                #                         + 'predictions.npz'

                save_model_path = args.save_path + str(cfg_reg.ALPHA) + '/' + k + '/' + 'exp_' + str(i)
                path_save_predictions = args.save_path + str(cfg_reg.ALPHA) + '/' + k + '/' + 'exp_' + str(i)  \
                                        + 'predictions.npz'
                since = time.time()
                y_train, loss_angs_tr, loss_rate_tr = make_data(rates_e[i], angs_e[i], y_pos_e[i], y_neg_e[i], times_e[i],
                                                                rates_t, angs_t, y_pos_t, y_neg_t, times_t)
                y_test, loss_angs_te, loss_rate_te = make_data(rates_e[i], angs_e[i], y_pos_e[i], y_neg_e[i], times_e[i],
                                                                rates_te, angs_te, y_pos_te, y_neg_te, times_te)


                regressor = GPRegressor(CFG_REG.GRID[k], save_model_path=save_model_path, save_predictions=path_save_predictions)

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

                with open(args.save_path + 'angs/' + str(cfg_reg.ALPHA) + '/' + k + '/' + 'exp_' + str(i) + 'metrics.json', 'w',
                      encoding='utf-8') as f:
                    json.dump(metrics, f, indent=4)
            except:
                print('fail', i)
                bad_samples.append(i)

        with open(args.save_path + 'angs/' + str(cfg_reg.ALPHA) + '/' + k + '/' + 'bad_samples.json', 'w',
                  encoding='utf-8') as f:
            json.dump(bad_samples, f, indent=4)





