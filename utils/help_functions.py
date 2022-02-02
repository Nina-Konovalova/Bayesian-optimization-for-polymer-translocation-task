import numpy as np
from sklearn.metrics import mean_squared_error as mse
import sys
sys.path.append("../")
import gp_regression_code.GP_config as CFG_REG
import Configurations.Config as CFG_OPT


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


def make_dict(path):
    d = {}
    dat = np.load(path)
    d['vecs'] = dat['vecs']
    d['angs'] = dat['angs']
    d['rates'] = dat['rates']
    d['y_pos'] = dat['y_pos']
    d['y_neg'] = dat['y_neg']
    d['times'] = dat['times']
    return d

def make_data(rate_real, angs_real, y_pos_real, y_neg_real, times_real, rate, angs, y_pos, y_neg, times, task):
    '''
    see all input variables in description for function()
    :return: array for loss function for each element of considered dataset
    '''
    f = []
    for i in (range(len(rate))):
        f.append(function(rate_real, angs_real, y_pos_real, y_neg_real, times_real,
                          rate[i], angs[i], y_pos[i], y_neg[i], times[i], task)[0])
    return np.array(f)


def function(rate_real, angs_real, y_pos_real, y_neg_real, times_real,
             rate_new, angs_new,  y_pos_new, y_neg_new, times_new, task):
    '''
        :param task: "approximation" or "optimization"
        :param times_new:
        :param rate_real: successful rate translocation for real free energy landscape (which is used in experiment)
        :param angs_real: successful angles from log time distributions (see function angle()) for real free energy landscape (which is used in experiment)
        :param y_pos_real: successful time distribution for real free energy landscape (which is used in experiment)
        :param y_neg_real: unsuccessful time distribution for real free energy landscape (which is used in experiment)
        :param times_real:
        :param rate_new: successful rate translocation for considering free energy landscape
        :param angs_new: uccessful angles from log time distributions (see function angle()) for considering free energy landscape
        :param y_pos_new: successful time distribution for considering free energy landscape
        :param y_neg_new: unsuccessful time distribution for considering free energy landscape
        :return: combinaton of losses between input variables
        '''
    loss_true = mse((y_pos_new), (y_pos_real))
    loss_false = mse((y_neg_new), (y_neg_real[:]))
    loss_mse = loss_true + loss_false
    loss_rate = abs(rate_new - rate_real)
    loss_angs = (abs(angs_new - angs_real)).sum()
    if task == 'approximation':
        alpha = CFG_REG.ALPHA
    else:
        alpha = CFG_OPT.ALPHA
    diff_new = loss_angs + loss_rate * alpha
    loss_times = (abs(times_new - times_real)).sum()

    return diff_new, loss_rate, loss_angs, loss_mse, loss_times