import numpy as np
from numpy import exp
from scipy.optimize import curve_fit
import sys
sys.path.append('../')
import Configurations.Config as cfg


def gaussian(x, params):
    '''
    :param x: array from 0 to number of monomers
    :param params: vector of parameters for free energy landscape
    :return: array of gaussian function with params applied to x
    '''

    eps = 1e-18
    cen = cfg.CENTERS
    #cen = np.linspace(10, 50, cfg.NUM_GAUSS)
    #cen = np.linspace(11, 81, 20)
   # cen = np.linspace(11, 81, 15)
    #sprint(params)
    wid = params[:len(cen)]
    amp = params[len(cen):-1] * params[-1] #last variable is +-1
    gauss = 0
    # print('wid,=', (wid))
    # print('amp',(amp))
    # print((cen))
    for i in range(len(cen)):
        gauss += amp[i] * 1 / (np.sqrt((wid[i] + eps) * 2 * np.pi)) * exp(-(x - cen[i]) ** 2 / (2 * (wid[i] + eps)))
    return gauss


def fitting_curves(y, function='gauss'):
    '''
    :param y: array
    :param function:
    :return: parameters of gauss function, that give best approximation for y with function
    '''
    x = np.arange(51)
    if function == 'gauss':
        wid_guess = np.ones(cfg.NUM_GAUSS) * 0.4
        amp_guess = np.ones(cfg.NUM_GAUSS) * 4
        guess = np.concatenate([wid_guess, amp_guess])
        best_vals, _ = curve_fit(gaussian, x, y, maxfev=100000, method='trf', p0=guess)
    else:
        raise ValueError('No such function for fitting')
    return best_vals
