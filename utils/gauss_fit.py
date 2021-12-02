import numpy as np
from numpy import exp
from scipy.optimize import curve_fit


def gaussian(x, *params):
    cen = np.linspace(3, 53, 10)
    wid = params[:len(cen)]
    amp = params[len(cen):]
    gauss = 0
    # print(cen)
    for i in range(len(cen)):
        gauss += amp[i] * 1 / (np.sqrt(wid[i] * 2 * np.pi)) * exp(-(x - cen[i]) ** 2 / (2 * wid[i]))
    return gauss


def fitting_curves(y, function='gauss'):
    x = np.arange(51)
    if function == 'gauss':
        wid_guess = np.ones(10) * 0.4
        amp_guess = np.ones(10) * 4
        guess = np.concatenate([wid_guess, amp_guess])
        best_vals, _ = curve_fit(gaussian, x, y, maxfev=100000, method='trf', p0=guess)
    return best_vals
