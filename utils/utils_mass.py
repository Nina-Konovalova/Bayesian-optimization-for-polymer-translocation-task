import subprocess
import sys
import numpy as np
from sklearn.metrics import mean_squared_error as mse

sys.path.append('../')
from utils.data_frotran_utils import *
from numpy.random import seed

seed(42)


def sample_from_mass_distr(shape, scale, n=100):
    '''
    gamma = x^(k-1) * exp(-x/theta) / (theta^k Gamma(k))
    :param shape: k
    :param scale: theta
    :param n: number of samples
    :return: random samples from gamma distribution
    '''
    return np.random.gamma(shape, scale, n)


def from_mass_to_landscape(N, const):
    '''
    some magic
    :param N: number of monomers
    :param const: const for distribution
    :return: np.ones(N)*const
    '''

    landscape = np.ones(N) * const
    return landscape


def make_data(all_samples_distributions_sum_real, all_samples_distributions_sum_train):
    '''
    :param all_samples_distributions_sum_real: distributions for real data (1 sample)
    :param all_samples_distributions_sum_train: distributions for train data (train/test size samples)
    :return: mse error
    '''

    f = []
    print(all_samples_distributions_sum_real)
    for i in (range(len(all_samples_distributions_sum_train))):
        f.append(mse(all_samples_distributions_sum_real, all_samples_distributions_sum_train[i]))

    return np.array(f)


def landscape_to_distribution_mass(N, energy_const):
    '''
    for different number of monomers and constant landscape this function solves fp equation get time distributions and rate ->
    summation p*time_success_translocation + (1-rate)*time_unsuccess_translocation

    :param energy_const: constant for energy profile
    :param N: number of monomers
    :return: results of solving fp equation
    '''
    N = int(N)
    landscape = from_mass_to_landscape(N, energy_const)
    make_input_file(landscape, N=N)
    subprocess.check_output(["./outputic"])
    rate, times, y_pos_new, y_neg_new = read_data(N=N)
    final_time_d = (rate * y_pos_new) + (1 - rate) * y_neg_new

    return final_time_d
