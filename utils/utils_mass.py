import subprocess
import sys
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import os
from tqdm import tqdm

sys.path.append('../')
from utils.data_frotran_utils import *
from numpy.random import seed
import Configurations.config_mass as cfg_mass

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

    for i in (range(len(all_samples_distributions_sum_train))):
        f.append(mse(all_samples_distributions_sum_real, all_samples_distributions_sum_train[i]))

    return np.array(f)

def prepare_distributions():
    final_time_d_array = []
    y_pos = []
    y_neg = []
    rate = []
    times = []
    for i, sample in tqdm(enumerate(cfg_mass.X)):
        d, y_pos_new, y_neg_new, rate_new, times_new = landscape_to_distribution_mass(sample, cfg_mass.ENERGY_CONST)
        final_time_d_array.append(d)
        y_pos.append(y_pos_new)
        y_neg.append(y_neg_new)
        rate.append(rate_new)
        times.append(times_new)

    try:
        os.mkdir('../../time_distributions')
    except:
        pass

    np.savez_compressed('../../time_distributions/' + 'time_distributions.npz',
                        time_distributions=np.array(final_time_d_array), rate=np.array(rate), times=np.array(times),
                        y_pos=np.array(y_pos), y_neg=np.array(y_neg))

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

    return final_time_d, y_pos_new, y_neg_new, rate, times
