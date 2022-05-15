import subprocess
import sys
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skfda import FDataGrid
import matplotlib.pyplot as plt

import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction.feature_extraction._fpca import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial

sys.path.append('../')
from utils.data_frotran_utils import *
from numpy.random import seed
import Configurations.config_mass as cfg_mass
import gp_regression_code.GP_mass_config as cfg_gp_mass
import multiprocessing as mp
from functools import partial
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


def make_data(all_samples_distributions_sum_real, all_samples_distributions_sum_train, shape_exp=None, scale_exp=None, \
              shape_train=None, scale_train=None, e=0, make_plots=False):
    '''
    :param all_samples_distributions_sum_real: distributions for real data (1 sample)
    :param all_samples_distributions_sum_train: distributions for train data (train/test size samples)
    :return: mse error
    '''

    f = []

    for i in (range(len(all_samples_distributions_sum_train))):

        f.append(mse(np.log(all_samples_distributions_sum_real), np.log(all_samples_distributions_sum_train[i])))
        if make_plots:
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS)
            except:
                pass
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/')
            except:
                pass
            plt.figure(figsize=(16, 12))
            plt.title(f'time_distribution with mse {str(f[-1])}')
            plt.plot(np.log(all_samples_distributions_sum_real), label=f'real_distr {round(shape_exp, 2), round(scale_exp, 2)}',
                     color='r')
            plt.plot(np.log(all_samples_distributions_sum_train[i]),
                     label=f'train_distr {round(shape_train[i], 2), round(scale_train[i], 2)}', color='g')
            plt.legend()
            plt.savefig(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/' + str(i) + 'jpg')

            plt.close()

    return np.array(f)

def make_data_vector_output(all_samples_distributions_sum_real, all_samples_distributions_sum_train, mult=1e-4, shape_exp=None, scale_exp=None, \
              shape_train=None, scale_train=None, e=0, make_plots=False):
    '''
    :param all_samples_distributions_sum_real: distributions for real data (1 sample)
    :param all_samples_distributions_sum_train: distributions for train data (train/test size samples)
    :return: mse error
    '''

    f = []

    for i in (range(len(all_samples_distributions_sum_train))):
        f1 = mse(np.log(all_samples_distributions_sum_real)[100:], np.log(all_samples_distributions_sum_train[i])[100:]) * mult
        f2 = mse((all_samples_distributions_sum_real[:100]), (all_samples_distributions_sum_train[i][:100]))
        f.append(np.hstack((f1, f2)))
        if make_plots:
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS)
            except:
                pass
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/')
            except:
                pass
            fig, ax = plt.subplots(1, 2, figsize=(16, 12))
            plt.title(f'time_distribution with mse {str(f[-1].sum())}')
            ax[0].plot(np.log(all_samples_distributions_sum_real)[100:], label=f'real_distr {round(shape_exp, 2), round(scale_exp, 2)}',
                       color='r')
            ax[0].plot(np.log(all_samples_distributions_sum_train[i])[100:],
                     label=f'train_distr {round(shape_train[i], 2), round(scale_train[i], 2)}', color='g')
            ax[1].plot((all_samples_distributions_sum_real)[:100],
                       label=f'real_distr {round(shape_exp, 2), round(scale_exp, 2)}',
                       color='r')
            ax[1].plot((all_samples_distributions_sum_train[i])[:100],
                       label=f'train_distr {round(shape_train[i], 2), round(scale_train[i], 2)}', color='g')

            ax[0].legend()
            ax[1].legend()
            plt.savefig(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/' + str(i) + 'jpg')

            plt.close()
    print(f)
    return np.array(f)

def fpca(data):
    data = FDataGrid(np.log(data), np.arange(len(np.log(data)[0])),
           dataset_name='time_distribution',
           argument_names=['t'],
           coordinate_names=['p(t)'])

    fpca_discretized = FPCA(n_components=3, centering=True)
    fpca_discretized.fit(data)
    return fpca_discretized

def make_data_vector_fpca_output(data,
                                 fpca_discretized,
                                 mult=1e-4, shape_exp=None, scale_exp=None, \
                                 shape_train=None, scale_train=None, e=0, make_plots=False):
    '''
    :param all_samples_distributions_sum_real: distributions for real data (1 sample)
    :param all_samples_distributions_sum_train: distributions for train data (train/test size samples)
    :return: mse error
    '''

    f = []
    all_samples_distributions_sum_all = FDataGrid(np.log(data),
                                                  np.arange(len(np.log(data)[0])),
                                                 dataset_name='time_distribution',
                                                 argument_names=['t'],
                                                 coordinate_names=['p(t)'])

    all_samples_distributions_sum_all_components = fpca_discretized.transform(all_samples_distributions_sum_all)

    for i in (range(1, len(data))):
        f1 = (all_samples_distributions_sum_all_components[0] - all_samples_distributions_sum_all_components[i])**2
        f.append(f1)
        if make_plots:
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS)
            except:
                pass
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/')
            except:
                pass
            fig, ax = plt.subplots(1, 2, figsize=(16, 12))
            plt.title(f'time_distribution with mse {str(f[-1].sum())}')
            ax[0].plot(np.log(all_samples_distributions_sum_real)[100:], label=f'real_distr {round(shape_exp, 2), round(scale_exp, 2)}',
                       color='r')
            ax[0].plot(np.log(all_samples_distributions_sum_train[i])[100:],
                     label=f'train_distr {round(shape_train[i], 2), round(scale_train[i], 2)}', color='g')
            ax[1].plot((all_samples_distributions_sum_real)[:100],
                       label=f'real_distr {round(shape_exp, 2), round(scale_exp, 2)}',
                       color='r')
            ax[1].plot((all_samples_distributions_sum_train[i])[:100],
                       label=f'train_distr {round(shape_train[i], 2), round(scale_train[i], 2)}', color='g')

            ax[0].legend()
            ax[1].legend()
            plt.savefig(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/' + str(i) + 'jpg')

            plt.close()

    return np.array(f)


def data_process(f, frame_jump_unit, all_samples_distributions_sum_real,
                 all_samples_distributions_sum_train,
                 make_plots, shape_exp, scale_exp, shape_train, scale_train, e, group_number):
    if frame_jump_unit * (group_number + 1) <= len(scale_train):
        array = np.arange(frame_jump_unit * group_number, frame_jump_unit * (group_number + 1))
    else:
        array = np.arange(frame_jump_unit * group_number, len(scale_train))

    for i in tqdm(array):

        f.append(mse(np.log(all_samples_distributions_sum_real), np.log(all_samples_distributions_sum_train[i])))

        if make_plots:
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS)
            except:
                pass
            try:
                os.mkdir(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/')
            except:
                pass
            plt.figure(figsize=(16, 12))
            plt.title(f'time_distribution with mse {str(f[-1])}')
            plt.plot(np.log(all_samples_distributions_sum_real),
                     label=f'real_distr {round(shape_exp, 2), round(scale_exp, 2)}',
                     color='r')
            plt.plot(np.log(all_samples_distributions_sum_train[i]),
                     label=f'train_distr {round(shape_train[i], 2), round(scale_train[i], 2)}', color='g')
            plt.legend()
            plt.savefig(cfg_gp_mass.PATH_TO_SAVE_PLOTS + str(e) + '/' + str(i) + 'jpg')

            plt.close()

    return np.array(f)


def make_data_parallel(all_samples_distributions_sum_real, all_samples_distributions_sum_train, num_processes=4,
                       shape_exp=None, scale_exp=None, shape_train=None, scale_train=None, e=0, make_plots=False):
    '''
    :param all_samples_distributions_sum_real: distributions for real data (1 sample)
    :param all_samples_distributions_sum_train: distributions for train data (train/test size samples)
    :return: mse error
    '''

    assert mp.cpu_count() >= num_processes, f'there are only {mp.cpu_count()} processes and you try {num_processes}'
    f = []
    frame_jump_unit = len(all_samples_distributions_sum_train) // num_processes
    p = mp.Pool(num_processes)

    result = (p.map(partial(data_process, f, frame_jump_unit, all_samples_distributions_sum_real,
                            all_samples_distributions_sum_train,
                            make_plots, shape_exp, scale_exp, shape_train, scale_train, e), range(num_processes)))
    p.close()
    p.join()
    f = result[0]
    for m in range(1, num_processes):
        f = np.concatenate((f, result[m]))
    return f


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

    np.savez_compressed('../../time_distributions/' + 'time_distributions_longer.npz',
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
