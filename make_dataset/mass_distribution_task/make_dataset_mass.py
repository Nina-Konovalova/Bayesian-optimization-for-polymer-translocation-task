import random

from tqdm import tqdm
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess as sp
from multiprocessing import Process, Queue
import multiprocessing as mp
import time
from scipy.stats import norm
import seaborn as sns

sys.path.append('../../')
from utils.utils_mass import *
import Configurations.config_mass as cfg_mass
import random
from datetime import datetime




class MakeDatasetMass:
    def __init__(self, path_to_save, parallel, num_processes, sampling, noise):
        '''
        This class make dataset where samples have the following parameters:
            - takes the random value for [shape, scale] from space
            - make gamma distribution pdf with [shape, scale]
            - for each element from gamma distribution take the results from landscape_to_distribution()
            - make summation

        As the result we save num_of_samples and for each we have:
            - image for gamma distribution
            - array of landscape_to_distribution() results
            - summation for array of landscape_to_distribution() results

        :param regime: train/test or experiment
        :param path_to_save: path to save made results
        :param num_of_samples: number of elements for dataset
        '''
        self.path_to_save = path_to_save
        self.parallel = parallel
        self.noise = noise
        self.sampling = sampling
        if self.parallel:
            self.num_processes = num_processes
            print("Folders processing using {} processes...".format(self.num_processes))



    def make_dirs(self):
        '''
            this function just checks, whether all dirs exists
            :return:
            '''
        try:
            os.mkdir(self.path_to_save)
            print(f'{self.path_to_save} has been created')
        except:
            print(f'{self.path_to_save} already exists')

        try:
            os.mkdir(self.path_to_save + self.regime)
            print(f'{self.path_to_save + self.regime} has been created')
        except:
            print(f'{self.path_to_save + self.regime} already exists')

        try:
            os.mkdir(self.path_to_save + self.regime + '/sample_images/')
        except:
            pass

        try:
            os.mkdir(self.path_to_save + self.regime + '/sample_data/')
        except:
            pass

    def data_process_sample(self, group_number):

        array = np.arange(self.frame_jump_unit * group_number, self.frame_jump_unit * (group_number + 1))
        all_samples_distributions = []
        all_samples_distributions_sum = []
        if isinstance(self.shape[0], int) or isinstance(self.shape[0], float):
            flag = False
        else:
            flag = True
        for i in tqdm(array):
            if not flag:
                g = gamma.pdf(cfg_mass.X, self.shape[i], scale=self.scale[i])
            else:
                g = gamma.pdf(cfg_mass.X, self.shape[i][0], scale=self.scale[i][0])
                for k in range(1, len(self.shape)):

                    g += gamma.pdf(cfg_mass.X, self.shape[k][i], scale=self.scale[k][i])


            plt.figure(figsize=(16, 12))
            plt.scatter(cfg_mass.X, g, marker='+', color='green')
            plt.title(f'shape {self.shape[i]}, scale {self.scale[i]}')
            plt.savefig(self.path_to_save + self.regime + '/sample_images/' + 'sample_' + str(i) + '.jpg')
            plt.close()
            final_time_d_array = []
            for k, sample in (enumerate(cfg_mass.X)):
                final_time_d_array.append(self.d[k] * g[k])
            all_samples_distributions.append(final_time_d_array)
            res = np.array(final_time_d_array).sum(axis=0)
            if self.noise:
                for z in range(len(res)):
                    sc = 0.1 * res[z]  # np.min(res)
                    n = norm.rvs(scale=sc, size=(1,))
                    res[z] = res[z] + n
            all_samples_distributions_sum.append(res)

        return np.array(self.shape[array]), np.array(self.scale[array]), np.array(all_samples_distributions_sum)

    def data_process(self, group_number):
        final_time_d_array = np.zeros_like(self.d[0])
        array = np.arange(len(cfg_mass.X))[self.frame_jump_unit * group_number: self.frame_jump_unit * (group_number + 1)]
        for k in array:
            final_time_d_array += (self.d[k] * self.g[k])

        return final_time_d_array

    def make_dataset_parallel(self, space_shape, space_scale, parallel_type='samples'):
        '''
        :param space_shape: space for random choice of shape for gamma distributions
        :param space_scale: space for random choice of scale for gamma distributions
        :return: save results (see init information)
        '''
        if isinstance(space_shape[0], int) or isinstance(space_shape[0], float):
            self.shape = np.random.uniform(low=space_shape[0], high=space_shape[1], size=self.num_samples)
            self.scale = np.random.uniform(low=space_scale[0], high=space_scale[1], size=self.num_samples)
        else:
            self.shape = [np.random.uniform(low=space_shape[0][0], high=space_shape[0][1], size=self.num_samples),
                          np.random.uniform(low=space_shape[1][0], high=space_shape[1][1], size=self.num_samples)]

            self.scale = [np.random.uniform(low=space_scale[0][0], high=space_scale[0][1], size=self.num_samples),
                          np.random.uniform(low=space_scale[1][0], high=space_scale[1][1], size=self.num_samples)]

        if not os.path.exists('../../time_distributions/time_distributions.npz'):
            prepare_distributions()

        distributions = np.load('../../time_distributions/time_distributions.npz')
        self.d = distributions['time_distributions']#distributions['y_pos'] * distributions['rate'].reshape(-1, 1)

        if parallel_type == 'samples':
            all_shape = np.empty(1)
            all_scale = np.empty(1)
            all_samples_distributions_sum = np.empty((1, 10_000))
            p = mp.Pool(self.num_processes)
            start_time = time.time()
            result = (p.map(self.data_process_sample, range(self.num_processes)))
            p.close()
            p.join()
            for m in range(self.num_processes):
                all_shape = np.concatenate((all_shape, result[m][0]))
                all_scale = np.concatenate((all_scale, result[m][1]))
                all_samples_distributions_sum = np.concatenate((all_samples_distributions_sum, result[m][2]))

            np.savez_compressed(self.path_to_save + self.regime + '/' +
                                'sample_data/' + 'samples_info.npz',
                                shape=all_shape[1:], scale=all_scale[1:],
                                # all_samples_distributions=np.array(all_samples_distributions),
                                all_samples_distributions_sum=(all_samples_distributions_sum)[1:]
                                )
            end_time = time.time()
            print(end_time - start_time)
        else:
            start_time = time.time()
            all_samples_distributions_sum = []
            for i in tqdm(range(self.num_samples)):
                self.g = gamma.pdf(cfg_mass.X, self.shape[i], scale=self.scale[i])
                plt.figure(figsize=(16, 12))
                plt.scatter(cfg_mass.X, self.g, marker='+', color='green')
                plt.title(f'shape {self.shape[i]}, scale {self.scale[i]}')
                plt.savefig(self.path_to_save + self.regime + '/sample_images/' + 'sample_' + str(i) + '.jpg')
                plt.close()
                p = mp.Pool(self.num_processes)
                result = (p.map(self.data_process, range(self.num_processes)))
                p.close()
                p.join()
                res = np.zeros_like(self.d[0])
                for m in range(self.num_processes):
                    res += (result[m])
                all_samples_distributions_sum.append(res)

            end_time = time.time()
            print(end_time - start_time)
            print(np.array(all_samples_distributions_sum).shape)


    def make_dataset(self, space_shape, space_scale):
        '''
        :param space_shape: space for random choice of shape for gamma distributions
        :param space_scale: space for random choice of scale for gamma distributions
        :return: save results (see init information)
        '''

        if self.parallel:
            modes = ['train', 'test', 'exp']
            for mode in modes:
                self.regime = mode
                self.num_samples = cfg_mass.DATA_SIZE[mode]
                self.frame_jump_unit = self.num_samples // self.num_processes
                self.make_dirs()
                self.make_dataset_parallel(space_shape, space_scale)

        else:
            print(space_shape)
            modes = ['train', 'test', 'exp']
            for mode in modes:
                self.regime = mode
                self.make_dirs()
                start_time = time.time()
                self.num_samples = cfg_mass.DATA_SIZE[mode]
                if isinstance(space_shape[0], int) or isinstance(space_shape[0], float):
                    self.shape = np.random.uniform(low=space_shape[0], high=space_shape[1], size=self.num_samples)
                    self.scale = np.random.uniform(low=space_scale[0], high=space_scale[1], size=self.num_samples)
                else:
                    self.shape = []
                    self.scale = []
                    for m in range(len(space_shape)):
                        self.shape.append(np.random.uniform(low=space_shape[m][0], high=space_shape[m][1], size=self.num_samples))
                        self.scale.append(np.random.uniform(low=space_scale[m][0], high=space_scale[m][1], size=self.num_samples))

                    # self.shape = [
                    #     np.random.uniform(low=space_shape[0][0], high=space_shape[0][1], size=self.num_samples),
                    #     np.random.uniform(low=space_shape[1][0], high=space_shape[1][1], size=self.num_samples)]
                    # self.scale = [
                    #     np.random.uniform(low=space_scale[0][0], high=space_scale[0][1], size=self.num_samples),
                    #     np.random.uniform(low=space_scale[1][0], high=space_scale[1][1], size=self.num_samples)]
                print(np.array(self.shape).shape)
                print(np.array(self.scale).shape)
                all_samples_distributions_sum = []
                all_samples_distributions_sum_1 = []
                all_samples_distributions = []
                self.make_dirs()

                if not os.path.exists('../../time_distributions/time_distributions_clew.npz'):
                    prepare_distributions('clew')

                distributions = np.load('../../time_distributions/time_distributions_clew.npz')
                d = distributions['time_distributions']
                #d = distributions['y_pos'] * distributions['rate'].reshape(-1,1)
                if isinstance(self.shape[0], int) or isinstance(self.shape[0], float):
                    flag = False
                else:
                    flag = True
                for i in tqdm(range(self.num_samples)):
                    if not self.sampling:
                        if not flag:
                            g = gamma.pdf(cfg_mass.X, self.shape[i], scale=self.scale[i])
                        else:
                            g = gamma.pdf(cfg_mass.X, self.shape[0][i], scale=self.scale[0][i])
                            for k in range(1, len(self.shape)):
                                g += gamma.pdf(cfg_mass.X, self.shape[k][i], scale=self.scale[k][i])
                    else:
                        if not flag:
                            # print(self.shape[i], self.scale[i])
                            s1 = np.random.gamma(self.shape[i], self.scale[i], 10_000)
                            s = []
                            for ff in range(len(s1)):
                                s.append(round(s1[ff]))
                            s = np.array(s)
                            (unique, counts) = np.unique(s, return_counts=True)

                            # for ax in sns.displot(s, kind="kde", bw_adjust=.25).axes.flat:
                            #     for line in ax.lines:
                            #         a = (line.get_ydata())
                            #         b = (line.get_xdata())
                            g = []
                            for mon in cfg_mass.X:
                                #place = np.where(abs(b - mon) <= 0.5)[0]

                                place = np.where(unique == mon)[0]
                                if len(place) == 0:
                                    g.append(1e-10)
                                else:
                                    # if len(place) > 1:
                                    #     print('we are lost')
                                    #     g.append(a[place[0]])
                                    g.append(counts[place][0]/2_000)

                            g = np.array(g)
                            g1 = gamma.pdf(cfg_mass.X, self.shape[i], scale=self.scale[i])
                        else:
                            s = {}
                            s[0] = np.random.gamma(self.shape[0][i], self.scale[0][i], 10_000).astype(np.int64)
                            g1 = gamma.pdf(cfg_mass.X, self.shape[0][i], scale=self.scale[0][i])
                            for k in range(1, len(self.shape)):
                                s[k] = np.random.gamma(self.shape[k][i], self.scale[k][i], 10_000).astype(np.int64)
                                g1 += gamma.pdf(cfg_mass.X, self.shape[k][i], scale=self.scale[k][i])

                            if len(s.keys()) == 2:

                                (unique1, counts1) = np.unique(s[1], return_counts=True)
                                counts1 = counts1 / 10_000
                                (unique, counts) = np.unique(s[0], return_counts=True)
                                counts = counts / 10_000
                                g_dict = {}
                                for mon in cfg_mass.X:
                                    if mon in unique:
                                        if mon not in g_dict:
                                            g_dict[mon] = counts[np.where(unique == mon)][0]
                                        else:
                                            g_dict[mon] += counts[np.where(unique == mon)][0]
                                    if mon in unique1:
                                        if mon not in g_dict:
                                            g_dict[mon] = counts1[np.where(unique1 == mon)][0]
                                        else:
                                            g_dict[mon] += counts1[np.where(unique1 == mon)][0]
                                    if mon not in unique and mon not in unique1:
                                        g_dict[mon] = 0

                                g = list(g_dict.values())

                            if len(s.keys()) == 3:
                                (unique1, counts1) = np.unique(s[1], return_counts=True)
                                counts1 = counts1 / 10_000
                                (unique, counts) = np.unique(s[0], return_counts=True)
                                counts = counts / 10_000
                                (unique2, counts2) = np.unique(s[2], return_counts=True)
                                counts2 = counts2 / 10_000
                                g_dict = {}
                                for mon in cfg_mass.X:
                                    if mon in unique:
                                        if mon not in g_dict:
                                            g_dict[mon] = counts[np.where(unique == mon)][0]
                                        else:
                                            g_dict[mon] += counts[np.where(unique == mon)][0]
                                    if mon in unique1:
                                        if mon not in g_dict:
                                            g_dict[mon] = counts1[np.where(unique1 == mon)][0]
                                        else:
                                            g_dict[mon] += counts1[np.where(unique1 == mon)][0]
                                    if mon in unique2:
                                        if mon not in g_dict:
                                            g_dict[mon] = counts2[np.where(unique2 == mon)][0]
                                        else:
                                            g_dict[mon] += counts2[np.where(unique2 == mon)][0]
                                    if mon not in unique and mon not in unique1 and mon not in unique2:
                                        g_dict[mon] = 0
                                g = list(g_dict.values())

                    g = np.array(g)

                    plt.figure(figsize=(16, 12))
                    plt.scatter(cfg_mass.X, g, marker='+', color='green')
                    plt.plot(g1)
                    if not flag:
                        plt.title(f'shape {self.shape[i]}, scale {self.scale[i]}')
                    else:
                        if len(s.keys()) == 3:
                            plt.title(f'shape {self.shape[0][i]};{self.shape[1][i]}, {self.shape[2][i]},'
                                      f' scale {self.scale[0][i]}, {self.scale[1][i]}, {self.scale[2][i]}')
                        if len(s.keys()) == 2:
                            plt.title(f'shape {self.shape[0][i]};{self.shape[1][i]},'
                                      f' scale {self.scale[0][i]}, {self.scale[1][i]}')
                    plt.savefig(self.path_to_save + self.regime + '/sample_images/' + 'sample_' + str(i) + '.jpg')
                    plt.close()
                    final_time_d_array = []
                    final_time_d_array_1 = []

                    for k, sample in (enumerate(cfg_mass.X)):
                        final_time_d_array.append(d[k] * g[k])
                        final_time_d_array_1.append(d[k] * g1[k])

                    all_samples_distributions.append(final_time_d_array)

                    res = np.array(final_time_d_array).sum(axis=0)
                    res_1 = np.array(final_time_d_array_1).sum(axis=0)
                    if self.noise:
                        for z in range(len(res)):
                            sc = 0.1*res[z]#np.min(res)
                            n = norm.rvs(scale=sc, size=(1,))
                            res[z]= res[z] + n

                    all_samples_distributions_sum.append(res)
                    all_samples_distributions_sum_1.append(res_1)

                print(self.path_to_save + self.regime + '/' +
                                    'sample_data/' + 'samples_info.npz')
                np.savez_compressed(self.path_to_save + self.regime + '/' +
                                    'sample_data/' + 'samples_info.npz',
                                    shape=self.shape, scale=self.scale,
                                    #all_samples_distributions=np.array(all_samples_distributions),
                                    all_samples_distributions_sum=np.array(all_samples_distributions_sum),
                                    all_samples_distributions_sum_1=np.array(all_samples_distributions_sum_1)
                                    )
                end_time = time.time()
                print(end_time - start_time)
                print(np.array(all_samples_distributions_sum).shape)