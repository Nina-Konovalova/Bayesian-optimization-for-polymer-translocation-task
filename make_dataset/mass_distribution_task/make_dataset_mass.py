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

sys.path.append('../../')
from utils.utils_mass import *
import Configurations.config_mass as cfg_mass


class MakeDatasetMass:
    def __init__(self, regime, path_to_save, num_samples, parallel, num_processes):
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
        self.num_samples = num_samples
        self.regime = regime
        self.parallel = parallel
        if self.parallel:
            self.num_processes = num_processes
            print("Folders processing using {} processes...".format(self.num_processes))
            self.frame_jump_unit = self.num_samples // self.num_processes


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
        for i in tqdm(array):
            g = gamma.pdf(cfg_mass.X, self.shape[i], scale=self.scale[i])
            plt.figure(figsize=(16, 12))
            plt.scatter(cfg_mass.X, g, marker='+', color='green')
            plt.title(f'shape {self.shape[i]}, scale {self.scale[i]}')
            plt.savefig(self.path_to_save + self.regime + '/sample_images/' + 'sample_' + str(i) + '.jpg')
            plt.close()
            final_time_d_array = []

            for k, sample in (enumerate(cfg_mass.X)):
                final_time_d_array.append(self.d[k] * g[k])

            all_samples_distributions.append(final_time_d_array)
            #print(np.array(all_samples_distributions).shape)
            all_samples_distributions_sum.append(np.array(final_time_d_array).sum(axis=0))

        return np.array(self.shape[array]), np.array(self.scale[array]), np.array(all_samples_distributions_sum)

    def data_process(self, group_number):
        final_time_d_array = []
        all_samples_distributions = []
        all_samples_distributions_sum = []
        for k, sample in tqdm(enumerate(cfg_mass.X)[self.frame_jump_unit * group_number: self.frame_jump_unit * (group_number + 1)]):
            final_time_d_array.append(self.d[k] * self.g[k])
        all_samples_distributions.append(final_time_d_array)
        print(np.array(all_samples_distributions).shape)
        all_samples_distributions_sum.append(np.array(final_time_d_array).sum(axis=0))
        #return np.array(all_samples_distributions), np.array(all_samples_distributions_sum)

    def make_dataset_parallel(self, space_shape, space_scale):
        '''
        :param space_shape: space for random choice of shape for gamma distributions
        :param space_scale: space for random choice of scale for gamma distributions
        :return: save results (see init information)
        '''
        self.shape = np.random.uniform(low=space_shape[0], high=space_shape[1], size=self.num_samples)
        self.scale = np.random.uniform(low=space_scale[0], high=space_scale[1], size=self.num_samples)
        self.make_dirs()
        if not os.path.exists('../../time_distributions/time_distributions.npz'):
            prepare_distributions()

        distributions = np.load('../../time_distributions/time_distributions.npz')
        self.d = distributions['time_distributions']
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



    def make_dataset(self, space_shape, space_scale):
        '''
        :param space_shape: space for random choice of shape for gamma distributions
        :param space_scale: space for random choice of scale for gamma distributions
        :return: save results (see init information)
        '''

        if self.parallel:
            self.make_dataset_parallel(space_shape, space_scale)

        else:
            start_time = time.time()
            shape = np.random.uniform(low=space_shape[0], high=space_shape[1], size=self.num_samples)
            scale = np.random.uniform(low=space_scale[0], high=space_scale[1], size=self.num_samples)
            all_samples_distributions_sum = []
            all_samples_distributions = []
            self.make_dirs()

            if not os.path.exists('../../time_distributions/time_distributions.npz'):
                prepare_distributions()

            distributions = np.load('../../time_distributions/time_distributions.npz')
            d = distributions['time_distributions']
            for i in tqdm(range(self.num_samples)):
                g = gamma.pdf(cfg_mass.X, shape[i], scale=scale[i])
                plt.figure(figsize=(16, 12))
                plt.scatter(cfg_mass.X, g, marker='+', color='green')
                plt.title(f'shape {shape[i]}, scale {scale[i]}')
                plt.savefig(self.path_to_save + self.regime + '/sample_images/' + 'sample_' + str(i) + '.jpg')
                plt.close()
                final_time_d_array = []

                for k, sample in (enumerate(cfg_mass.X)):
                    final_time_d_array.append(d[k] * g[k])

                all_samples_distributions.append(final_time_d_array)

                all_samples_distributions_sum.append(np.array(final_time_d_array).sum(axis=0))

            np.savez_compressed(self.path_to_save + self.regime + '/' +
                                'sample_data/' + 'samples_info.npz',
                                shape=shape, scale=scale,
                                #all_samples_distributions=np.array(all_samples_distributions),
                                all_samples_distributions_sum=np.array(all_samples_distributions_sum)
                                )
            end_time = time.time()
            print(end_time - start_time)
            print(np.array(all_samples_distributions_sum).shape)