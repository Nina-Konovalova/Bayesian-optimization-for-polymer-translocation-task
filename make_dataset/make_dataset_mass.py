from tqdm import tqdm
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../')
from utils.utils_mass import *
import Configurations.config_mass as cfg_mass


class MakeDatasetMass:
    def __init__(self, regime, path_to_save, num_samples):
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

    def make_dataset(self, space_shape, space_scale):
        '''
        :param space_shape: space for random choice of shape for gamma distributions
        :param space_scale: space for random choice of scale for gamma distributions
        :return: save results (see init information)
        '''
        shape = np.random.uniform(low=space_shape[0], high=space_shape[1], size=self.num_samples)
        scale = np.random.uniform(low=space_scale[0], high=space_scale[1], size=self.num_samples)
        all_samples_distributions_sum = []
        all_samples_distributions = []
        self.make_dirs()
        for i in (range(self.num_samples)):
            g = gamma.pdf(cfg_mass.X, shape[i], scale=scale[i])

            plt.figure(figsize=(16, 12))
            plt.scatter(cfg_mass.X, g, marker='+', color='green')
            plt.title(f'shape {shape[i]}, scale {scale[i]}')
            plt.savefig(self.path_to_save + self.regime + '/sample_images/' + 'sample_' + str(i) + '.jpg')

            final_time_d_array = []

            for i, sample in tqdm(enumerate(cfg_mass.X)):
                d = landscape_to_distribution_mass(sample, cfg_mass.ENERGY_CONST)

                final_time_d_array.append(d * g[i])

            all_samples_distributions.append(final_time_d_array)
            print(np.array(all_samples_distributions).shape)
            all_samples_distributions_sum.append(np.array(final_time_d_array).sum(axis=0))

        np.savez_compressed(self.path_to_save + self.regime + '/' +
                            'sample_data/' + 'samples_info.npz',
                            shape=shape, scale=scale,
                            all_samples_distributions=np.array(all_samples_distributions),
                            all_samples_distributions_sum=np.array(all_samples_distributions_sum)
                            )
