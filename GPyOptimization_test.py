import os
import sys
from tqdm import tqdm
import GPyOpt
from numpy.random import seed
import Configurations.config_mass as cfg_mass
import Configurations.Config as CFG
from scipy.stats import gamma
sys.path.append('/')
from utils.gauss_fit import *
from utils.data_frotran_utils import *
from utils.help_functions import *
import subprocess
import matplotlib.pyplot as plt
from utils.utils_mass import *
import json


from GPy.models import GPRegression

from emukit.model_wrappers import GPyModelWrapper
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from target_vector_estimation.l2_bayes_opt.acquisitions import (
    L2NegativeLowerConfidenceBound as L2_LCB,
    L2ExpectedImprovement as L2_EI)

subprocess.call(["gfortran", "-o", "outputic", "F.f90"])
seed(42)


class BayesianOptimizationMass:
    def __init__(self):
        self.target = np.array([4.5, 2.0, 0.8])
        self.xmin, self.xmax = 0, 2 * np.pi


    @staticmethod
    def h_noiseless(x):
        f1 = 5 + np.sin(x)
        f2 = 2 + 1.3 * np.cos(x)
        f3 = np.tanh(x)
        return np.hstack((f1, f2, f3))

    def h(self, x):
        res = self.h_noiseless(x)
        return res + norm.rvs(scale=0.1, size=res.shape)

    def d_noiseless(self, x):
        return ((self.h_noiseless(x) - self.target) ** 2).sum(axis=1)[:, None]

    def d(self, x):
        return ((self.h(x) - self.target) ** 2).sum(axis=1)[:, None]

    @staticmethod
    def plots(g, sample, path_to_save):
        shape = sample[0]
        scale = sample[1]
        plt.figure(figsize=(16, 12))
        plt.scatter(cfg_mass.X, g, marker='+', color='green')
        plt.title(f'shape {shape}, scale {scale}')
        plt.savefig(path_to_save + 'sample_' + str(round(shape, 2)) + '_' + str(round(scale, 2)) + '.jpg')
        plt.close()


    def optimization_step(self):

        n_samples = 5
        parameter_space = ParameterSpace([ContinuousParameter("x", self.xmin, self.xmax)])
        latin_design = LatinDesign(parameter_space=parameter_space)
        X0 = latin_design.get_samples(n_samples)
        Y0 = self.h(X0)
        D0 = ((Y0 - self.target) ** 2).sum(axis=1)

        self.model = GPRegression(X0, Y0) # многомерный выход у модели
        self.model_wrapped = GPyModelWrapper(self.model) # просто обернули модель в эмукит
        acq = L2_LCB(model=self.model_wrapped, target=self.target)

        fit_update = lambda a, b: self.model.optimize_restarts(verbose=False)
        bayesopt_loop = BayesianOptimizationLoop(
            model=self.model_wrapped, space=parameter_space, acquisition=acq)
        bayesopt_loop.iteration_end_event.append(fit_update)
        bayesopt_loop.run_loop(self.h, 55)
        print('end')
        return None


if __name__ == '__main__':
    optimizer = BayesianOptimizationMass()
    optimizer.optimization_step()
