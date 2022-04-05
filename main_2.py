from GPyOptimization_2 import BayesianOptimization
from GPyOptimization_mass import BayesianOptimizationMass
import numpy as np
import Configurations.Config as CFG
from Configurations.arguments import parse_args
import os


def prepare_path():
    try:
        print(f'make dir {CFG.SAVE_PATH}')
        os.mkdir(CFG.SAVE_PATH)
    except:
        pass

    try:
        print(f'in dir {CFG.SAVE_PATH} make new dir {CFG.EXPERIMENT_NAME}')
        os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME)
    except:
        pass


def optimization(args):
    if args.task == 'translocation':
        x_e = np.load(CFG.EXP_PATH)
        print(f'experimental dataset contains {x_e["vecs"].shape[0]} elements')

        prepare_path()

        for i in range(len(x_e['vecs'])):
            try:
                os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/')
            except:
                pass
            gp_model = BayesianOptimization(args.model_type, x_e, i, CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/',
                                            True)
            gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS, args.acquisition_type)

    elif args.task == 'mass_distribution':
        '''
        main code for mass distribution
        '''
        x_e = np.load(CFG.EXP_PATH)
        print(f'experimental dataset contains {x_e["shape"].shape[0]} elements')

        prepare_path()

        for i in range(len(x_e['shape'])):
            try:
                os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/')
            except:
                pass

            gp_model = BayesianOptimizationMass(args.model_type, x_e, i, CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/',
                                            True)
            gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS, args.acquisition_type)



def main():
    args = parse_args()
    optimization(args)



if __name__ == '__main__':
    main()
