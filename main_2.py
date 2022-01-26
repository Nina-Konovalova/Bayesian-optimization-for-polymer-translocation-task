from GPyOptimization_2 import BayesianOptimization
import numpy as np
import Configurations.Config as CFG
from Configurations.arguments import parse_args
import os


def optimization(args):
    x_e = np.load(CFG.EXP_PATH)
    try:
        os.mkdir(CFG.SAVE_PATH)
    except:
        pass
    try:
        os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME)
    except:
        pass

    for i in range(len(x_e['vecs'])):
        try:
            os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/')
        except:
            pass
        gp_model = BayesianOptimization(args.model_type, x_e, i, args.kernel_type)
        gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS,
                                   CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/', args.acquisition_type)



def main():
    args = parse_args()
    optimization(args)



if __name__ == '__main__':
    main()
