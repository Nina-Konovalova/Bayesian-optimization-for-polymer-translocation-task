from GPyOptimization import BayesianOptimization
import numpy as np
import pandas as pd
from arguments import parse_args


def optimization(args):
    x_e = pd.read_csv(args.path_experiments)
    x_e = np.array(x_e)

    x_parametr_pol = np.array(pd.read_csv('X_train_gauss.csv'))[:, 1:]
    print('shape', x_parametr_pol.shape)
    y_train = np.array(pd.read_csv('Y_train.csv'))

    for i in range(6, len(x_e)):
        GP_model = BayesianOptimization(args.model_type, x_e[i], i)
        GP_model.optimization_step(x_parametr_pol, y_train, args.number_steps,
                                   args.path_for_save, args.acquisition_type)


def main():
    args = parse_args()
    optimization(args)


if __name__ == '__main__':
    main()
