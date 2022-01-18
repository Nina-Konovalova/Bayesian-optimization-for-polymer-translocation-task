from GPyOptimization import BayesianOptimization
import numpy as np
import pandas as pd
from Configurations.arguments import parse_args


def optimization(args):
    #x_e = pd.read_csv(args.path_experiments)
    #x_e = np.array(x_e)

    x_e = np.load('new_test_10.npz')['vecs']

    print('experiment shape', x_e.shape)
    #x_parameter_pol = np.array(pd.read_csv('X_train_gauss_20.csv').dropna())
    x_parameter_pol0 = np.load('new_train_10.npz')['vecs']
    print(x_parameter_pol0.shape)
    x_parameter_pol1 = np.load('new_train_1_10.npz')['vecs']
    print(x_parameter_pol1.shape)
    x_parameter_pol2 = np.load('new_train_2_10.npz')['vecs']
    print(x_parameter_pol2.shape)
    x_parameter_pol = np.concatenate([x_parameter_pol0, x_parameter_pol1])
    x_parameter_pol = np.concatenate([x_parameter_pol, x_parameter_pol2])
    print('shape', x_parameter_pol.shape)

    for i in range(2, len(x_e)):
        print('experiment', i)
        gp_model = BayesianOptimization(args.model_type, x_e[i], i, args.kernel_type)
        gp_model.optimization_step(x_parameter_pol, args.number_steps,
                                   args.path_for_save, args.acquisition_type)


def main():
    args = parse_args()
    optimization(args)


if __name__ == '__main__':
    main()