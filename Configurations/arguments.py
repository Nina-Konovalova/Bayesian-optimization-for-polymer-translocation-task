import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')

    # configuration and regime
    parser.add_argument('-model', '--model_type', default='GP', type=str,
                        metavar='NAME',
                        help='type of used model (GP, GP_MCMC, WarpedGP)')
    parser.add_argument('-steps', '--number_steps', default=70, type=int,
                        help='number of optimization steps')
    parser.add_argument('-a', '--acquisition_type', default='MPI', type=str,
                        help='acquisition_type')
    parser.add_argument('-n', '--normalize', default=True, type=bool,
                        help='if output should be normalized')
    parser.add_argument('-c', '--num_cores', default=-1, type=int,
                        help='num_cores')
    parser.add_argument('-eval', '--evaluator_type', default='lbfgs', type=str,
                        help='type of acquisition function to use. - ‘lbfgs’: L-BFGS. - ‘DIRECT’: Dividing Rectangles. - ‘CMA’: covariance matrix adaptation')
    parser.add_argument('-p', '--path_for_save', default='/content/drive/MyDrive/research/experiments_5/RBF/', type=str,
                        help='path_for_saving_images')
    parser.add_argument('-exp', '--path_experiments', default='experimental_data.csv', type=str,
                        help='path_for_experiments')
    parser.add_argument('--x_parameter_pol_path', default='init_dataset_3000.npz', type=str,
                        help='path_for_train_data')
    parser.add_argument('-kernel', '--kernel_type', default='RatQuad', type=str,
                        help='type of kernel')

    args = parser.parse_args()
    return args

