import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')

    # configuration and regime
    parser.add_argument('-model', '--model_type', default='GP', type=str,
                        metavar='NAME',
                        help='type of used model (GP, GP_MCMC, WarpedGP)')
    parser.add_argument('-a', '--acquisition_type', default='MPI', type=str,
                        help='acquisition_type')
    parser.add_argument('-n', '--normalize', default=True, type=bool,
                        help='if output should be normalized')
    parser.add_argument('-c', '--num_cores', default=-1, type=int,
                        help='num_cores')
    parser.add_argument('-eval', '--evaluator_type', default='lbfgs', type=str,
                        help='type of acquisition function to use. - ‘lbfgs’: L-BFGS. - ‘DIRECT’: Dividing Rectangles. - ‘CMA’: covariance matrix adaptation')

    args = parser.parse_args()
    return args

