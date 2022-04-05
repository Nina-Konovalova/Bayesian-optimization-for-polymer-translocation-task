import numpy as np
import seaborn as sns
import argparse
import os
sns.set_theme()
from make_dataset import MakeDataset
import config_dataset as cfg_dataset
import subprocess
from numpy import exp

def main():
    '''
    :return:
    '''
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--dir_name',
                        default='../dataset_3_gaussians_small_0/',
                        type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')
    parser.add_argument('--num_of_all_g',
                        default=3,
                        type=int,
                        help='Number of all gaussians')
    parser.add_argument('--plot', default=False, type=bool, help='whether to illustrate each sample of dataset')

    args = parser.parse_args()

    try:
        os.mkdir(args.dir_name)
    except:
        pass

    subprocess.call(["gfortran", "-o", "outputic", "../F.f90"])

    make_data = MakeDataset(args.num_of_all_g, args.dir_name)
    if args.num_of_all_g == 3:

        make_data.one_gaussian(cfg_dataset.STD_AMP_1, cfg_dataset.STD_BIAS_1,
                               cfg_dataset.AMPL_AMP_1, cfg_dataset.AMPL_BIAS_1,
                               cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_1)
        print('one gaussian ready!')
        make_data.two_gaussians(cfg_dataset.STD_AMP_2, cfg_dataset.STD_BIAS_2,
                                cfg_dataset.AMPL_AMP_2, cfg_dataset.AMPL_BIAS_2,
                                cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_2)
        print('two gaussian ready!')
        make_data.three_gaussians(cfg_dataset.STD_AMP_3, cfg_dataset.STD_BIAS_3,
                                  cfg_dataset.AMPL_AMP_3, cfg_dataset.AMPL_BIAS_3,
                                  cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_3)
        print('three gaussian ready!')
    elif args.num_of_all_g == 4:
        make_data.one_gaussian(cfg_dataset.STD_AMP_1, cfg_dataset.STD_BIAS_1,
                               cfg_dataset.AMPL_AMP_1, cfg_dataset.AMPL_BIAS_1,
                               cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_1)
        print('one gaussian ready!')
        make_data.two_gaussians(cfg_dataset.STD_AMP_2, cfg_dataset.STD_BIAS_2,
                                cfg_dataset.AMPL_AMP_2, cfg_dataset.AMPL_BIAS_2,
                                cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_2)
        print('two gaussian ready!')
        make_data.three_gaussians(cfg_dataset.STD_AMP_3, cfg_dataset.STD_BIAS_3,
                                  cfg_dataset.AMPL_AMP_3, cfg_dataset.AMPL_BIAS_3,
                                  cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_3)
        print('three gaussian ready!')
        make_data.four_gaussians(cfg_dataset.STD_AMP_4, cfg_dataset.STD_BIAS_4,
                                  cfg_dataset.AMPL_AMP_4, cfg_dataset.AMPL_BIAS_4,
                                  cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_4)
        print('four gaussian ready!')
    elif args.num_of_all_g == 5:
        make_data.one_gaussian(cfg_dataset.STD_AMP_1, cfg_dataset.STD_BIAS_1,
                               cfg_dataset.AMPL_AMP_1, cfg_dataset.AMPL_BIAS_1,
                               cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_1)
        print('one gaussian ready!')
        make_data.two_gaussians(cfg_dataset.STD_AMP_2, cfg_dataset.STD_BIAS_2,
                                cfg_dataset.AMPL_AMP_2, cfg_dataset.AMPL_BIAS_2,
                                cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_2)
        print('two gaussian ready!')
        make_data.three_gaussians(cfg_dataset.STD_AMP_3, cfg_dataset.STD_BIAS_3,
                                  cfg_dataset.AMPL_AMP_3, cfg_dataset.AMPL_BIAS_3,
                                  cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_3)
        print('three gaussian ready!')
        make_data.four_gaussians(cfg_dataset.STD_AMP_4, cfg_dataset.STD_BIAS_4,
                                  cfg_dataset.AMPL_AMP_4, cfg_dataset.AMPL_BIAS_4,
                                  cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_4)
        print('four gaussian ready!')
        make_data.five_gaussians(cfg_dataset.STD_AMP_5, cfg_dataset.STD_BIAS_5,
                                  cfg_dataset.AMPL_AMP_5, cfg_dataset.AMPL_BIAS_5,
                                  cfg_dataset.MODE, cfg_dataset.SAMPLES_NUM_5)
        print('five gaussian ready!')
    else:
        raise ValueError("no such number of gaussians is implemented")

    if args.plot:
        make_data.draw_all_dataset(cfg_dataset.MODE)
        print('plotting done!')

    make_data.make_all_dataset(cfg_dataset.MODE)
    print('all data!')
    print('check shapes!')
    make_data.check_data_shape(args.dir_name + 'exp_gaussians_' + str(args.num_of_all_g) + '_' + cfg_dataset.MODE + '.npz')
    try:
        os.remove('new_input.txt')
        os.remove('new_output.txt')
        os.remove('der_output.txt')
        os.remove('outputic.exe')
    except:
        pass


if __name__ == '__main__':
    main()
