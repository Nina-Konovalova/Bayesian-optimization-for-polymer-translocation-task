import argparse
from make_dataset_mass import MakeDatasetMass
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description='End-to-end make dataset')
    parser.add_argument('--dir_name',
                        default='../dataset_mass_0/',
                        type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')
    parser.add_argument('--num_of_samples',
                        default=12,
                        type=int,
                        help='Number of all sample')
    parser.add_argument('--mode',
                        default="train",
                        type=str,
                        help='train/test/exp')

    args = parser.parse_args()
    subprocess.call(["gfortran", "-o", "outputic", "../F.f90"])
    sys.path.append('../')
    import Configurations.config_mass as cfg_mass

    data_maker = MakeDatasetMass(args.mode, args.dir_name, args.num_of_samples)
    data_maker.make_dataset(cfg_mass.SPACE[0]['domain'], cfg_mass.SPACE[1]['domain'])


if __name__ == '__main__':
    main()
