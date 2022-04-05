import argparse
from make_dataset_mass import MakeDatasetMass
import sys
import subprocess
import os
import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser(description='End-to-end make dataset')
    parser.add_argument('--dir_name',
                        default='../dataset_mass_0/',
                        type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')
    parser.add_argument('--num_of_samples',
                        default=100,
                        type=int,
                        help='Number of all sample')
    parser.add_argument('--mode',
                        default="train",
                        type=str,
                        help='train/test/exp')
    parser.add_argument('--processes',
                        default=4,
                        type=int,
                        help='number of processes for parallel work')
    parser.add_argument('--parallel',
                        default=False,
                        type=bool,
                        help='parallel work with samples')

    args = parser.parse_args()
    subprocess.call(["gfortran", "-o", "outputic", "../../F.f90"])
    sys.path.append('../../')
    import Configurations.config_mass as cfg_mass

    assert mp.cpu_count() >= args.processes, f'there are only {mp.cpu_count()} processes and you try {args.processes}'
    print(args.parallel)
    data_maker = MakeDatasetMass(args.mode, args.dir_name, args.num_of_samples, args.parallel, args.processes)
    data_maker.make_dataset(cfg_mass.SPACE[0]['domain'], cfg_mass.SPACE[1]['domain'])

    try:
        os.remove('new_input.txt')
        os.remove('new_output.txt')
        os.remove('der_output.txt')
        os.remove('outputic.exe')
    except:
        pass


if __name__ == '__main__':
    main()
