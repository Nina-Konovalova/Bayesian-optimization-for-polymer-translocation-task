import argparse
from make_dataset_mass import MakeDatasetMass
import sys
import subprocess
import os
import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser(description='End-to-end make dataset')
    parser.add_argument('--dir_name',
                        default='../dataset_mass_4_sampling_10_000_1_gamma_clew/',
                        type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')
    parser.add_argument('--processes',
                        default=4,
                        type=int,
                        help='number of processes for parallel work')
    parser.add_argument('--parallel',
                        default=False,
                        type=bool,
                        help='parallel work with samples')
    parser.add_argument('--noise',
                        default=False,
                        type=bool,
                        help='presence of noise in data')
    parser.add_argument('--sampling',
                        default=True,
                        type=bool,
                        help='whether to use sampling or true gamma')
    parser.add_argument('--num_of_distr',
                        default=1,
                        type=int,
                        help='number of gamma distributions')

    args = parser.parse_args()
    subprocess.call(["gfortran", "-o", "outputic_1", "../../F_1.f90"])
    sys.path.append('../../')
    import Configurations.config_mass as cfg_mass

    assert mp.cpu_count() >= args.processes, f'there are only {mp.cpu_count()} processes and you try {args.processes}'
    print(args.parallel)
    data_maker = MakeDatasetMass(args.dir_name, args.parallel, args.processes, args.sampling, args.noise)
    if args.num_of_distr == 1:
        data_maker.make_dataset(cfg_mass.SPACE[0]['domain'], cfg_mass.SPACE[1]['domain'])
    elif args.num_of_distr == 2:
        data_maker.make_dataset([cfg_mass.SPACE[0]['domain'], cfg_mass.SPACE[2]['domain']],
                                [cfg_mass.SPACE[1]['domain'], cfg_mass.SPACE[3]['domain']])
    elif args.num_of_distr == 3:
        data_maker.make_dataset([cfg_mass.SPACE[0]['domain'], cfg_mass.SPACE[2]['domain'], cfg_mass.SPACE[4]['domain']],
                                [cfg_mass.SPACE[1]['domain'], cfg_mass.SPACE[3]['domain'], cfg_mass.SPACE[5]['domain']])

    try:
        os.remove('new_input.txt')
        os.remove('new_output.txt')
        os.remove('der_output.txt')
        os.remove('outputic.exe')
    except:
        pass


if __name__ == '__main__':
    main()
