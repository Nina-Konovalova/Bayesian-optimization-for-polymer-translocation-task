from GPyOptimization_2 import BayesianOptimization
from GPyOptimization_mass import BayesianOptimizationMass
from GpyOptimization_mass_vector_output import BayesianOptimizationMassVectorOutput
from GpyOptimization_mass_fpca_vector_output import BayesianOptimizationMassFunctionalOutput
import numpy as np
import Configurations.Config as CFG
from Configurations.arguments import parse_args
import os
import multiprocessing as mp
from functools import partial


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


def data_process(x_e, frame_jump_unit, model_type, acquisition_type, output, group_number):
    if frame_jump_unit * (group_number + 1) <= len(x_e['shape']):
        array = np.arange(frame_jump_unit * group_number, frame_jump_unit * (group_number + 1))
    else:
        array = np.arange(frame_jump_unit * group_number, len(x_e['shape']))
    for i in array:
        try:
            os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/')
        except:
            pass
        if output == 'scalar':
            print('==============SCALAR====================')
            gp_model = BayesianOptimizationMass(model_type, x_e, i,
                                                CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/',
                                                True)
            gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS, acquisition_type)
        elif output == 'vector':
            print('==============VECTOR====================')
            gp_model = BayesianOptimizationMassVectorOutput(x_e, i,
                                                CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/',
                                                True)
            gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS)

# def help( x_e, frame_jump_unit, model_type, acquisition_type, group_number):
#     print('help')
#     try:
#         os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(0) + '/')
#     except:
#         pass


def optimization(args):
    if args.task == 'translocation':
        x_e = np.load(CFG.EXP_PATH)
        print(f'experimental dataset contains {x_e["vecs"].shape[0]} elements')

        prepare_path()

        for i in range(80, len(x_e['vecs'])):

            try:
                os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/')
            except:
                pass
            gp_model = BayesianOptimization(args.model_type, x_e, i, CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/',
                                            False)
            gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS, args.acquisition_type)

    elif args.task == 'mass_distribution':
        '''
        main code for mass distribution
        '''
        x_e = np.load(CFG.EXP_PATH, allow_pickle=True)
        if args.num_gamma == 1:
            print(f'experimental dataset contains {x_e["shape"].shape[0]} elements')
            len_exp = len(x_e['shape'])
        else:
            print(f'experimental dataset contains {x_e["shape"].shape[1]} elements')
            len_exp = len(x_e['shape'][0])

        prepare_path()
        if args.parallel:
            assert mp.cpu_count() >= args.num_processes
            frame_jump_unit = len(x_e['shape']) // args.num_processes
            p = mp.Pool(args.num_processes)
            num_processes = args.num_processes
            x_e = {'shape': x_e['shape'],
                   'scale': x_e['scale'],
                   'all_samples_distributions_sum': x_e['all_samples_distributions_sum']}
            p.map(partial(data_process, x_e, frame_jump_unit, args.model_type, args.acquisition_type, args.output),
                            range(num_processes))
            p.map(partial(help, x_e, frame_jump_unit, args.model_type, args.acquisition_type), range(num_processes) )
            p.close()
            p.join()
        else:

            for i in range(len_exp):
                if i >= 0:
                    try:
                        os.mkdir(CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/')
                    except:
                        pass

                    if args.output == 'scalar':
                        print('==============SCALAR====================')
                        gp_model = BayesianOptimizationMass(args.model_type, x_e, i,
                                                            CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(i) + '/',
                                                            True, args.num_gamma)
                        gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS, args.acquisition_type)
                    elif args.output == 'vector':
                        print('==============VECTOR====================')
                        gp_model = BayesianOptimizationMassVectorOutput(x_e, i,
                                                                        CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(
                                                                            i) + '/',
                                                                        True)
                        gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS)
                        break
                    elif args.output == 'functional':
                        print('==============FUNCTIONAL====================')
                        gp_model = BayesianOptimizationMassFunctionalOutput(x_e, i,
                                                                        CFG.SAVE_PATH + CFG.EXPERIMENT_NAME + str(
                                                                            i) + '/',
                                                                        True)
                        gp_model.optimization_step(CFG.TRAIN_PATH, CFG.NUM_STEPS)



def main():
    args = parse_args()
    optimization(args)



if __name__ == '__main__':
    main()
