import GPy
# TRAIN_PATH = '../dataset_3_gaussians_small_1/exp_gaussians_3_train.npz'
# TEST_PATH = '../dataset_3_gaussians_small_1/exp_gaussians_3_test.npz'
# VAL_PATH = '../dataset_3_gaussians_small_1/exp_gaussians_3_exp.npz'
TRAIN_PATH = '../dataset_3_gaussians_small_0/exp_gaussians_3_train.npz'
TEST_PATH = '../dataset_3_gaussians_small_0/exp_gaussians_3_test.npz'
VAL_PATH = '../dataset_3_gaussians_small_0/exp_gaussians_3_exp.npz'

ALPHA = 0.2 #10 #0 1

INPUT_DIM = 6

GRID = {
    'RatQuad': GPy.kern.RatQuad(INPUT_DIM),
    'Matern52': GPy.kern.Matern52(INPUT_DIM),
    #'Matern32': GPy.kern.Matern32(INPUT_DIM),
    #'RBF': GPy.kern.RBF(INPUT_DIM),
    #'EXPQuad': GPy.kern.ExpQuad(INPUT_DIM),
    # 'sum_gaussians_RatQuad': GPy.kern.RatQuad(2, active_dims=[0, 5]) +
    #                          GPy.kern.RatQuad(2, active_dims=[1, 6]) +
    #                          GPy.kern.RatQuad(2, active_dims=[2, 7]) +
    #                          GPy.kern.RatQuad(2, active_dims=[3, 8]) +
    #                          GPy.kern.RatQuad(2, active_dims=[4, 9])
    }

# GRID = {'sum_Matern52_RatQuad_20': GPy.kern.Matern52(2, active_dims=[0, 10]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[5, 15]) +
#                                   GPy.kern.Matern52(2, active_dims=[1, 11]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[6, 16]) +
#                                   GPy.kern.Matern52(2, active_dims=[2, 12]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[7, 17]) +
#                                   GPy.kern.Matern52(2, active_dims=[3, 13]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[8, 18]) +
#                                   GPy.kern.Matern52(2, active_dims=[4, 14]) + GPy.kern.Matern52(2, active_dims=[9, 19]) +
#                                   GPy.kern.RatQuad(INPUT_DIM)
#         }

# GRID = {#'sum_gaussians_EXPQuad': GPy.kern.ExpQuad(2, active_dims=[0, 10]) + GPy.kern.ExpQuad(2, active_dims=[5, 15]) +
# #                                  GPy.kern.ExpQuad(2, active_dims=[1, 11]) + GPy.kern.ExpQuad(2, active_dims=[6, 16]) +
# #                                  GPy.kern.ExpQuad(2, active_dims=[2, 12]) + GPy.kern.ExpQuad(2, active_dims=[7, 17]) +
# #                                  GPy.kern.ExpQuad(2, active_dims=[3, 13]) + GPy.kern.ExpQuad(2, active_dims=[8, 18]) +
# #                                  GPy.kern.ExpQuad(2, active_dims=[4, 14]) + GPy.kern.ExpQuad(2, active_dims=[9, 19]),
#         'sum_gaussians_RatQuad': GPy.kern.RatQuad(2, active_dims=[0, 10]) + GPy.kern.RatQuad(2, active_dims=[5, 15]) +
#                                  GPy.kern.RatQuad(2, active_dims=[1, 11]) + GPy.kern.RatQuad(2, active_dims=[6, 16]) +
#                                  GPy.kern.RatQuad(2, active_dims=[2, 12]) + GPy.kern.RatQuad(2, active_dims=[7, 17]) +
#                                  GPy.kern.RatQuad(2, active_dims=[3, 13]) + GPy.kern.RatQuad(2, active_dims=[8, 18]) +
#                                  GPy.kern.RatQuad(2, active_dims=[4, 14]) + GPy.kern.RatQuad(2, active_dims=[9, 19]),
#         'sum_gaussians_Matern52': GPy.kern.Matern52(2, active_dims=[0, 10]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[5, 15]) +
#                                   GPy.kern.Matern52(2, active_dims=[1, 11]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[6, 16]) +
#                                   GPy.kern.Matern52(2, active_dims=[2, 12]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[7, 17]) +
#                                   GPy.kern.Matern52(2, active_dims=[3, 13]) + GPy.kern.Matern52(2,
#                                                                                                 active_dims=[8, 18]) +
#                                   GPy.kern.Matern52(2, active_dims=[4, 14]) + GPy.kern.Matern52(2, active_dims=[9, 19])
#         }

# GRID = {#'sum_gaussians_RBF': GPy.kern.RBF(2, active_dims=[0, 10]) + GPy.kern.RBF(2, active_dims=[5, 15]) +
# #                                   GPy.kern.RBF(2, active_dims=[1, 11]) + GPy.kern.RBF(2, active_dims=[6, 16]) +
# #                                   GPy.kern.RBF(2, active_dims=[2, 12]) + GPy.kern.RBF(2, active_dims=[7, 17]) +
# #                                   GPy.kern.RBF(2, active_dims=[3, 13]) + GPy.kern.RBF(2, active_dims=[8, 18]) +
# #                                   GPy.kern.RBF(2, active_dims=[4, 14]) + GPy.kern.RBF(2, active_dims=[9, 19]),
#         'sum_gaussians_EXPQuad': GPy.kern.ExpQuad(2, active_dims=[0, 10]) + GPy.kern.ExpQuad(2, active_dims=[5, 15]) +
#                                   GPy.kern.ExpQuad(2, active_dims=[1, 11]) + GPy.kern.ExpQuad(2, active_dims=[6, 16]) +
#                                   GPy.kern.ExpQuad(2, active_dims=[2, 12]) + GPy.kern.ExpQuad(2, active_dims=[7, 17]) +
#                                   GPy.kern.ExpQuad(2, active_dims=[3, 13]) + GPy.kern.ExpQuad(2, active_dims=[8, 18]) +
#                                   GPy.kern.ExpQuad(2, active_dims=[4, 14]) + GPy.kern.ExpQuad(2, active_dims=[9, 19]),
#         'sum_gaussians_RatQuad': GPy.kern.RatQuad(2, active_dims=[0, 10]) + GPy.kern.RatQuad(2, active_dims=[5, 15]) +
#                                   GPy.kern.RatQuad(2, active_dims=[1, 11]) + GPy.kern.RatQuad(2, active_dims=[6, 16]) +
#                                   GPy.kern.RatQuad(2, active_dims=[2, 12]) + GPy.kern.RatQuad(2, active_dims=[7, 17]) +
#                                   GPy.kern.RatQuad(2, active_dims=[3, 13]) + GPy.kern.RatQuad(2, active_dims=[8, 18]) +
#                                   GPy.kern.RatQuad(2, active_dims=[4, 14]) + GPy.kern.RatQuad(2, active_dims=[9, 19]),
#         # 'sum_gaussians_Periodic': GPy.kern.StdPeriodic(2, active_dims=[0, 10]) + GPy.kern.StdPeriodic(2, active_dims=[5, 15]) +
#                           GPy.kern.StdPeriodic(2, active_dims=[1, 11]) + GPy.kern.StdPeriodic(2, active_dims=[6, 16]) +
#                           GPy.kern.StdPeriodic(2, active_dims=[2, 12]) + GPy.kern.StdPeriodic(2, active_dims=[7, 17]) +
#                           GPy.kern.StdPeriodic(2, active_dims=[3, 13]) + GPy.kern.StdPeriodic(2, active_dims=[8, 18]) +
#                           GPy.kern.StdPeriodic(2, active_dims=[4, 14]) + GPy.kern.StdPeriodic(2, active_dims=[9, 19]),
# 'sum_gaussians_Matern52': GPy.kern.Matern52(2, active_dims=[0, 10]) + GPy.kern.Matern52(2, active_dims=[5, 15]) +
#                           GPy.kern.Matern52(2, active_dims=[1, 11]) + GPy.kern.Matern52(2, active_dims=[6, 16]) +
#                           GPy.kern.Matern52(2, active_dims=[2, 12]) + GPy.kern.Matern52(2, active_dims=[7, 17]) +
#                           GPy.kern.Matern52(2, active_dims=[3, 13]) + GPy.kern.Matern52(2, active_dims=[8, 18]) +
#                           GPy.kern.Matern52(2, active_dims=[4, 14]) + GPy.kern.Matern52(2, active_dims=[9, 19]),

# 'sum_gaussians_Matern32': GPy.kern.Matern32(2, active_dims=[0, 10]) + GPy.kern.Matern32(2, active_dims=[5, 15]) +
#                           GPy.kern.Matern32(2, active_dims=[1, 11]) + GPy.kern.Matern32(2, active_dims=[6, 16]) +
#                           GPy.kern.Matern32(2, active_dims=[2, 12]) + GPy.kern.Matern32(2, active_dims=[7, 17]) +
#                           GPy.kern.Matern32(2, active_dims=[3, 13]) + GPy.kern.Matern32(2, active_dims=[8, 18]) +
#                           GPy.kern.Matern32(2, active_dims=[4, 14]) + GPy.kern.Matern32(2, active_dims=[9, 19]),
# }
# 'sum_gaussians_Matern52': GPy.kern.Matern52(2, active_dims=[0, 10]) + GPy.kern.Matern52(2, active_dims=[5, 15]) +
#                           GPy.kern.Matern52(2, active_dims=[1, 11]) + GPy.kern.Matern52(2, active_dims=[6, 16]) +
#                           GPy.kern.Matern52(2, active_dims=[2, 12]) + GPy.kern.Matern52(2, active_dims=[7, 17]) +
#                           GPy.kern.Matern52(2, active_dims=[3, 13]) + GPy.kern.Matern52(2, active_dims=[8, 18]) +
#                           GPy.kern.Matern52(2, active_dims=[4, 14]) + GPy.kern.Matern52(2, active_dims=[9, 19]),
# 'mult_gaussians_Matern52': GPy.kern.Matern52(2, active_dims=[0, 10]) * GPy.kern.Matern52(2, active_dims=[5, 15]) *
#                            GPy.kern.Matern52(2, active_dims=[1, 11]) * GPy.kern.Matern52(2, active_dims=[6, 16]) *
#                            GPy.kern.Matern52(2, active_dims=[2, 12]) * GPy.kern.Matern52(2, active_dims=[7, 17]) *
#                            GPy.kern.Matern52(2, active_dims=[3, 13]) * GPy.kern.Matern52(2, active_dims=[8, 18]) *
#                            GPy.kern.Matern52(2, active_dims=[4, 14]) * GPy.kern.Matern52(2, active_dims=[9, 19]),
# 'sum_amp_vae_Matern52': GPy.kern.Matern32(10, active_dims=np.arange(10)) + GPy.kern.Matern32(10, active_dims=np.arange(10)+10),
# 'mult_amp_vae_Matern52': GPy.kern.Matern32(10, active_dims=np.arange(10)) * GPy.kern.Matern32(10, active_dims=np.arange(10)+10),
# 'sum_gaussians_Matern32': GPy.kern.Matern32(2, active_dims=[0, 10]) + GPy.kern.Matern32(2, active_dims=[5, 15]) +
#                           GPy.kern.Matern32(2, active_dims=[1, 11]) + GPy.kern.Matern32(2, active_dims=[6, 16]) +
#                           GPy.kern.Matern32(2, active_dims=[2, 12]) + GPy.kern.Matern32(2, active_dims=[7, 17]) +
#                           GPy.kern.Matern32(2, active_dims=[3, 13]) + GPy.kern.Matern32(2, active_dims=[8, 18]) +
#                           GPy.kern.Matern32(2, active_dims=[4, 14]) + GPy.kern.Matern32(2, active_dims=[9, 19]),
# 'mult_gaussians_Matern32': GPy.kern.Matern32(2, active_dims=[0, 10]) * GPy.kern.Matern32(2, active_dims=[5, 15]) *
#                            GPy.kern.Matern32(2, active_dims=[1, 11]) * GPy.kern.Matern32(2, active_dims=[6, 16]) *
#                            GPy.kern.Matern32(2, active_dims=[2, 12]) * GPy.kern.Matern32(2, active_dims=[7, 17]) *
#                            GPy.kern.Matern32(2, active_dims=[3, 13]) * GPy.kern.Matern32(2, active_dims=[8, 18]) *
#                            GPy.kern.Matern32(2, active_dims=[4, 14]) * GPy.kern.Matern32(2, active_dims=[9, 19]),
# 'sum_amp_vae_Matern32': GPy.kern.Matern32(10, active_dims=np.arange(10)) + GPy.kern.Matern32(10, active_dims=np.arange(10)+10),
# 'mult_amp_vae_Matern32': GPy.kern.Matern32(10, active_dims=np.arange(10)) * GPy.kern.Matern32(10, active_dims=np.arange(10)+10),
# }
