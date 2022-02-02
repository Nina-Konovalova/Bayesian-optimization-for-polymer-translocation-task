import GPy

ALPHA = 0.2

NUM_GAUSS = 3

INPUT_DIM = 3

KERNEL_RQ = GPy.kern.RatQuad(INPUT_DIM)
KERNEL_M32 = GPy.kern.Matern32(INPUT_DIM)
KERNEL_M52 = GPy.kern.Matern52(INPUT_DIM)
KERNEL_RBF = GPy.kern.RBF(INPUT_DIM)
KERNEL_EQ = GPy.kern.ExpQuad(INPUT_DIM)

KERNEL = KERNEL_RQ

TRAIN_PATH = 'dataset_3_gaussians_small_3/exp_gaussians_3_train.npz'
EXP_PATH = 'dataset_3_gaussians_small_3/exp_gaussians_3_exp.npz'
SAVE_PATH = 'experiment_3_gaussians_small_3/'
NUM_STEPS = 2

EXPERIMENT_NAME = 'angs_0.2_Matern32/'

MONOMERS = 51

SPACE_5 = [             {'name': 'var_1', 'type': 'continuous', 'domain': (15, 35)},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, 50)},  # 2
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, 50)},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, 50)},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (0, 50)},
                      {'name': 'var_6', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_7', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_8', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_9', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_10', 'type': 'continuous', 'domain': (-100, 100)},
                      ]

SPACE_4 = [             {'name': 'var_1', 'type': 'continuous', 'domain': (1, 50)},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, 50)},  # 2
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, 50)},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, 50)},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_6', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_7', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_8', 'type': 'continuous', 'domain': (-100, 100)},

                      ]


SPACE_3 = [{'name': 'var_1', 'type': 'continuous', 'domain': (15, 35)},
         {'name': 'var_2', 'type': 'continuous', 'domain': (15, 35)},  # 2
         {'name': 'var_3', 'type': 'continuous', 'domain': (15, 35)},
         {'name': 'var_4', 'type': 'continuous', 'domain': (-10, 10)},
         {'name': 'var_5', 'type': 'continuous', 'domain': (-10, 10)},
         {'name': 'var_6', 'type': 'continuous', 'domain': (-10, 10)},  # 2
         ]

# #
SPACE_20 = [{'name': 'var_1', 'type': 'continuous', 'domain': (20, 400)},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, 400)},  # 2
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_6', 'type': 'continuous', 'domain': (0, 400)},  # 2
                      {'name': 'var_7', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_8', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_9', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_10', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_11', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_12', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_13', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_14', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_15', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_16', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_17', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_18', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_19', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_20', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_21', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_22', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_23', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_24', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_25', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_26', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_27', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_28', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_29', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_30', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_31', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_32', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_33', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_34', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_35', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_36', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_37', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_38', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_39', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_40', 'type': 'continuous', 'domain': (-100, 100)}
                      ]
# # #
SPACE_15 = [             {'name': 'var_1', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, 400)},  # 2
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_6', 'type': 'continuous', 'domain': (0, 400)},  # 2
                      {'name': 'var_7', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_8', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_9', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_10', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_11', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_12', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_13', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_14', 'type': 'continuous', 'domain': (0, 400)},
                      {'name': 'var_15', 'type': 'continuous', 'domain': (0, 400)},

                      {'name': 'var_16', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_17', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_18', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_19', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_20', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_21', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_22', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_23', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_24', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_25', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_26', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_27', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_28', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_29', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_30', 'type': 'continuous', 'domain': (-100, 100)},
                      ]
SPACE_10 = [          {'name': 'var_1', 'type': 'continuous', 'domain': (20, 200)},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, 200)},  # 2
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, 200)},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, 200)},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (0, 200)},
                      {'name': 'var_6', 'type': 'continuous', 'domain': (0, 200)},  # 2
                      {'name': 'var_7', 'type': 'continuous', 'domain': (0, 200)},
                      {'name': 'var_8', 'type': 'continuous', 'domain': (0, 200)},
                      {'name': 'var_9', 'type': 'continuous', 'domain': (0, 200)},
                      {'name': 'var_10', 'type': 'continuous', 'domain': (0, 200)},


                      {'name': 'var_11', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_12', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_13', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_14', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_15', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_16', 'type': 'continuous', 'domain': (-100, 100)},  # 2
                      {'name': 'var_17', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_18', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_19', 'type': 'continuous', 'domain': (-100, 100)},
                      {'name': 'var_20', 'type': 'continuous', 'domain': (-100, 100)},  # 2

                      ]
# ####################################
SPACE = {3: SPACE_3,
         4: SPACE_4,
         5: SPACE_5,
         10: SPACE_10,
         15: SPACE_15,
         20: SPACE_20}
#####################################

CONSTRAINTS_5 = [             {'name': 'constr_1', 'constraint': 'abs(x[:,5])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
                            {'name': 'constr_2', 'constraint': 'abs(x[:,6])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
                            {'name': 'constr_3', 'constraint': 'abs(x[:,7])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,2]'},
                            {'name': 'constr_4', 'constraint': 'abs(x[:,8])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,3]'},
                            {'name': 'constr_5', 'constraint': 'abs(x[:,9])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,4]'},
                            ]

CONSTRAINTS_4 = [             {'name': 'constr_1', 'constraint': 'abs(x[:,4])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
                            {'name': 'constr_2', 'constraint': 'abs(x[:,5])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
                            {'name': 'constr_3', 'constraint': 'abs(x[:,6])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,2]'},
                            {'name': 'constr_4', 'constraint': 'abs(x[:,7])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,3]'},
                            ]
CONSTRAINTS_3 = [             {'name': 'constr_1', 'constraint': 'abs(x[:,3])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
                            {'name': 'constr_2', 'constraint': 'abs(x[:,4])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
                            {'name': 'constr_3', 'constraint': 'abs(x[:,5])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,2]'},
                            ]


CONSTRAINTS_20 = [{'name': 'constr_1', 'constraint': 'abs(x[:,20])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
                            {'name': 'constr_2', 'constraint': 'abs(x[:,21])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
                            {'name': 'constr_3', 'constraint': 'abs(x[:,22])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,2]'},
                            {'name': 'constr_4', 'constraint': 'abs(x[:,23])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,3]'},
                            {'name': 'constr_5', 'constraint': 'abs(x[:,24])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,4]'},
                            {'name': 'constr_6', 'constraint': 'abs(x[:,25])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,5]'},
                            {'name': 'constr_7', 'constraint': 'abs(x[:,26])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,6]'},
                            {'name': 'constr_8', 'constraint': 'abs(x[:,27])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,7]'},
                            {'name': 'constr_9', 'constraint': 'abs(x[:,28])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,8]'},
                            {'name': 'constr_10', 'constraint': 'abs(x[:,29])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,9]'},
                            {'name': 'constr_11', 'constraint': 'abs(x[:,30])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,10]'},
                            {'name': 'constr_12', 'constraint': 'abs(x[:,31])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,11]'},
                            {'name': 'constr_13', 'constraint': 'abs(x[:,32])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,12]'},
                            {'name': 'constr_14', 'constraint': 'abs(x[:,33])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,13]'},
                            {'name': 'constr_15', 'constraint': 'abs(x[:,34])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,14]'},
                            {'name': 'constr_16', 'constraint': 'abs(x[:,35])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,15]'},
                            {'name': 'constr_17', 'constraint': 'abs(x[:,36])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,16]'},
                            {'name': 'constr_18', 'constraint': 'abs(x[:,37])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,17]'},
                            {'name': 'constr_19', 'constraint': 'abs(x[:,38])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,18]'},
                            {'name': 'constr_20', 'constraint': 'abs(x[:,39])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,19]'}
                            ]
# #
CONSTRAINTS_15 = [             {'name': 'constr_1', 'constraint': 'abs(x[:,15])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
                            {'name': 'constr_2', 'constraint': 'abs(x[:,16])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
                            {'name': 'constr_3', 'constraint': 'abs(x[:,17])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,2]'},
                            {'name': 'constr_4', 'constraint': 'abs(x[:,18])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,3]'},
                            {'name': 'constr_5', 'constraint': 'abs(x[:,19])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,4]'},
                            {'name': 'constr_6', 'constraint': 'abs(x[:,20])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,5]'},
                            {'name': 'constr_7', 'constraint': 'abs(x[:,21])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,6]'},
                            {'name': 'constr_8', 'constraint': 'abs(x[:,22])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,7]'},
                            {'name': 'constr_9', 'constraint': 'abs(x[:,23])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,8]'},
                            {'name': 'constr_10', 'constraint': 'abs(x[:,24])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,9]'},
                            {'name': 'constr_11', 'constraint': 'abs(x[:,25])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,10]'},
                            {'name': 'constr_12', 'constraint': 'abs(x[:,26])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,11]'},
                            {'name': 'constr_13', 'constraint': 'abs(x[:,27])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,12]'},
                            {'name': 'constr_14', 'constraint': 'abs(x[:,28])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,13]'},
                            {'name': 'constr_15', 'constraint': 'abs(x[:,29])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,14]'},

                            ]

CONSTRAINTS_10 = [             {'name': 'constr_1', 'constraint': 'abs(x[:,10])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
                            {'name': 'constr_2', 'constraint': 'abs(x[:,11])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
                            {'name': 'constr_3', 'constraint': 'abs(x[:,12])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,2]'},
                            {'name': 'constr_4', 'constraint': 'abs(x[:,13])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,3]'},
                            {'name': 'constr_5', 'constraint': 'abs(x[:,14])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,4]'},
                            {'name': 'constr_6', 'constraint': 'abs(x[:,15])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,5]'},
                            {'name': 'constr_7', 'constraint': 'abs(x[:,16])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,6]'},
                            {'name': 'constr_8', 'constraint': 'abs(x[:,17])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,7]'},
                            {'name': 'constr_9', 'constraint': 'abs(x[:,18])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,8]'},
                            {'name': 'constr_10', 'constraint': 'abs(x[:,19])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,9]'}

                            ]

CONSTRAINTS = {3: CONSTRAINTS_3,
         4: CONSTRAINTS_4,
         5: CONSTRAINTS_5,
         10: CONSTRAINTS_10,
         15: CONSTRAINTS_15,
         20: CONSTRAINTS_20}