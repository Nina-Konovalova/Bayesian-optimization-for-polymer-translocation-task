

MONOMERS = 51

SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 400)},
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

CONSTRAINTS = [{'name': 'constr_1', 'constraint': 'abs(x[:,20])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
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

