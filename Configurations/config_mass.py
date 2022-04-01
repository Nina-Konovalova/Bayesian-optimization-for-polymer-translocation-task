import numpy as np

X = np.linspace(8, 187, 180) # samples from mass distributions

ENERGY_CONST = 0.

SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (5, 15)},  # shape space for gamma distribution
         {'name': 'var_2', 'type': 'continuous', 'domain': (5, 15)},  # scale space for gamma distribution
        ]

# CONSTRAINTS_3 = [{'name': 'constr_1', 'constraint': 'abs(x[:,3])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
#                  {'name': 'constr_2', 'constraint': 'abs(x[:,4])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
#                  2]
