import numpy as np
import GPy

X = np.linspace(8, 187, 180) # samples from mass distributions

ENERGY_CONST = 0.

SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (5, 15)},  # shape space for gamma distribution
         {'name': 'var_2', 'type': 'continuous', 'domain': (5, 15)},  # scale space for gamma distribution
        ]

INPUT_DIM = 2

KERNEL_RQ = GPy.kern.RatQuad(INPUT_DIM)
KERNEL_M32 = GPy.kern.Matern32(INPUT_DIM)
KERNEL_M52 = GPy.kern.Matern52(INPUT_DIM)
KERNEL_RBF = GPy.kern.RBF(INPUT_DIM)
KERNEL_EQ = GPy.kern.ExpQuad(INPUT_DIM)

KERNEL = KERNEL_RQ


# CONSTRAINTS_3 = [{'name': 'constr_1', 'constraint': 'abs(x[:,3])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,0]'},
#                  {'name': 'constr_2', 'constraint': 'abs(x[:,4])-4*np.sqrt(2*np.pi*np.exp(1))*x[:,1]'},
#                  2]
