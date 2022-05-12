import numpy as np
import GPy
from emukit.core import ParameterSpace, ContinuousParameter

#X = np.linspace(5, 299, 295) # samples from mass distributions
X = np.linspace(5, 421, 417)


ENERGY_CONST = 0.

DATA_SIZE = {
    'train': 100,
    'test': 100,
    'exp': 50
}

# SPACE = ParameterSpace([ContinuousParameter('x1', 2, 30),
#                         ContinuousParameter('x2', 2, 30)])

# SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (0.01, 100)},  # shape space for gamma distribution
#          {'name': 'var_2', 'type': 'continuous', 'domain': (0.1, 100)},  # scale space for gamma distribution
#          ]

# SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (1, 50)},  # shape space for gamma distribution
#          {'name': 'var_2', 'type': 'continuous', 'domain': (1, 50)},  # scale space for gamma distribution
#          ]
#
# SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (2, 30)},  # shape space for gamma distribution
#          {'name': 'var_2', 'type': 'continuous', 'domain': (2, 30)},  # scale space for gamma distribution
#          ]

# SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (5, 15)},  # shape space for gamma distribution
#          {'name': 'var_2', 'type': 'continuous', 'domain': (5, 15)},  # scale space for gamma distribution
#          ]

# SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (3, 13)},  # shape space for gamma distribution
#          {'name': 'var_2', 'type': 'continuous', 'domain': (3, 13)},  # scale space for gamma distribution
#          {'name': 'var_3', 'type': 'continuous', 'domain': (14, 18)},  # shape space for gamma distribution
#          {'name': 'var_4', 'type': 'continuous', 'domain': (14, 18)},  # scale space for gamma distribution
#          ]

SPACE = [{'name': 'var_1', 'type': 'continuous', 'domain': (3, 7)},  # shape space for gamma distribution
         {'name': 'var_2', 'type': 'continuous', 'domain': (3, 7)},  # scale space for gamma distribution
         {'name': 'var_3', 'type': 'continuous', 'domain': (7, 12)},  # shape space for gamma distribution
         {'name': 'var_4', 'type': 'continuous', 'domain': (7, 12)},  # scale space for gamma distribution
         {'name': 'var_5', 'type': 'continuous', 'domain': (13, 18)},  # shape space for gamma distribution
         {'name': 'var_6', 'type': 'continuous', 'domain': (13, 18)},  # scale space for gamma distribution
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
