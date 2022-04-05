import GPy

TRAIN_PATH = '../dataset_mass_0/train/samples_info.npz'
TEST_PATH = '../dataset_mass_0/test/samples_info.npz'
VAL_PATH = '../dataset_mass_0/exp/samples_info.npz'

INPUT_DIM = 2

GRID = {
    'RatQuad': GPy.kern.RatQuad(INPUT_DIM),
    'Matern52': GPy.kern.Matern52(INPUT_DIM),
    'Matern32': GPy.kern.Matern32(INPUT_DIM),
    'RBF': GPy.kern.RBF(INPUT_DIM),
    'EXPQuad': GPy.kern.ExpQuad(INPUT_DIM),
    }

