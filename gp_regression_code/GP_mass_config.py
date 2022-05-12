import GPy

TRAIN_PATH = '../make_dataset/dataset_mass_2_0/train/sample_data/samples_info.npz'
TEST_PATH = '../make_dataset/dataset_mass_2_0/test/sample_data/samples_info.npz'
VAL_PATH = '../make_dataset/dataset_mass_2_0/exp/sample_data/samples_info.npz'


PATH_TO_SAVE_PLOTS = 'gp_regression_mass_2_0/plots/'

INPUT_DIM = 4

GRID = {
    'RatQuad': GPy.kern.RatQuad(INPUT_DIM),
    'Matern52': GPy.kern.Matern52(INPUT_DIM),
    'Matern32': GPy.kern.Matern32(INPUT_DIM),
    'RBF': GPy.kern.RBF(INPUT_DIM),
    'EXPQuad': GPy.kern.ExpQuad(INPUT_DIM),
    }

