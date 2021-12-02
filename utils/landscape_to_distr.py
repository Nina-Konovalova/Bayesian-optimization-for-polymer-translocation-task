import subprocess
from utils.gauss_fit import *
from utils.data_frotran_utils import *


def probabilities_from_init_distributions(x_end):
        best_vals = x_end
        y = gaussian(np.arange(51), *best_vals)
        make_input_file(y)
        subprocess.check_output(["./outputic"])
        # saving new output data
        rate, y_pos_new, y_neg_new = read_data()
        return y_pos_new, y_neg_new, rate

