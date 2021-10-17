import subprocess
from gauss_fit import *
from data_frotran_utils import *


def probabilities_from_init_distributions(x_end):
        best_vals = x_end
        y = gaussian(np.arange(51), best_vals[0], best_vals[1], best_vals[2],
                          best_vals[3], best_vals[4], best_vals[5],
                          best_vals[6], best_vals[7], best_vals[8],
                          best_vals[9], best_vals[10], best_vals[11])
        make_input_file(y)
        # plt.plot(y, 'go--', linewidth=4, markersize=2,
        #          color='red', label='parametrized X')

        # fortran programm - making new distributions
        subprocess.check_output(["./outputic"])
        # saving new output data
        rate, y_pos_new, y_neg_new = read_data()
        return y_pos_new, y_neg_new, rate

