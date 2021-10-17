import matplotlib.pyplot as plt
import re
from arguments import parse_args
from landscape_to_distr import *
from gauss_fit import *
from data_frotran_utils import *


def main():
    args = parse_args()
    x_e = pd.read_csv(args.path_experiments)
    x_e = np.array(x_e)
    rates = []
    path = 'experiments_1/GP/matern52/not_normalized/MPI/lbfgs/'
    for i in range(len(x_e)):
        x_predicted = read_data_from_file(path + 'report_file_' + str(i) + '.txt')
        x_true = fitting_curves(x_e[i])
        y_pos_new, y_neg_new, rate = probabilities_from_init_distributions(x_true)
        y_pos_new_pred, y_neg_new_pred, rate_pred = probabilities_from_init_distributions(x_predicted)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(abs(rate-rate_pred))
        ax1.plot(y_pos_new, label='true')
        ax1.plot(y_pos_new_pred, label='pred')
        ax2.plot(y_neg_new, label='true')
        ax2.plot(y_neg_new_pred, label='pred')
        ax1.legend()
        ax2.legend()
        rates.append([rate, rate_pred, abs(rate-rate_pred)])
        print(abs(rate-rate_pred))
        plt.savefig(path + 'compare' + str(i) + '.png')
    np.savez_compressed(path + 'rates.npz', rates=rates)


if __name__ == '__main__':
    main()
