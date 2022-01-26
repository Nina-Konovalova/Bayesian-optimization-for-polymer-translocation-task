import matplotlib.pyplot as plt
from Configurations.arguments import parse_args
from utils.landscape_to_distr import *
from utils.gauss_fit import *
from utils.data_frotran_utils import *
from sklearn.metrics import mean_squared_error as mse


def read_data_from_file(path):
    with open(path) as f:
        data = f.read().split('\n')[31:-2]
    dat_str = ''
    for elem in data:
        dat_str = dat_str + (str(elem))
    import re
    dat_str = re.split(' ', dat_str)
    data = []
    for elem in dat_str:
        if elem not in ['', 'Best', 'found', 'minimum', 'location:']:
            data.append(elem)
    data = np.array(data).astype('float32')

    return data


def make_report_compare(path, x_true, x_pred, MSE_pos, MSE_neg, rate_true, rate_predicted):
    f = open(path, "w+")

    f.write("x_true:\n")
    f.write(str(x_true) + "\n")
    f.write('------------------------\n')

    f.write("x_pred:\n")
    f.write(str(x_pred) + "\n")
    f.write('------------------------\n')

    f.write("MSE_pos:\n")
    f.write(str(MSE_pos) + "\n")
    f.write('------------------------\n')

    f.write("MSE_neg:\n")
    f.write(str(MSE_neg) + "\n")
    f.write('------------------------\n')

    f.write("Difference between rates:\n")
    f.write(str(rate_true) + ' ' + str(rate_predicted) + ' ' + str(abs(rate_true-rate_predicted)) + "\n")
    f.write('------------------------\n')

    f.close()




def main():
    args = parse_args()
    #x_e = pd.read_csv(args.path_experiments)
    #x_e = np.array(x_e)
    x_e = np.load('new_test_10.npz')['vecs']
    path = args.path_for_save #'experiments_1/GP/rbf/not_normalized/MPI/lbfgs/gauss_20_200/'
    for i in range(22, 49):#len(x_e)):
        x_predicted = read_data_from_file(path + 'report_file_' + str(i) + '.txt')
        #x_true = fitting_curves(x_e[i])
        x_true = x_e[i]
        print(len(x_true))

        y_pos_new, y_neg_new, rate = probabilities_from_init_distributions(x_true)
        print(len(x_predicted))
        y_pos_new_pred, y_neg_new_pred, rate_pred = probabilities_from_init_distributions(x_predicted)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(abs(rate - rate_pred))
        ax1.plot(y_pos_new, label='true')
        ax1.plot(y_pos_new_pred, label='pred')
        ax1.set_yscale('log')
        ax2.plot(y_neg_new, label='true')
        ax2.plot(y_neg_new_pred, label='pred')
        ax2.set_yscale('log')
        ax1.legend()
        ax2.legend()
        MSE_pos = mse((y_pos_new[:]), (y_pos_new_pred[:]))
        MSE_neg = mse((y_neg_new[:]), (y_neg_new_pred[:]))
        plt.savefig(path + 'compare' + str(i) + '.png')
        make_report_compare(path + 'compare_' + str(i) + '.txt', x_true, x_predicted,
                            MSE_pos, MSE_neg, rate, rate_pred)


if __name__ == '__main__':
    main()
