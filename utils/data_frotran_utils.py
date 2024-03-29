import numpy as np
import pandas as pd


def make_input_file(curv, N=51, t=1, num1=50000):
    '''
    :param curv: free energy landscape
    :param N: number of monomers
    :param t: init number of monomers in trans compartment
    :param num1:number of points for time distr
    :return:txt file - input file to fortran program
    '''
    f = open('./new_input_1.txt', 'w')
    f.write(
        str(N - 1) + '\t' + str(t) + '\n' + str(num1) + '\t' + str(t) + '\t' + str(10000) + '\t' + str(t) + '\n')
    for i in range(N):
        f.write(str(i) + '\t' + str(curv[i]) + '\n')
    f.close()


def read_derives(path_to_file='./der_output_1.txt'):
        '''
        :param path_to_file: file with derivatives for free energy landscape
        :return: array of derivatives
        '''
        derives_dat = pd.read_csv(path_to_file, sep=' ', header=None)
        derives_dat[2][derives_dat[2] == 0] = -1
        for i in range(derives_dat.shape[0]):
            if str(derives_dat[1][i]).find('E') == -1 \
                    and str(derives_dat[1][i]).find('e') == -1 \
                    and str(derives_dat[1][i]).find('-') != -1 \
                    and str(derives_dat[1][i]).find('nan') == -1:
                derives_dat[1][i] = (str(derives_dat[1][i]).split('-')[0]) + 'E' + \
                                    '-' + (str(derives_dat[1][i]).split('-')[1])
        derives = np.array(np.array(derives_dat[1]).astype('float32') * derives_dat[2]).astype('float32')
        return derives


def read_data(path_to_file='./new_output_1.txt', N=50):
    '''
    :param path_to_file: path to output from fortran program
    :return: rate, time, y_pos_new, y_neg_new, got from FP equation solution
    '''
    dat = pd.read_csv(path_to_file, sep=' ', skiprows=[0, 1, 2], header=None)
    dat.drop(dat.columns[0], axis=1, inplace=True)
    #dat.fillna(1e+20, inplace=True)
    r = pd.read_csv('./new_output_1.txt', sep=' ', nrows=2, header=None)
    #r.fillna(1e+20, inplace=True)

    if N < 11:
        rate = np.array(r)[0][12]
        rate = float(rate)
        time = [np.array(r)[0][14], float(np.array(r)[1][11])]
        y_pos_new = np.array(dat[1][:], dtype=float)
        y_neg_new = np.array(dat[2][:], dtype=float)
    elif N < 101:
        rate = np.array(r)[0][11]
        rate = float(rate)
        time = [np.array(r)[0][13], float(np.array(r)[1][10])]
        y_pos_new = np.array(dat[1][:], dtype=float)
        y_neg_new = np.array(dat[2][:], dtype=float)
    elif N < 1001:
        rate = np.array(r)[0][10]
        rate = float(rate)
        time = [np.array(r)[0][12], float(np.array(r)[1][10])]
        y_pos_new = np.array(dat[1][:], dtype=float)
        y_neg_new = np.array(dat[2][:], dtype=float)

    del dat
    del r

    return rate, time, y_pos_new, y_neg_new