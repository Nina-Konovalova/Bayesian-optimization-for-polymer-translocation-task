import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('../')
from utils.landscape_to_distr import probabilities_from_init_distributions
from utils.data_frotran_utils import read_derives
from numpy import exp

x = np.arange(51)




class MakeDataset:
    def __init__(self, num_of_all_g, dir_name):
        self.num_of_all_g = num_of_all_g

        if self.num_of_all_g == 3:
            self.nums = ['one', 'two', 'three']
        elif self.num_of_all_g == 4:
            self.nums = ['one', 'two', 'three', 'four']
        elif self.num_of_all_g == 5:
            self.nums = ['one', 'two', 'three', 'four', 'five']
        try:
            os.mkdir(dir_name)
        except:
            pass

        self.dir_name = dir_name

    @staticmethod
    def angle(y_pos_new, y_neg_new):
        xx = np.arange(len(y_pos_new))
        t_pos = np.polyfit(xx, np.log(y_pos_new), 1)
        t_neg = np.polyfit(xx, np.log(y_neg_new), 1)
        return [t_pos[0], t_neg[0]]

    def gaussian(self, x, *params):
        '''
        :param x: array from 0 to number of monomers
        :param params: vector of parameters for free energy landscape
        :return: array of gaussian function with params applied to x
        '''
        eps = 1e-18
        cen = np.linspace(10, 50, self.num_of_all_g)
        wid = params[:len(cen)]
        amp = params[len(cen):]
        gauss = 0
        for i in range(len(cen)):
            gauss += amp[i] * 1 / (np.sqrt((wid[i] + eps) * 2 * np.pi)) * exp(-(x - cen[i]) ** 2 / (2 * (wid[i] + eps)))
        return gauss

    def make_data(self, vecs, angs, rates, y_pos, y_neg, times, name):
        d = {}
        d['vecs'] = vecs.tolist()
        d['angs'] = angs.tolist()
        d['rates'] = rates.tolist()
        d['y_pos'] = y_pos.tolist()
        d['y_neg'] = y_neg.tolist()
        d['times'] = times.tolist()

        with open(name + '.json', 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=4)

        np.savez_compressed(name + '.npz', vecs=vecs, angs=angs, rates=rates, y_pos=y_pos, y_neg=y_neg, times=times)

    def check_data(self, vec):
        problem = False
        y_pos_new, y_neg_new, rate, time = probabilities_from_init_distributions(vec)
        old_der = read_derives()
        if rate > 1 or rate == None or rate == 1e20:
            problem = True
            return rate, [0, 0], y_pos_new, y_neg_new, time, problem
        elif (abs(old_der) > 4.).sum() != 0:
            problem = True
            return rate, [0, 0], y_pos_new, y_neg_new, time, problem
        else:
            a = self.angle(y_pos_new, y_neg_new)
            if np.isnan(a).sum() != 0 or a == 0:
                problem = True
                return 0, 0, 0, 0, 0, problem
            else:
                return rate, a, y_pos_new, y_neg_new, time, problem

    def make_all_dataset(self, mode):
        data = {}
        for n in self.nums:
            data[n] = self.make_dict(self.dir_name + n + '_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')

        for key in data.keys():

            try:
                vecs_t = np.concatenate([vecs_t, data[key]['vecs']])
                angs_t = np.concatenate([angs_t, data[key]['angs']])
                rates_t = np.concatenate([rates_t, data[key]['rates']])
                y_pos_t = np.concatenate([y_pos_t, data[key]['y_pos']])
                y_neg_t = np.concatenate([y_neg_t, data[key]['y_neg']])
                times_t = np.concatenate([times_t, data[key]['times']])
            except:
                vecs_t = data[key]['vecs']
                angs_t = data[key]['angs']
                rates_t = data[key]['rates']
                y_pos_t = data[key]['y_pos']
                y_neg_t = data[key]['y_neg']
                times_t = data[key]['times']

        self.make_data(vecs_t, angs_t, rates_t, y_pos_t, y_neg_t, times_t,
                        self.dir_name + 'exp_gaussians_'+ str(self.num_of_all_g) + '_' + mode)


    def check_data_shape(self, path):
        data = np.load(path)
        print(data['vecs'].shape)
        print(data['angs'].shape)
        print(data['y_neg'].shape)
        print(data['y_pos'].shape)
        print(data['rates'].shape)
        print(data['times'].shape)

    def make_dict(self, path):
        d = {}
        dat = np.load(path)
        d['vecs'] = dat['vecs']
        d['angs'] = dat['angs']
        d['rates'] = dat['rates']
        d['y_pos'] = dat['y_pos']
        d['y_neg'] = dat['y_neg']
        d['times'] = dat['times']
        return d

    def draw_all_dataset(self, mode):
        data = {}
        try:
            os.mkdir(self.dir_name + mode)
        except:
            pass

        for n in self.nums:
            data[n] = self.make_dict(self.dir_name + n + '_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')
            try:
                os.mkdir(self.dir_name + mode + '/' + n + '_gauss')
            except:
                pass

            d = data[n]
            for i in tqdm(range(len(d['vecs']))):
                fig, axs = plt.subplots(1, 3)
                axs[0].plot(self.gaussian(x, *d['vecs'][i]))
                axs[0].set_title('init landscape')
                axs[1].plot(np.log(d['y_pos'][i]), 'tab:red')
                axs[1].set_title('y_pose_log')
                axs[2].plot(np.log(d['y_neg'][i]), 'tab:green')
                axs[2].set_title('y_neg_log')
                fig.suptitle(d['rates'][i])
                plt.savefig(self.dir_name + mode + '/' + n + '_gauss/' + 'pic_' + str(i) + '.jpg')


    def one_gaussian(self, std_amp, std_bias, ampl_amp, ampl_bias, mode, num_of_samples):
        num = 0
        vecs = []
        rates = []
        y_pos = []
        y_neg = []
        angs = []
        times = []
        for elem in [-1, 1]:
            for position in range(self.num_of_all_g):
                k = 0
                while k < num_of_samples:
                    vec = np.zeros(self.num_of_all_g * 2)
                    vec[:self.num_of_all_g] = 1
                    vec[self.num_of_all_g + position] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                    vec[position] = std_bias + np.random.rand() * std_amp
                    rate, a, y_pos_new, y_neg_new, time, problem = self.check_data(vec)
                    if not problem:
                        num += 1
                        k += 1
                        vecs.append(vec)
                        rates.append(rate)
                        angs.append(a)
                        y_pos.append(y_pos_new)
                        y_neg.append(y_neg_new)
                        times.append(time)
        print('save 1 gaussians in file', self.dir_name + 'one_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')
        np.savez_compressed(self.dir_name + 'one_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz',
                            vecs=np.array(vecs), angs=np.array(angs),
                            y_neg=np.array(y_neg), y_pos=np.array(y_pos),
                            rates=np.array(rates), times=np.array(times))


    def two_gaussians(self, std_amp, std_bias, ampl_amp, ampl_bias, mode, num_of_samples):
        vecs = []
        rates = []
        y_pos = []
        y_neg = []
        times = []
        angs = []
        num = 0
        for elem in [-1, 1]:
            for i in range(self.num_of_all_g):
                for j in range(i + 1, self.num_of_all_g):
                    k = 0
                    while k < num_of_samples:
                        vec = np.zeros(self.num_of_all_g * 2)
                        vec[:self.num_of_all_g] = 1
                        vec[self.num_of_all_g + i] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                        vec[i] = np.random.rand() * std_amp + std_bias
                        vec[self.num_of_all_g + j] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                        vec[j] = np.random.rand() * std_amp + std_bias
                        rate, a, y_pos_new, y_neg_new, time, problem = self.check_data(vec)
                        if not problem:
                            num += 1
                            k += 1
                            vecs.append(vec)
                            rates.append(rate)
                            angs.append(a)
                            y_pos.append(y_pos_new)
                            y_neg.append(y_neg_new)
                            times.append(time)
        print('save 2 gaussians in file', self.dir_name + 'two_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')
        np.savez_compressed(self.dir_name + 'two_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz',
                            vecs=np.array(vecs), angs=np.array(angs),
                            y_neg=np.array(y_neg), y_pos=np.array(y_pos),
                            rates=np.array(rates), times=np.array(times))

    def three_gaussians(self, std_amp, std_bias, ampl_amp, ampl_bias, mode, num_of_samples):
        vecs = []
        rates = []
        y_pos = []
        y_neg = []
        angs = []
        times = []
        num = 0
        for elem in [-1, 1]:
            for i in range(self.num_of_all_g):
                for j in range(i + 1, self.num_of_all_g):
                    for m in range(j + 1, self.num_of_all_g):
                        k = 0
                        while k < num_of_samples:
                            vec = np.zeros(self.num_of_all_g * 2)
                            vec[:self.num_of_all_g] = 1
                            vec[self.num_of_all_g + i] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                            vec[i] = np.random.rand() * std_amp + std_bias
                            vec[self.num_of_all_g + j] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                            vec[j] = np.random.rand() * std_amp + std_bias
                            vec[self.num_of_all_g + m] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                            vec[m] = np.random.rand() * std_amp + ampl_amp
                            rate, a, y_pos_new, y_neg_new, time, problem = self.check_data(vec)
                            if not problem:
                                num += 1
                                k += 1
                                vecs.append(vec)
                                rates.append(rate)
                                angs.append(a)
                                y_pos.append(y_pos_new)
                                y_neg.append(y_neg_new)
                                times.append(time)
        print('save 3 gaussians in file', self.dir_name + 'three_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')
        np.savez_compressed(self.dir_name + 'three_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz',
                            vecs=np.array(vecs), angs=np.array(angs),
                            y_neg=np.array(y_neg), y_pos=np.array(y_pos),
                            rates=np.array(rates), times=np.array(times))


    def four_gaussians(self, std_amp, std_bias, ampl_amp, ampl_bias, mode, num_of_samples):
        vecs = []
        rates = []
        y_pos = []
        y_neg = []
        angs = []
        times = []

        num = 0
        for elem in [-1, 1]:
            for i in range(self.num_of_all_g):
                for j in range(i + 1, self.num_of_all_g):
                    for m in range(j + 1, self.num_of_all_g):
                        for l in range(m + 1, self.num_of_all_g):
                            k = 0
                            while k < num_of_samples:
                                vec = np.zeros(self.num_of_all_g * 2)
                                vec[:self.num_of_all_g] = 1
                                vec[self.num_of_all_g + i] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                vec[i] = np.random.rand() * std_amp + std_bias
                                vec[self.num_of_all_g + j] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                vec[j] = np.random.rand() * std_amp + std_bias
                                vec[self.num_of_all_g + m] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                vec[m] = np.random.rand() * std_amp + std_bias
                                vec[self.num_of_all_g + l] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                vec[l] = np.random.rand() * std_amp + std_bias
                                rate, a, y_pos_new, y_neg_new, time, problem = self.check_data(vec)
                                if not problem:
                                    num += 1
                                    k += 1
                                    vecs.append(vec)
                                    rates.append(rate)
                                    angs.append(a)
                                    y_pos.append(y_pos_new)
                                    y_neg.append(y_neg_new)
                                    times.append(time)
        print('save 4 gaussians in file', self.dir_name + 'four_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')
        np.savez_compressed(self.dir_name + 'four_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz',
                            vecs=np.array(vecs), angs=np.array(angs),
                            y_neg=np.array(y_neg), y_pos=np.array(y_pos),
                            rates=np.array(rates), times=np.array(times))

    def five_gaussians(self, std_amp, std_bias, ampl_amp, ampl_bias, mode, num_of_samples):
        vecs = []
        rates = []
        y_pos = []
        y_neg = []
        angs = []
        times = []

        num = 0
        for elem in [-1, 1]:
            for i in range(self.num_of_all_g):
                for j in range(i + 1, self.num_of_all_g):
                    for m in range(j + 1, self.num_of_all_g):
                        for l in range(m + 1, self.num_of_all_g):
                            for z in range(l + 1, self.num_of_all_g):
                                k = 0
                                while k < num_of_samples:
                                    vec = np.zeros(self.num_of_all_g * 2)
                                    vec[:self.num_of_all_g] = 1
                                    vec[self.num_of_all_g + i] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                    vec[i] = np.random.rand() * std_amp + std_bias
                                    vec[self.num_of_all_g + j] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                    vec[j] = np.random.rand() * std_amp + std_bias
                                    vec[self.num_of_all_g + m] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                    vec[m] = np.random.rand() * std_amp + std_bias
                                    vec[self.num_of_all_g + l] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                    vec[l] = np.random.rand() * std_amp + std_bias
                                    vec[self.num_of_all_g + z] = (ampl_bias + np.random.rand() * ampl_amp) * elem
                                    vec[z] = np.random.rand() * std_amp + std_bias
                                    rate, a, y_pos_new, y_neg_new, time, problem = self.check_data(vec)
                                    if not problem:
                                        num += 1
                                        k += 1
                                        vecs.append(vec)
                                        rates.append(rate)
                                        angs.append(a)
                                        y_pos.append(y_pos_new)
                                        y_neg.append(y_neg_new)
                                        times.append(time)
        print('save 5 gaussians in file', self.dir_name + 'five_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz')
        np.savez_compressed(self.dir_name + 'five_gauss_' + str(self.num_of_all_g) + '_' + mode + '.npz',
                            vecs=np.array(vecs), angs=np.array(angs),
                            y_neg=np.array(y_neg), y_pos=np.array(y_pos),
                            rates=np.array(rates), times=np.array(times))









