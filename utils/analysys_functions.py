import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_different_kernels(path):
    '''
    Analyze loss function for different kernels.
    Also save plots for convergence of bayesian optimization for different kernels for each experiment
    :param path: path to folder with folders with kernels
    :return: 'i.jpg' - plots of convergence for different kernels,
             'result.json' - resulting loss for each experiment, mean loss for each kernel,
             'result.csv' - mean loss for each kernel
    '''
    dirs = os.listdir(path)
    try:
        os.mkdir('analyze_different_kernels')
    except:
        pass
    colors = ['red', 'green', 'DarkBlue', 'm', 'y', 'seagreen']
    final_info = {}
    for i in tqdm(range(50)):
        final_info[i] = {}
        loss_all = {}
        plt.figure(figsize=(16, 12))
        for n, d in enumerate(dirs):
            loss_all[d] = 0
            p = path + d + '/' + str(i) + '/'
            with open(p + 'predicted_data.json') as json_file:
                data = json.load(json_file)
                data_loss = data['all_way']['loss']
                loss = data['prediction_from_opt']['loss']
                true_val = data['real']
                pred_val = data['prediction_from_opt']
            loss_all[d] += loss
            final_info[i][d] = {'loss': loss,
                                'true_val': true_val,
                                'pred_val': pred_val}

            min_loss = []
            for k in range(len(data_loss)):
                min_val = np.min(data_loss[:k + 1])
                min_loss.append(min_val)

            plt.plot(min_loss, c=colors[n], label=d, linewidth=2.5, alpha=0.5)
            plt.scatter(np.arange(len(min_loss)), min_loss, c=colors[n], s=40, marker='+')
        plt.legend(fontsize=22)
        plt.ylabel('f', fontsize=22)
        plt.xlabel('bayesian steps', fontsize=22)
        plt.savefig('analyze_different_kernels/' + str(i) + '.jpg')
        plt.close()

    for k in loss_all.keys():
        loss_all[k] /= 50
    results = {'all': final_info, 'mean': loss_all}
    with open('analyze_different_kernels/result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    d = pd.DataFrame({'Kernels': loss_all.keys(), 'MSE': loss_all})
    d.to_csv(f + '/result.csv', index=False)


def analyze_different_space(spaces):
    '''
        Analyze loss function for different kernels.
        Also save plots for convergence of bayesian optimization for different kernels for each experiment
        :param spaces: dict - path to folder: info about space
        ----------------------------------------------------
        example:
        spaces = {'experiment_mass_new_space': '[0.01, 100]',
              'experiment_mass_new_space_smaller': '[1, 50]',
              'experiment_mass_usual_space_smaller': '[2, 30]'}
        ----------------------------------------------------
        :return: 'i.jpg' - plots of convergence for different kernels,
                 'result.json' - resulting loss for each experiment, mean loss for each kernel,
                 'result.csv' - mean loss for each kernel
    '''

    try:
        os.mkdir('analyze_different_spaces')
    except:
        pass
    colors = ['red', 'green', 'DarkBlue', 'm']
    final_info = {}
    for i in tqdm(range(50)):
        final_info[i] = {}
        loss_all = {}
        plt.figure(figsize=(16, 12))
        for n, d in enumerate(list(spaces.keys())):

            loss_all[d] = 0
            p = spaces[d] + '/RatQuad/' + str(i) + '/'

            with open(p + 'predicted_data.json') as json_file:
                data = json.load(json_file)
                data_loss = data['all_way']['loss']
                loss = data['prediction_from_opt']['loss']
                true_val = data['real']
                pred_val = data['prediction_from_opt']
            loss_all[d] += loss
            final_info[i][d] = {'loss': loss,
                                'true_val': true_val,
                                'pred_val': pred_val}

            min_loss = []
            for k in range(len(data_loss)):
                min_val = np.min(data_loss[:k + 1])
                min_loss.append(min_val)

            plt.plot(min_loss, c=colors[n], label=spaces[d], linewidth=2.5, alpha=0.5)
            plt.scatter(np.arange(len(min_loss)), min_loss, c=colors[n], s=40, marker='+')
        plt.legend(fontsize=22)
        plt.ylabel('f', fontsize=22)
        plt.xlabel('bayesian steps', fontsize=22)
        plt.savefig('analyze_different_spaces/' + str(i) + '.jpg')
        plt.close()

    for k in loss_all.keys():
        loss_all[k] /= 50
    results = {'all': final_info, 'mean': loss_all}
    with open('analyze_different_spaces/result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    dat = pd.DataFrame({'Space': loss_all.keys(), 'MSE': loss_all})
    dat.to_csv(f + '/result.csv', index=False)


def analyze_different_shape_of_train(spaces):
    '''
        Analyze loss function for different kernels.
        Also save plots for convergence of bayesian optimization for different kernels for each experiment
        :param spaces: dict - path to folder: info about space
        ----------------------------------------------------
        example:
        spaces = {'experiment_mass_usual_space_smaller': '100 samples',
                  'experiment_mass_usual_space_smaller_less_data': '50 samples',
                  'experiment_mass_usual_space_smaller_less_less_data': '20 samples'}
        ----------------------------------------------------
        :return: 'i.jpg' - plots of convergence for different kernels,
                 'result.json' - resulting loss for each experiment, mean loss for each kernel,
                 'result.csv' - mean loss for each kernel
        '''

    try:
        os.mkdir('analyze_different_train_shape')
    except:
        pass
    colors = ['red', 'green', 'DarkBlue', 'm']
    final_info = {}
    for i in tqdm(range(50)):
        final_info[i] = {}
        loss_all = {}
        plt.figure(figsize=(16, 12))
        for n, d in enumerate(list(spaces.keys())):

            loss_all[d] = 0
            p = spaces[d] + '/RatQuad/' + str(i) + '/'

            with open(p + 'predicted_data.json') as json_file:
                data = json.load(json_file)
                data_loss = data['all_way']['loss']
                loss = data['prediction_from_opt']['loss']
                true_val = data['real']
                pred_val = data['prediction_from_opt']
            loss_all[d] += loss
            final_info[i][d] = {'loss': loss,
                                'true_val': true_val,
                                'pred_val': pred_val}

            min_loss = []
            for k in range(len(data_loss)):
                min_val = np.min(data_loss[:k + 1])
                min_loss.append(min_val)

            plt.plot(min_loss, c=colors[n], label=spaces[d], linewidth=2.5, alpha=0.5)
            plt.scatter(np.arange(len(min_loss)), min_loss, c=colors[n], s=40, marker='+')
        plt.legend(fontsize=22)
        plt.ylabel('f', fontsize=22)
        plt.xlim((-2, 150))
        plt.xlabel('bayesian steps', fontsize=22)
        plt.savefig('analyze_different_train_shape/' + str(i) + '.jpg')
        plt.close()

    for k in loss_all.keys():
        loss_all[k] /= 50
    results = {'all': final_info, 'mean': loss_all}
    with open('analyze_different_train_shape/result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    dat = pd.DataFrame({'Space': loss_all.keys(), 'MSE': loss_all})
    dat.to_csv(f + '/result.csv', index=False)
