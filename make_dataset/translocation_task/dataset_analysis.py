import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
sns.set_theme()


def file_analythis_pos(path):
    '''
    :param path: path to .npz file, that contains data 'angs', 'rates', 'times' and parametrization 'vecs'
    :return: save historgrams for 'angs' (successful and unsuccessful translocation), 'rates' and 'times'
    (successful and unsuccessful translocation) for potential hills
    '''
    data = np.load(path)
    files = ['angs', 'rates', 'times']
    print(data['rates'].shape)
    print(data['angs'].shape)
    print(data['times'].shape)
    for file in files:
        print(os.path.splitext(path)[0] + '_' +  file +  '_distribution_pos.jpg')
        # try:

        if len(data[file].shape) > 1:
            if data[file].shape[1] == 2:
                fig, axs = plt.subplots(1, 2, figsize=(16, 10))
                axs[0].hist((data[file][:, 0][(data['vecs'][:, 3:] > 0).sum(axis=1) > 0]), bins=80)
                axs[1].hist((data[file][:, 1][(data['vecs'][:, 3:] > 0).sum(axis=1) > 0]), bins=80)
                axs[1].set_ylabel('Count', fontsize=20)
                axs[1].set_xlabel(file, fontsize=20)
                axs[0].set_ylabel('Count', fontsize=20)
                axs[0].set_xlabel(file, fontsize=20)
                axs[1].grid(True)
                axs[0].grid(True)
                axs[0].set_title('successful', fontsize=20)
                axs[1].set_title('unsuccessful', fontsize=20)
                plt.savefig(os.path.splitext(path)[0] + '_' +  file +  '_distribution_pos.jpg')

            else:
                plt.figure(figsize=(16, 12))
                sns.histplot(data=data[file][(data['vecs'][:, 3:] > 0).sum(axis=1) > 0], kde=True, bins=100,
                             color='blue')
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.ylabel('Count', fontsize=20)
                plt.xlabel(file, fontsize=20)
                plt.grid(True)

                plt.savefig(os.path.splitext(path)[0] + '_' +  file +  '_distribution_pos.jpg')
        else:
            plt.figure(figsize=(16, 12))
            sns.histplot(data=data[file][(data['vecs'][:, 3:] > 0).sum(axis=1) > 0], kde=True, bins=100, color='blue')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylabel('Count', fontsize=20)
            plt.xlabel(file, fontsize=20)
            plt.grid(True)

            plt.savefig(os.path.splitext(path)[0] + '_' +  file +  '_distribution_pos.jpg')


def file_analythis_neg(path):
    '''
        :param path: path to .npz file, that contains data 'angs', 'rates', 'times' and parametrization 'vecs'
        :return: save historgrams for 'angs' (successful and unsuccessful translocation), 'rates' and 'times'
        (successful and unsuccessful translocation) for potential hole
    '''
    data = np.load(path)
    files = ['angs', 'rates', 'times']
    for file in files:
        print(os.path.splitext(path)[0] +'_' +  file +  '_distribution_neg.jpg')
        # try:
        if len(data[file].shape) > 1:
            if data[file].shape[1] == 2:
                fig, axs = plt.subplots(1, 2, figsize=(16, 10))
                axs[0].hist((data[file][:, 0][(data['vecs'][:, 3:] < 0).sum(axis=1) > 0]), bins=80)
                axs[1].hist((data[file][:, 1][(data['vecs'][:, 3:] < 0).sum(axis=1) > 0]), bins=80)
                axs[1].set_ylabel('Count', fontsize=20)
                axs[1].set_xlabel(file, fontsize=20)
                axs[0].set_ylabel('Count', fontsize=20)
                axs[0].set_xlabel(file, fontsize=20)
                axs[1].grid(True)
                axs[0].grid(True)
                axs[0].set_title('successful', fontsize=20)
                axs[1].set_title('unsuccessful', fontsize=20)
                plt.savefig(os.path.splitext(path)[0] +'_' +  file +  '_distribution_neg.jpg')

            else:
                plt.figure(figsize=(16, 12))
                sns.histplot(data=data[file][(data['vecs'][:, 3:] < 0).sum(axis=1) > 0], kde=True, bins=100,
                             color='blue')
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.ylabel('Count', fontsize=20)
                plt.xlabel(file, fontsize=20)
                plt.grid(True)

                plt.savefig(os.path.splitext(path)[0] +'_' +  file +  '_distribution_neg.jpg')
        else:
            plt.figure(figsize=(16, 12))
            sns.histplot(data=data[file][(data['vecs'][:, 3:] < 0).sum(axis=1) > 0], kde=True, bins=100, color='blue')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylabel('Count', fontsize=20)
            plt.xlabel(file, fontsize=20)
            plt.grid(True)

            plt.savefig(os.path.splitext(path)[0] +'_' +  file +  '_distribution_neg.jpg')


def main():
    '''
    :return: analyze distribuion of angs, rates and times for potential hills and holes for chosen dataset
    '''
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--path_to_data',
                        default='../dataset_3_gaussians_small_3/exp_gaussians_3_train.npz',
                        type=str,
                        metavar='PATH',
                        help='Path to dir for saving data')

    args = parser.parse_args()
    file_analythis_pos(args.path_to_data)
    file_analythis_neg(args.path_to_data)



if __name__ == '__main__':
    main()



