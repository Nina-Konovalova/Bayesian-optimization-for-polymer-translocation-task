# Make your dataset
----------------------------------
Here you can find some instruction for creation dataset. 

First of all, max number of gaussians in dataset 3, 4 or 5. Dataset contains all possible combination of
gaussians from 1 to max number of gaussians and for each possible combination you can set 
sample numbers. Also, for each number of gaussians it is possible to set
amplitude and bias for both dispersion and amplitude for each gaussians.
That means, that your amplitude and std will take following possible values:

- Amplitude: (-bias - amp; -bias) and (bias; bias + amp);
- Std: (bias; bias + amp).

All these parameters can be changes in [config_dataset.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/make_dataset/config_dataset.py).

Also in that file you may change mode. Usually mode means *train*/*exp*/*test*.

To start creating dataset just run:

```buildoutcfg
python main_dataset.py --dir_name=<dir to save dataset> --num_of_all_g=<max num of gaussians(3,4,5)> --plot=<False/True>
```

As the result you will have dir *dir to save dataset* in which you will have 
**.npz** files for data with 1,2 ... num_of_all_g gaussians, one **.npz** and **.json** files with
all dataset. If you had `--plot True` you will also have dir with `mode` name in which you will
have all plots of initial landscapes, log time distributions for these landscapes and rate of
successful translocation as title for plot.

## Dataset analysis
-----------------------------------
You also can plot histograms for main parameters of your data.
Such parameters include following values:

1) rates;

2) tilts angles for log time distributions;

3) mean times translocation;

As we have 2 main possible options for amplitude: positive and negative - histograms for
positive and negative amplitudes will be saved in different files. To make these plots just run:

```buildoutcfg
python dataset_analysis.py --path_to_data=<path_to_npz_file_with_data>
```

