# Make your dataset
----------------------------------

## Make dataset for translocation task

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

### Dataset analysis

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

--------------------------------------------------
## Make dataset for mass distribution task

In this task we want to generate dataset for different mass (monomer length).

Dataset contains different samples, that depends on shape and scale for mass distribution. 
Scale and shape are chosen randomly from uniform distribution of space, that can be changed in `Configuration/config_mass.py`.

For each sample we have corresponding to it gamma distribution. For each gamma distribution
we take `x=np.linspace(a,b, num_samples)` (can be changed in `Configuration/config_mass.py`).

For each `x` element and constant enegry landscape (you can change constat in `Configuration/config_mass.py`) 
using fortran Fokker-Plack solver we get 

**time_distribution_for_success * rate + time_distribution_for_unsuccess * (1-rate)**

And then this value multiply on `gamma_distribution(x_element)`. All such values are summarized.

So for each gamma distribuion we get the value, described before.

To start creating dataset just run:

```buildoutcfg
python main_dataset_mass.py --dir_name=<dir to save dataset> --num_of_samples=<num of samples in dataset> --mode=<train/test/experiment>
```
As the result you will have *dir_name/mode* directory and the following results, saved there:

- **sample_images** directory with gamma distribution samples saved as image;

- **samples_info.npz** file, that containes:
    1) shape for each dataset sample - vector with size `num_of_samples x 1`;
    2) scale for each dataset sample - vector with size `num_of_samples x 1`;
    3) info from FP solution - matrix with size `num_of_samples x num_of_elements_in_distribution x 10_000`;
    4) summarized value from info - vector with size `num_of_samples x 10_000`;



