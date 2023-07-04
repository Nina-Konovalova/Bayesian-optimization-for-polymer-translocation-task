# Bayesian optimization for polymer translocation problem
-----------------------------------------------------------
This is my Master's thesis written during my education at Skoltech University.

## Intro to a problem
This work applies Bayesian optimization to recover information on polymer chemistry from statistical data on polymer translocation through a narrow opening in a planar membrane. In particular, two problems are considered:
- determining the polymer length distribution from the distribution of translocation time (Fig. 1);
- deciphering the translocation-free energy profile and thus obtaining information on polymer chemistry (Fig. 2).

To solve these problems, we assume that the dynamic of polymer translocation is described by the Fokker-Plank (FP) equation that models one-dimensional motion in a free energy landscape, with the translocation degree chosen as the order parameter. The FP equation solves the direct problem: obtaining the translocation time distribution and success rate for known polymer structure. The inverse problem of deciphering polymer properties is solved by an active learning scheme: the Bayesian optimization algorithm <<queries>> the system to produce the Fokker-Planck solution for a particular polymer. The result is added to the database of known solutions and the optimal structural characteristics that produce the best match to the experimental data are found. 

The work focuses on the methodology of Bayesian optimization and explores how the target functions, vector output, algorithm details and the quality of the translocation time distributions affect the accuracy of the inverse problem solution. At the same time, we hope that the work also has a practical significance and nanopore sensing devices can be applied in analytical labs dealing with polymer production and quality control.
   
Besides the classical algorithm of Bayesian optimization it is also possible to add several improvements:
   - vector output processing
   - functional output processing
   - input model warping
   - optimize the logarithm of the objective function

An example of decyphering polymer length distribution is presented below
   
<p align="center">
  <img src="https://github.com/Nina-Konovalova/bayes_experiment/blob/main/images/16pdf.jpg" width="250"  alt="Free energy profile decyphering for one Gaussian" >
  <img src="https://github.com/Nina-Konovalova/bayes_experiment/blob/main/images/6pdf.jpg" width="250"  alt="Free energy profile decyphering for two Gaussians" >
  <img src="https://github.com/Nina-Konovalova/bayes_experiment/blob/main/images/5pdf.jpg" width="250"  alt="Free energy profile decyphering for three Gaussians" >
</p>
<p align="center">   
   <em> Fig.1 Polymers length decyphering</em>
</p> 

<p align="center">
  <img src="https://github.com/Nina-Konovalova/bayes_experiment/blob/main/images/one_gaussian.png" width="250"  alt="Free energy profile decyphering for one Gaussian" >
  <img src="https://github.com/Nina-Konovalova/bayes_experiment/blob/main/images/two_gaussians.png" width="250"  alt="Free energy profile decyphering for two Gaussians" >
  <img src="https://github.com/Nina-Konovalova/bayes_experiment/blob/main/images/three_gaussians.png" width="250"  alt="Free energy profile decyphering for three Gaussians" >
</p>
<p align="center">   
   <em> Fig.2 Free energy landscape decyphering</em>
</p>   
   

---------------------------------------------------------

## Requirements

Make sure you have all dependencies from `requirements.txt` installed before proceeding. 

To install this dependency run 

```
pip install -r requirements.txt
```
Also you have to check [change_gpy_opt.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/change_gpy_opt.md)
and change GPy library

------------------------------------------------------------
## Quick start and results

### Translocation task

Make directory for your output results: **<directory_for_output>**. In [Config.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/Configurations/Config.py) change **EXPERIMENT_NAME**
as you want. In **<directory_for_output>** will be made new one with **EXPERIMENT_NAME**. Then for each experiment new directory
   will be made with the name equals to the number of experiment. In each dir the following information will be contained:
   - dir with optimization steps - pictures for each successful optimization step
    - *experiment_i.png* - picture of true profile (blue color), best from optimization steps (green color), best for optimization (red color)
    - *model_params_i.txt* - parameters for model (kernel hyperparameters)
    - *predicted_data.json* - all optimization steps with vector of parameters for each step and value of target function.
    Also best predicted result vector, best vector of parameters for just optimization steps and real vector of parameters for landcape
   - *report_file.txt* - some information about experiment
   - *results_compare_a.csv* - file with different loss functions for each successful step (mse loss, rate loss, and so on)

To run your test video just run:

```
python main_2.py -t translocation
```

### Mass distribution task

The second option is to run recovery for the mass distribution task. All configs and arguments can be found

To run your test video just run:

```
python main_2.py -t mass_distribution
```


For detailed information about arguments, that can be changed, you can read [Documentation.md](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/DOCUMENTATION.MD).
  
-------------------------------------------------------------

## Compare time distribution and rate for predicted and initial landscapes.

There is also possibility to make files of comparison, that contain information about:

1) true and predicted parameters for landscapes;
 
2) MSE for time distributions for true and predicted landscapes

3) rates for true and predicted landscapes and their difference.
  
 

