# Bayesian optymization for polymer translocation problem
-----------------------------------------------------------
## Intro to a poblem
Polymer translocation can be considered in terms of random walks along the free energy profile. This process can be described by the Fokker-Planck equation. Depending on the free energy profile different time distributions of successful (unsuccessful) translocation may be obtained. 


The problem, that should be solved, can be formulated the following way. Suppose there is a translocation time distribution for a successful case, a similar time distribution for an unsuccessful translocation, as well as a percentage of successful translocation. It is necessary to reproduce the energy profile corresponding to these conditions as accurately as possible.

## Methods
This problem is going to be solved using Bayesian optimization \cite{snoek2012practical}. The whole task can be described the following way:

1) First of all the dataset should be constructed. Then this dataset should be parameterized. Different options of parametrization are considered:
 - Chebyshev polynomials with different degrees;
 - Several numbers of Gaussian functions with fixed means;
 - Several numbers of Gaussian functions with optimized means;

2) For each known profile find the corresponding time translocation distribution for successful and unsuccessful translocation and translocation rate.

3) Chose function for minimization: Here we look for MSE for time distributions of successful and unsuccessful distributions for real and not real distributions, summing that with the difference between rates.

4) Using the initial dataset and Gaussian Processes \cite{williams2006gaussian} - build the model for Bayesian optimization is built.

5) Use active learning for optimization

---------------------------------------------------------

## Requirements

Make sure you have all dependencies from `requirements.txt` installed before proceeding. 

To install this dependencies run 

```
pip install -r requirements.txt
```

------------------------------------------------------------
## Quick start

1) Make directory for your output results: **<directory_for_output>**. In that directory you should make such folders as
 
 - **big_rate** - save pictures in *.png* format of landscape and parameters that define this landscape for experimnets, that leads to rate equial 1e20;
 - **bigger_one** - save pictures in *.png* format of landscape and parameters that define this landscape for experimnets, that leads to rate bigger than 1;
 - **out_space** - save pictures in *.png* format of landscape and parameters that define this landscape for experimnets, that contains derivative bigger than 4.

In **<directory_for_output>** will be saves:

 - pictures of experiments, that contains initial landscape and predicted landscape;
 - parameters of model for each iteration step;
 - report file with information about experiment.


To run your test video just run:

```
python main.py -p <directory_for_output>
```

For detailed information about arguments, that can be changed, you can read [Documentation.md]().

-------------------------------------------------------------
## Results

The results for different kernels and different numbers of gaussians can be found via link.

--------------------------------------------------------------

## Compare time distribution and rate for predicted and initial landscapes.

There is also possibilities to make files of comparison, that contains information about:

1) true and predicted parameters for landscapes;
 
2) MSE for time diatributions for true and predicted landscapes

3) rates for true and predicted landscapes and their difference.

