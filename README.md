# Bayesian optymization for polymer translocation problem
-----------------------------------------------------------
## Intro to a poblem
In this problem we are trying to solve problem of reconstruction the free energy landscape for polymer translocation
knowing information about time distribution for such translocation.

For more information you may read [proposal.pdf]()

---------------------------------------------------------

## Requirements

Make sure you have all dependencies from `requirements.txt` installed before proceeding. 

To install this dependencies run 

```
pip install -r requirements.txt
```

------------------------------------------------------------
## Quick start and results

Make directory for your output results: **<directory_for_output>**. In [Config.py]() change **EXPERIMENT_NAME**
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
python main_2.py -p <directory_for_output>
```

For detailed information about arguments, that can be changed, you can read [Documentation.md]().

-------------------------------------------------------------

## Compare time distribution and rate for predicted and initial landscapes.

There is also possibilities to make files of comparison, that contains information about:

1) true and predicted parameters for landscapes;
 
2) MSE for time diatributions for true and predicted landscapes

3) rates for true and predicted landscapes and their difference.

