Approximation the loss function with gaussian processes.
-------------------------------------------------------

In this folder you can run experiments to explore the quality of approximation of loss function.

Firstly, it is necessary to tune configuration files.

In the file [GP_config.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/gp_regression_code/GP_config.py)
you have to provide paths to train, test and val files. 
Each of these files has to be in **.npz** format and contain information about:

- *'vecs'* - parametrized initial landscape;
- *'rates''* - successful rate translocation for each energy profile;
- *'times'* - translocation times ([successful time, unsuccessful time]);
- *'y_pos'* - time distribution for successful translocation;
- *'y_neg'* - time distribution for unsuccessful translocation;  
- *'angs'* - angle of slope for logarithmic time distribution ([successful angle, unsuccessful angle]).

Also, you can change in different ways the target function for approximation.
To do it - just change `function()` in [help_functions.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/utils/help_functions.py). You can choose any combination of rate, times, angles and time distributions.
Now it is set as `angles_difference + ALPHA * rates_difference` and the `ALPHA` can be chosen in
[GP_config.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/gp_regression_code/GP_config.py).

Also in configuration file [GP_config.py](https://github.com/Nina-Konovalova/bayes_experiment/blob/main/gp_regression_code/GP_config.py)
you can set the following parameters:

- *'INPUT_DIM'*: - dimension for kernels. Usually the dimension is equal doubled number of gaussains for parametrization.
- *'Grid'* - different kernels, that you want to use. Any number of kernels can be set.

To run experiments just run:

```buildoutcfg
python main_reg_2.py --save_path=<path to save the results>
```

If no *path to save the results* exists - it will be made. In this folder there will be made
directory with 'ALPHA value' name. For each kernel will be made separate directory. 
In each experimental directory the model, **.npz** file with predicted test results and true test data and **.json** file with metrics:
- mse
- r2 error
- explained_variance_score



