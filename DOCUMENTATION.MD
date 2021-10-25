# Different optimization commands
----------------------------------

## Arguments

### Model's parameters

`--kernel_type` or `-kernel` - type of kernel, that should be used. Default - **Matern52**, another options: **RatQuad, ExpQuad, RBF, Matern32**;

`--model_type` or `-model` - type of model. Default - **GP**. Other possible options: **GP_MCMC**;

### Optimization parameters

`--acquisition_type` or `-a` - acquisition function type. Default - **MPI**. Other possible options: **EI**;

`--normalize` or `-n` - whether the output should be normalized. Default - **False**;

`--evaluator_type` or `-eval` - type of optimizer to use. Default - **lbfgs**. Other possible options: **CMA**: covariance matrix adaptation;

`--num_cores` or `-c` - number of used cores. Default - **1**;

`--number_steps` or `-steps` - number of optimization steps. Default - **70**.

### Other parameters

`--path_for_save` or `-p` - path for saving outputs of experiments.

`--path_experiments` or `-exp` - path for experimental data. Default - **experimental_data.csv**


---------------------------------------------------

## Configuration

