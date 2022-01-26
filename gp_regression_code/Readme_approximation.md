# Fokker-plank function approximation
----------------------------------------

In this section the main task was to check different kernels and diffferent number of gaussians for approximation.

As the initial dataset - the synthetic dataset was used. It consists of train part, test part and experimental part.

For each data of experimental dataset the train and test targets were considered. As for creation of such target differet functions were used.

1) alpha x rate + MSE

2) angle&&&&&&

alpha was considered as different weight value from 0 to 1 with step &&&&


### Dataset
-----------------------------
To create datasets the following technic was used. In each dataset (train, experimental and test) samles consists of 1, 2, 3 (4) gaussians were used.

### Kernels
------------------------------
Several main kernels were considered:
- RatQuad, 
- Matern52,
- Matern32,
- RBF, 
- EXPQuad,
- Sum of RatQuad.