# Bayesian optymization for polymer translocation problem
-----------------------------------------------------------

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

## Compare time distribution and rate for predicted and initial landscapes.


