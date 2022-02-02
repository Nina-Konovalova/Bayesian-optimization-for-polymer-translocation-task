# Change GPyOpt lib
----------------------------------

For some reasons some part of GPyOpt lib were changed. To make same improvements - just follow theese steps.

1) Find file **GpyOpt/core/bo.py**;

2) Change the function `run_optimization()` and `evaluate_objective()`; it will add possibility to check problems
and make black list

```
    def run_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, save_models_parameters= True, report_file = None, evaluations_file = None, models_file=None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context
        # add bad_samples_list and problems - if added sample has problems
        self.ignored_X = None
        self.problems = False

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            while True:
                # add argument in self._compute_next_evaluations (ignored_X = self.ignored_X)
                self.suggested_sample = self._compute_next_evaluations(ignored_zipped_X=np.array(self.ignored_X))

                # --- Evaluate *f* in X, augment Y and update cost function (if needed)
                # --- add ignored X
                self.evaluate_objective()
                if not self.problems:
                    break
                else:
                    # добавляем игноред иксы
                    if self.ignored_X is None:
                        self.ignored_X = []
                        self.ignored_X.append(self.suggested_sample)
                    else:
                        self.ignored_X.append(self.suggested_sample)
                print('ignored_X', len(self.ignored_X))

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)
```
```buildoutcfg
def evaluate_objective(self):
        """
        Evaluates the objective
        """
        self.Y_new, cost_new, self.problems = self.objective.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        self.Y = np.vstack((self.Y,self.Y_new))
```
3) Then find **GPyOpt\core\task\objective.py** and change functions:

```buildoutcfg
def evaluate(self, x):
        """
        Performs the evaluation of the objective at x.
        """

        if self.n_procs == 1:
            f_evals, cost_evals, problem = self._eval_func(x)
        else:
            try:
                f_evals, cost_evals, problem = self._syncronous_batch_evaluation(x)
            except:
                if not hasattr(self, 'parallel_error'):
                    print('Error in parallel computation. Fall back to single process!')
                else:
                    self.parallel_error = True
                f_evals, cost_evals, problem = self._eval_func(x)

        return f_evals, cost_evals, problem
```

```buildoutcfg
def _eval_func(self, x):
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each
        evaluation is also provided.
        """
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        x_new = x.copy()  #####  Добавила копию х
        for i in range(x.shape[0]):
            st_time    = time.time()
            rlt, x_new[i], problem = self.func(np.atleast_2d(x[i]))
            f_evals     = np.vstack([f_evals,rlt])
            cost_evals += [time.time()-st_time]
        return f_evals, cost_evals, problem
```


