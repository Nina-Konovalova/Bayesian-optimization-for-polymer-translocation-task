# Change GPyOpt lib
----------------------------------

For some reasons some part of GPyOpt lib were changed. To make same improvements - just follow theese steps.

1) Find file `GpyOpt/core/bo.py`;

2) Find the function `run_optimization()`;

3) Add blacklist:

```
 while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            # add while
            while True:
            # add argument in self._compute_next_evaluations (ignored_X = self.ignored_X)
                self.suggested_sample = self._compute_next_evaluations(ignored_zipped_X=np.array(self.ignored_X))


            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            # --- add ignored X
                self.evaluate_objective()
                if not self.problems:
                    break
                else:
                    #добавляем игноред иксы
                    if self.ignored_X is None:
                        self.ignored_X = []
                        self.ignored_X.append(self.suggested_sample)
                    else:
                        self.ignored_X.append(self.suggested_sample)
                print('ignored_X', len(self.ignored_X))
```

