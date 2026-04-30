# Round 8 Summary

`fdp`: multi-seed evidence does not rescue the parked residual. Five-seed
analysis of `classic_fmmm` vs `graphviz_fdp` produced a dagua-vs-graphviz
median of `0.248375` against a within-graphviz floor below `0.000001`; every
graph with enough graphviz seed pairs was `not_equivalent` under TOST.

`sfdp`: multi-seed evidence also leaves the residual in place. Five-seed
analysis of `classic_sfdp` vs `graphviz_sfdp` produced a dagua-vs-graphviz
median of `0.107379` against a `0.000000` median within-graphviz floor; every
testable graph was `not_equivalent`.

`neato`: the Round 7 outlier residual could not be statistically reclassified
because this benchmark cache has no `graphviz_neato__seed*.pt` files. The
stress and MDS medians remain `0.035298` and `0.045489`, respectively, but
there is no measured within-graphviz stochastic floor for the worst outliers.

Round 8 result: comparator infrastructure is now reusable for multi-seed
stochastic-floor checks. No parked family moves from residual to converged based
on the available cache.
