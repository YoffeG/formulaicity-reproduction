# Reference results

This directory contains the numerical reference results used by
`validate_reproduction.py`. They were generated with Python 3.13.7 and the
package versions recorded in `requirements-lock.txt`.

Commands:

```bash
python -m pip install -r requirements-lock.txt
python figure1.py --workers 12
python figure2.py --workers 12
python figure6.py
python validate_reproduction.py
```

The complete Figure 1 run evaluated 8,100 simulated datasets and took
384.73 seconds. The complete Figure 2 run evaluated 3,600 simulated datasets
and took 101.26 seconds. The complete Figure 6 run used six optimizer
restarts, 200 iterations, 500 uncertainty subsamples, and 100,000 permutations
per panel; it took 58.02 seconds.

`figure_1_reference.pdf` is the vector output of the Figure 1 categorical
benchmark. Its numerical source is `figure_1_summary.csv`. The reported and
plotted uncertainty is one population standard deviation across the 100
simulations in each condition.

`figure_2_reference.pdf` is the vector output of the Figure 2 Gaussian
benchmark. Its numerical source is `figure_2_summary.csv`. The weighted
covariance uses the Equation 19 normalization, and the simulated covariance
spectra have eigenvalues 10 and 30.

`validate_reproduction.py` audits the Figure 1 Bernoulli loss and analytic
gradient against a direct implementation of Equations 7--10 and the Figure 2
weighted covariance against Equation 19, then compares numerical summaries
and metrics with tight tolerances.
It does not compare PDF or PNG bytes because embedded metadata and font
rendering can vary without changing the scientific result.
