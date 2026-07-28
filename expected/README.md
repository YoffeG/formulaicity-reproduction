# Verified numerical baselines

The simulation trials were generated on 2026-07-28 from a fresh clone of
public GitHub commit `d15c25bac0cd86115e191216605c350c2bdf1750` in a newly
created Python 3.13.7 virtual environment. The final summary was sorted
deterministically and its uncertainty column made identical to the plotted
population standard deviation before the corrected vector PDF was committed.

Commands:

```bash
python -m pip install -r requirements-lock.txt
python figure1.py --workers 12
python figure6.py
python validate_reproduction.py
```

The complete Figure 1 run evaluated 8,100 simulated datasets and took
384.73 seconds. The complete Figure 6 run used six optimizer restarts, 200
iterations, 500 uncertainty subsamples, and 100,000 permutations per panel;
it took 58.02 seconds.

`figure_1_corrected.pdf` is the vector output of the fully rerun Figure 1
Bernoulli simulation. It is the corrected replacement for the older
archived-panel reconstruction. Its numerical source is
`figure_1_summary.csv`; it is not a redraw from the archived manuscript PDF.
The reported and plotted uncertainty is one population standard deviation
across the 100 simulations in each condition.

`validate_reproduction.py` first audits the Figure 1 Bernoulli loss and
analytic gradient against a direct implementation of manuscript Equations
7--10, then compares numerical summaries and metrics with tight tolerances.
It does not compare PDF or PNG bytes because embedded metadata and font
rendering can vary without changing the scientific result.
