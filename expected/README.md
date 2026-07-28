# Verified numerical baselines

These files were generated on 2026-07-28 from a fresh clone of public GitHub
commit `d15c25bac0cd86115e191216605c350c2bdf1750` in a newly created Python
3.13.7 virtual environment.

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

`validate_reproduction.py` compares numerical summaries and metrics with
tight tolerances. It does not compare PDF or PNG bytes because embedded
metadata and font rendering can vary without changing the scientific result.
