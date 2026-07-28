# An Unsupervised Information-Theoretic Approach to Identifying Formulaic Clusters in Textual Data

This repository contains the reference implementation, data, and
reproducibility materials associated with the paper *An Unsupervised
Information-Theoretic Approach to Identifying Formulaic Clusters in Textual
Data*. It provides four command-line programs for regenerating Figures 1--6.
The programs use fixed random seeds, preserve intermediate results in CSV
format, resume interrupted computations, and produce vector PDF and PNG
outputs.

The Figure 1 result distributed in this repository is the complete rerun using
the corrected Bernoulli specification. It comprises 8,100 synthetic datasets
(nine panels, nine feature fractions, and 100 simulations) and is derived from
the simulation outputs rather than reconstructed from archived panel PDFs.
The corresponding vector figure is provided as
`expected/figure_1_corrected.pdf`; manuscript versions containing the earlier
archived-panel reconstruction should use this corrected figure.

The default statistical specifications are as follows:

- **Figure 1:** independent Bernoulli likelihood on binary feature activations.
- **Figure 2:** the manuscript Gaussian self-information objective with
  continuous weights; bounded L-BFGS-B replaces slow Powell evaluation.
- **Figures 3--5:** raw n-gram counts and a proper Multinomial likelihood.
  The information method and k-means receive the same count matrix.
- **Figure 6:** the same Multinomial text model. Formulaicity is assigned only
  after clustering, using the cluster with lower mean leave-one-out
  self-information.

The optional formulaicity-gap penalty remains disabled by default, so it does
not alter the published clustering objective.

## Installation

For exact reproduction, use Python 3.13.7 and the locked dependencies:

```bash
cd reproduction_suite
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`requirements.txt` retains compatible version ranges for exploratory use;
`requirements-lock.txt` records the environment used for the verified runs.

All examples below assume that `reproduction_suite/` is the working directory.

## Inputs

Figures 3--6 read the following files from the repository's `data/` directory.
The files are stored directly in the repository rather than as symbolic links,
making the repository self-contained:

- `Genesis_Lists_Narrative.csv`
- `Exodus_P-nonP_Roemer_AB.csv`
- `Leviticus _PH_14-3-2022.csv`
- `TOTHT_corpus_full`

Figure 2 uses the local, annotated reference module `figure2_reference.py`; it
no longer requires the Google Drive source file. Figure 1 is fully synthetic
and requires no data file. The numerical engines and optimizer are also
included in this directory, so no parent-project Python file is imported.

Redistribution of the biblical corpus and expert-annotation files should
conform to their applicable licenses. Their roles and SHA-256 checksums are
documented in `data/README.md`.

## Figure 1

Program: `figure1.py`

The nine panels use the parameter combinations in the manuscript. At every
fraction of formulaic dimensions, Cross-Entropy, k-means, GMM, and DBSCAN
receive the same binary realization. The plotted envelope is one population
standard deviation across simulations. The Cross-Entropy calculation follows
Equations 7--10 directly:

```text
p_j(s) = sum_i(s_i x_ij) / sum_i(s_i)
L(s) = -sum_i [s_i log P(x_i|s) + (1-s_i) log P(x_i|1-s)]
```

The `p_j` values are independent Bernoulli probabilities and are therefore
not normalized to sum to one across features. Both `log(p_j)` and
`log(1-p_j)` terms are included. Memberships stay continuous during
optimization and are thresholded once, after convergence.

```bash
python figure1.py \
  --simulations 100 \
  --workers 12
```

Audit the implementation independently against the displayed equations:

```bash
python audit_figure1_math.py
```

Outputs are written to `output/figure_1/`:

- `figure_1.pdf` and `figure_1.png`
- `figure_1_trials.csv` (one row per method and trial)
- `figure_1_summary.csv`
- `figure_1_metadata.json`

An interrupted run resumes automatically. To redraw without simulation:

```bash
python figure1.py \
  --simulations 100 \
  --plot-only
```

## Figure 2

Program: `figure2.py`

The default is 100 simulations per noise level and panel. For the
high-precision audit run, use 1,000:

```bash
python figure2.py \
  --simulations 1000 \
  --workers 12
```

Differential Entropy is disabled by default because it is disabled in the
current manuscript source. It can be restored explicitly with
`--include-differential-entropy`.

Outputs are written to `output/figure_2/`, including the
trial CSV, aggregate CSV, metadata JSON, PDF, and PNG. The long simulation
resumes from its trial CSV. To reformat existing results only:

```bash
python figure2.py \
  --plot-only
```

## Figures 3--5

Program: `figures3_5.py`

The following command processes Genesis, Exodus, and Leviticus concurrently:

```bash
python figures3_5.py \
  --book all \
  --score-model multinomial \
  --jobs 3
```

Each book contains 260 combinations:

- n-gram size `n = 1, 2, 3, 4, 5`
- running-window width
  `ell = 2, 3, 4, 6, 8, 10, 12, 14, 18, 22, 24, 26, 28`
- feature count `f = 100, 300, 500, all`

Every completed combination is appended immediately to its CSV, so the same
command safely resumes. A single book can be run with `--book genesis`,
`--book exodus`, or `--book leviticus`. For a limited pipeline verification:

```bash
python figures3_5.py \
  --book genesis \
  --limit-combinations 1 \
  --optimizer-restarts 1 \
  --optimizer-iterations 5
```

The complete PDFs are created only after all 260 combinations for a book are
present. To redraw all three completed figures:

```bash
python figures3_5.py \
  --book all \
  --plot-only
```

Outputs are written to `output/figures_3_5/`. The final PDF
names are:

- `genesis_gen_results_new.pdf`
- `Exodus_results_new.pdf`
- `leviticus_results_new.pdf`

`--score-model binary` and `--score-model binomial` are retained only for
diagnostic comparisons. They are not the default count-text analysis.

## Figure 6

Program: `figure6.py`

The program generates both panels in parallel, writes their feature tables,
and stacks the vector panels into one PDF:

```bash
python figure6.py \
  --score-model multinomial \
  --jobs 2
```

The fixed configurations are:

- panel (a): `ell = 12`, `n = 3`, `f = 500` (highest Multinomial MCC);
- panel (b): `ell = 6`, `n = 5`, `f = 500`
  (selected H-formulaic result).

The main axes report normalized feature importance with repeated half-sample
uncertainty. The inset reports the empirical self-information distributions;
blue is the non-formulaic cluster and orange is the formulaic cluster. The
sigma annotation is the absolute mean separation divided by its
randomization-null standard deviation.

Outputs are written to `output/figure_6/`:

- `figure_6.pdf` and `figure_6.png`
- one PDF, PNG, and feature CSV for each panel
- one machine-readable metrics JSON for each panel
- `figure_6_metadata.json`

To restack existing panel PDFs without refitting:

```bash
python figure6.py \
  --combine-only
```

## Validate a complete reproduction

After running Figures 1 and 6 with their defaults, compare their numerical
outputs with the independently verified clean-clone baselines:

```bash
python validate_reproduction.py
```

The command first audits the Figure 1 loss and analytic gradient against a
separate direct implementation of Equations 7--10. It also checks every
installed package against `requirements-lock.txt`. It compares numerical
summaries rather than PDF bytes, thereby permitting non-substantive
differences in font rendering or embedded metadata. The clean-run baselines,
corrected Figure 1 vector PDF, and verification record are in `expected/`.

## Reproducibility notes

- All random generators have explicit seeds.
- Simulation checkpoints retain per-trial seeds.
- Figures have a white background, no grid, Times New Roman-compatible serif
  fonts, consistent 14-point ticks and labels, and vector PDF output.
- Existing checkpoint CSVs should not be mixed across different model choices.
  Use a different `--output-dir` when comparing Multinomial, Binomial, binary,
  or formulaicity-regularized runs.
- Use `--help` on any program for all numerical and output options.
