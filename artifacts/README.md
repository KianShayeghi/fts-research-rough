# artifacts

Generated files, committed so that the notebooks open and the tests run without R and without
refitting. They are kept in two directories rather than one because they are rebuilt by
different commands, and only one of the two is reproducible here.

## `reference/`

What CRAN `fdaSP::f2sSP` produced, for `fdaSP.SBC` to be checked against. `R_basis_bs.csv` is
the B-spline basis `f2sSP` builds internally; `tecator_match_*.csv` its Tecator fit at four
fixed lambdas together with the design it fitted on; `tecator_f2sSP_beta_path.csv` and
`tecator_f2sSP_lam_path.csv` its whole automatically chosen lambda path; and
`tecator_f2sSP_cv*.csv` the cross-validation curve behind the paper's own tuning rule.

These are cross-language fixtures, so they are committed on purpose: neither
`tests/test_f2ssp_match.py` nor `notebooks/LocSpaFLR_validation.ipynb` needs R installed to
run, and the numbers they compare against cannot drift with a local R version. Where R with
the `fdaSP` package is available:

```bash
python scripts/make_r_artifacts.py
```

The R source itself is `R_SNIPPETS` in `src/fts_research/fdaSP/r_reference.py`.

## `cache/`

What this repository produced, where recomputing it takes minutes. `basis_target_study.csv`
holds one row per replication of the nine-cell `bench` comparison, `basis_target_support.csv`
how often each method left each grid point in the support, and `basis_target_curves.csv` the
coefficient functions every method fitted on the showcase replication.

Every replication is seeded from `bench.study.CONFIG` by `bench.study.seed_of`, so a rebuild
on the same configuration reproduces these files:

```bash
python scripts/make_study_cache.py --refresh
```

`shang_rolling.csv` and `shang_rolling_panel.csv` are the other pair: one row per held-out
session and intraday point of the rolling-origin evaluation in `ftsa.evaluation`, holding what
each method forecast, the band it put around it, and what was realised. The first is the S&P 500
replication design of 2017, the second the same evaluation across the nine instruments of the
five-minute panel. What is cached is the forecasts rather than the scores, so a measure this
repository has not thought of yet can be computed from them without refitting -- every number in
`notebooks/shang_replication.ipynb` is derived from these two files rather than from a stored
summary.

`shang_updating.csv` and `shang_updating_profiles.csv` are the updating half of the same
replication: the first holds one row per held-out session, update point, anchoring and method,
carrying the sums every accuracy measure is built from, and the second the mean error at each
intraday point. Sums per session rather than every forecast, in the way the study above keeps
per-replication records: that sweep crosses nine update points with thirteen method-anchoring
pairs, so a pointwise cache would repeat each realised curve thirteen times and reach fifteen
megabytes for nothing the analysis uses. What is kept is what a paired test needs, every method
having forecast the same fifty sessions.

```bash
python scripts/make_ftsa_cache.py --refresh
```

The design is `ftsa.evaluation.CONFIG` with `ftsa.updating.UPDATE_FRACTIONS`, and the bootstrap
is seeded from it, so a rebuild on the same configuration reproduces the committed files.

Everything here is committed so that `notebooks/basis_and_coef_comp.ipynb` and
`notebooks/shang_replication.ipynb` open on a fresh checkout without a multi-minute fit, not
because they could not be recreated.
