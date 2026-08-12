# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A test rig for a **fork of LightGBM** that adds an `unused_feature_penalty` parameter (regularised trees / redundancy-penalising feature selection). This repo contains no model code — it generates data with *known* ground truth and measures whether the penalty actually selects the right features.

The core idea: on real data you can usually only measure downstream accuracy loss, because the true relevant feature set is unknown. With planted structure you can measure *correctness of selection* directly.

**The project has reached its conclusion — read `README.md` first.** The method beats its matched baseline under a random forest (10/10 seeds) and loses under gradient boosting (10/10), and top-k by gain importance beats every configuration tested. New work should be presentation, replication, or literature-checking rather than further tuning; the experimental arc (mechanism → targeted fix → positive control) is complete.

## Critical external dependency

`pyproject.toml` pins LightGBM to a **local wheel**:

```
lightgbm @ file:///Users/cono/code/LightGBM_RegularisedTrees/dist/lightgbm-4.6.0.99-py3-none-macosx_26_0_arm64.whl
```

The `unused_feature_penalty` and `unused_feature_penalty_scope` params only exist in that build. If a run errors on an unknown parameter, the installed library is stale — that's the first thing to check.

**Don't rebuild the wheel during development.** `../LightGBM_RegularisedTrees/rebuild-dev.sh` builds only the C++ shared library and copies it straight into this project's `.venv`, which takes seconds instead of minutes. It works because the LightGBM Python package is a thin ctypes wrapper that locates the library at import time and does not validate parameter names, so a new C++ config field needs no Python-side change. Use `uv sync --reinstall-package lightgbm` only when you want a real wheel again.

That script also re-signs the dylib. On Apple Silicon, overwriting a dylib in place invalidates its signature and the kernel SIGKILLs the loading process — Python exits 137 with *no output at all*, which is otherwise very hard to diagnose.

## Commands

```bash
uv sync                                          # install/refresh the env

uv run python -m trees_testing.run_milestones           # all experiments (~10 min)
uv run python -m trees_testing.run_milestones E3        # just the justification test
uv run python -m trees_testing.run_milestones --quick   # plumbing check, NOT results

uv run pytest tests/ -q                          # harness's own correctness tests
uv run python -m trees_testing.test_harness      # legacy generator/metric checks

uv run jupyter lab                               # notebooks/
```

**`src/trees_testing/test_harness.py` is not a pytest module** despite the name — it is a script of module-level `check(...)` asserts, and `pytest` on it collects 0 items. Run it as a module (above). The real pytest suite is `tests/test_harness_properties.py`.

## Architecture

Three layers, deliberately decoupled so the first two can be verified without LightGBM:

**`synthetic.py`** — pure numpy generators. Each returns `(X, y, truth)`, where `truth` is a dict of ground-truth column indices **already remapped through the column shuffle**. `_shuffle_columns` permutes features (so position leaks no structure) and carries the bookkeeping through; any new generator must go through it and return post-shuffle indices.

| Generator | Probes |
| --- | --- |
| `make_redundant_blocks` | Does the penalty collapse each correlated block to ~one representative? Perfect selector: `\|F\| == n_blocks` |
| `make_relevant_irrelevant` | Recovery precision/recall/F1 vs penalty strength, with no redundancy present |
| `make_xor_interaction` | The failure-mode probe: features individually uninformative but jointly determining. A marginal-gain penalty may kill half a pair |

`xor_probe_power` is not optional decoration. The XOR probe originally had **no statistical power** — main effects and label noise swamped the interaction, so destroying a pair cost 0.008 accuracy and `pairs_broken` was unfalsifiable. Defaults are now `n_main_effects=0` and `xor_weight=4.0`; E4 measures power first and *suppresses its own results* if the gap is too small to mean anything.

**`selection_harness.py`** — metrics, baselines, and the pluggable fit path.

- **The selected set F is "used in ≥1 split anywhere in the model"** — `booster.feature_importance('split') > 0`, via `used_features()`. Matches the paper.
- **`fit_fn` is injectable** — `fit_and_select(..., fit_fn=...)` takes any callable `(X, y, penalty, param_name, task, num_boost_round, extra_params, seed) -> (split_importance, fitted_object)`. LightGBM is imported *inside* `_default_lightgbm_fit` only.
- **`selection_curve(booster)` derives the whole |F|-vs-rounds curve from one fit**, relying on boosting being prefix-consistent: the first r trees of an R-round model equal an r-round model. That assumption is asserted against independent fits in `tests/test_harness_properties.py` — if it ever breaks, every convergence result breaks with it.
- **`tree_diagnostics`** reports features-per-tree. Every selection metric here is computed on the union across trees and is therefore blind to trees collapsing to single-feature stumps, which is the real failure mode at small λ. Never report a selection metric without it.

**`real_data.py`** — MADELON via OpenML, with ground truth *recovered and verified* rather than trusted. The challenge published the counts (20 real / 480 probes) but not the indices; the correlation rule (`|corr| > 0.5` with any other feature) recovers exactly 20, and those are the same 20 that top gain importance. `load_madelon` raises rather than returning a silently-wrong truth set if either check fails. Blocks use **average** linkage — single linkage chains 18 of the 20 into one block and makes `block_coverage` trivially 1.00 for every selector.

**`run_milestones.py`** — eight experiments, E1–E8, each printing a "READ:" block. E8 is the headline (base-learner swap); E6 is the acceptance test on real data. Every experiment is written to report a negative verdict plainly if that's what the numbers say, and several do.

## Three things that will bite you

**1. The penalty convention is settled — don't re-derive it.** The fork hard-checks `0.0 ≤ unused_feature_penalty ≤ 1.0`, default `1.0`. **1.0 = vanilla, smaller = stronger**, matching the paper's λ. Passing `None` omits the param and is identical to `1.0` (asserted in the test suite). All the dynamic range is below ~0.1, so sweep log-spaced; a linear sweep wastes most of its points.

**2. Scope decides whether this is a feature selector at all.** `unused_feature_penalty_scope` defaults to `tree`, under which F resets every tree and |F| **never converges** — it grows with `num_boost_round` toward "all features", so any |F| quoted at a fixed round count is an artifact of where you stopped. Use `ensemble` (the paper's semantics) for anything about selection. E1 exists to make this visible and distinguishes three outcomes: `SELECTED` (converged to a proper subset), `saturated` (stopped only because every feature is in F — what vanilla does), `growing`.

**3. |F| is confounded with boosting rounds.** Hold `num_boost_round` fixed across every compared model and never use early stopping in a comparison.

**4. Any new comparison must be matched on cardinality and paired on seed.** The whole project turns on head-to-heads against `top-k by gain importance`, which is a much stronger baseline than it looks — beating random subsets or all-features proves nothing. `H.aggregate` plus a paired per-seed difference with its SE is the pattern; single-seed numbers here are indistinguishable from noise.

**5. `selection_curve` assumes boosting is prefix-consistent** — that the first r trees of an R-round model equal an r-round model — so it derives the whole |F|-vs-rounds curve from one fit. That holds only without bagging or early stopping. `tests/test_harness_properties.py` asserts it against independent fits; if you add bagging to a convergence experiment, that assumption (and E1) breaks.

## Notebooks

`notebooks/tree_test.ipynb` is the real-data track: MSLR-WEB10K learning-to-rank (`lambdarank`/NDCG@10) loaded from `~/Downloads/MSLR-WEB10K/Fold1`, sweeping `unused_feature_penalty` over `np.linspace(1, 0.5, 6)`. Note that sweep covers only the range where the penalty does almost nothing, and sets no scope, so it inherits the non-converging `tree` default — it predates both findings and needs revisiting.

`notebooks/tree_test_redundant_blocks.ipynb` re-hosts an **old copy** of the `run_milestones.py` code inline. It is now well out of date (pre-scope, pre-baselines, single-seed) and its results should not be trusted. Import from `trees_testing` rather than extending the inline copy.
