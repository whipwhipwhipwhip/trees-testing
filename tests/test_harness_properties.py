"""
Tests for the evaluation harness itself.

The experiments in `run_milestones` are only worth reading if the machinery
producing them is sound, so these check the harness's own assumptions -- most
importantly the prefix property that makes `selection_curve` cheap, which is an
assumption about how LightGBM builds trees and would silently corrupt every
convergence result in E1 if it were wrong.

    uv run pytest tests/ -v

Note `src/trees_testing/test_harness.py` is NOT pytest despite the name -- it is
a script of module-level asserts. Run it with
`uv run python -m trees_testing.test_harness`.
"""

from __future__ import annotations

import numpy as np
import pytest

import trees_testing.synthetic as syn
import trees_testing.selection_harness as H

SCOPE = {"unused_feature_penalty_scope": "ensemble"}


@pytest.fixture(scope="module")
def blocks():
    return syn.make_redundant_blocks(
        n_samples=2000, block_sizes=(4, 4, 3, 1, 1), n_noise=20,
        corr=0.95, task="classification", random_state=0)


# --------------------------------------------------------------------------- #
# the assumption E1 rests on
# --------------------------------------------------------------------------- #
def test_selection_curve_matches_independent_fits(blocks):
    """The first r trees of an R-round model must equal an r-round model.

    `selection_curve` derives the whole |F|-vs-rounds curve from ONE fit on this
    basis. If boosting were not prefix-consistent here (bagging, early stopping,
    any per-run randomness that depends on the round count) the curve would be
    wrong and every convergence verdict with it. Checked against fits done the
    slow, obviously-correct way.
    """
    X, y, _ = blocks
    _, _, booster = H.fit_and_select(X, y, penalty=0.05, task="classification",
                                     num_boost_round=200, seed=0, extra_params=SCOPE)
    curve = H.selection_curve(booster)

    for r in (1, 5, 25, 100, 200):
        sel, _, _ = H.fit_and_select(X, y, penalty=0.05, task="classification",
                                     num_boost_round=r, seed=0, extra_params=SCOPE)
        assert curve[r - 1] == len(sel), (
            f"curve says |F|={curve[r-1]} after {r} rounds, "
            f"an independent {r}-round fit says {len(sel)}")


def test_per_tree_features_use_real_column_indices(blocks):
    """dump_model's split_feature must live in the same index space as
    feature_importance, or every ground-truth comparison is silently misaligned."""
    X, y, _ = blocks
    sel, imp, booster = H.fit_and_select(X, y, penalty=0.05, task="classification",
                                         num_boost_round=50, seed=0, extra_params=SCOPE)
    union = set().union(*H.per_tree_features(booster))
    assert union == sel == H.used_features(imp)


# --------------------------------------------------------------------------- #
# convergence_report: the three outcomes must be distinguishable
# --------------------------------------------------------------------------- #
def test_convergence_report_distinguishes_selected_from_saturated():
    """Stopping at a proper subset is selection; stopping because you ran out of
    features is not. Conflating them would make vanilla look like a selector."""
    selected = H.convergence_report([1, 3, 5, 5, 5, 5], n_features=50)
    assert selected["converged"] and selected["selected"] and not selected["saturated"]

    saturated = H.convergence_report([10, 30, 50, 50, 50, 50], n_features=50)
    assert saturated["converged"] and saturated["saturated"] and not saturated["selected"]

    growing = H.convergence_report([1, 3, 5, 9, 14, 20], n_features=50)
    assert not growing["converged"] and not growing["selected"]


def test_convergence_report_plateau_location():
    rep = H.convergence_report([1, 2, 4, 4, 4, 4, 4, 4], n_features=50)
    assert rep["plateau_at"] == 3
    assert rep["final"] == 4


# --------------------------------------------------------------------------- #
# tree_diagnostics: catches what |F| is blind to
# --------------------------------------------------------------------------- #
def test_tree_diagnostics_flags_degenerate_trees(blocks):
    """At tiny lambda under tree scope, trees collapse to one feature each. |F|
    cannot see this -- the union across trees still looks like a sensible set --
    so tree_diagnostics is the only thing standing between us and reporting a
    model with no interactions as a good result."""
    X, y, _ = blocks
    _, _, degenerate = H.fit_and_select(
        X, y, penalty=0.005, task="classification", num_boost_round=100, seed=0,
        extra_params={"unused_feature_penalty_scope": "tree"})
    d = H.tree_diagnostics(degenerate)
    assert d["mean_features_per_tree"] < 1.5 and d["degenerate"]

    _, _, healthy = H.fit_and_select(X, y, penalty=1.0, task="classification",
                                     num_boost_round=100, seed=0)
    assert not H.tree_diagnostics(healthy)["degenerate"]


# --------------------------------------------------------------------------- #
# the fork's core claim, as an executable assertion
# --------------------------------------------------------------------------- #
def test_ensemble_scope_converges_and_tree_scope_does_not(blocks):
    """The reason the scope parameter exists. Under `tree` scope |F| keeps
    growing with the round count, so there is no selected set; under `ensemble`
    it settles."""
    X, y, _ = blocks
    n_features = X.shape[1]

    def report(scope, rounds=600):
        _, _, b = H.fit_and_select(
            X, y, penalty=0.01, task="classification", num_boost_round=rounds,
            seed=0, extra_params={"unused_feature_penalty_scope": scope})
        return H.convergence_report(H.selection_curve(b), n_features=n_features)

    assert report("ensemble")["selected"], "ensemble scope should reach a stable subset"
    assert not report("tree")["selected"], "tree scope should NOT reach a stable subset"


def test_ensemble_scope_recovers_one_feature_per_block(blocks):
    """The headline selection claim: one member of each correlated block, no
    noise. This is the thing the whole fork exists to do."""
    X, y, t = blocks
    sel, _, _ = H.fit_and_select(X, y, penalty=0.01, task="classification",
                                 num_boost_round=400, seed=0, extra_params=SCOPE)
    m = H.block_metrics(sel, t["blocks"], t["noise_cols"])
    assert m["block_coverage"] == 1.0
    assert m["members_per_covered"] == 1.0
    assert m["noise_selected"] == 0
    assert m["n_selected"] == t["n_blocks"]


def test_penalty_of_one_is_identical_to_omitting_it(blocks):
    """lambda=1.0 is documented as 'no penalty'. If it differed from stock, the
    vanilla baseline in every experiment would be wrong."""
    X, y, _ = blocks
    a, _, _ = H.fit_and_select(X, y, penalty=None, task="classification",
                               num_boost_round=50, seed=0)
    b, _, _ = H.fit_and_select(X, y, penalty=1.0, task="classification",
                               num_boost_round=50, seed=0, extra_params=SCOPE)
    assert a == b


def test_unknown_scope_fails_loudly(blocks):
    """A typo must not silently fall back to the default scope."""
    X, y, _ = blocks
    with pytest.raises(Exception, match="unused_feature_penalty_scope"):
        H.fit_and_select(X, y, penalty=0.1, task="classification",
                         num_boost_round=5, seed=0,
                         extra_params={"unused_feature_penalty_scope": "globl"})


# --------------------------------------------------------------------------- #
# the XOR probe must be able to fail
# --------------------------------------------------------------------------- #
def test_xor_probe_has_power_with_current_defaults():
    """Regression test on the generator fix.

    Under the original settings (3 main effects at the same weight as the XOR
    term, xor_weight 1.5) the full-size probe measured an accuracy cost of just
    0.008 for destroying interaction pairs -- so `pairs_broken` was effectively
    unfalsifiable. The defaults now drop main effects and raise xor_weight.

    Asserted as a ratio rather than an absolute for the old settings: at this
    smaller size they retain a modest gap (~0.10 AUC), so the meaningful claim
    is that the new defaults are decisively more sensitive, not that the old
    ones are literally flat.
    """
    def power_for(**kw):
        X, y, t = syn.make_xor_interaction(n_samples=3000, n_xor_pairs=2,
                                           n_noise=15, random_state=0, **kw)
        tr = np.arange(0, len(y), 2)
        te = np.arange(1, len(y), 2)
        return syn.xor_probe_power(t, lambda cols: H.score_subset(
            X[tr], y[tr], X[te], y[te], cols, task="classification",
            num_boost_round=100, seed=0)["auc"])

    now = power_for()
    old = power_for(n_main_effects=3, xor_weight=1.5)
    assert now["has_power"], "current defaults must be able to detect a broken pair"
    assert now["gap"] > 2 * old["gap"], (
        f"current defaults ({now['gap']:+.3f}) should be far more sensitive than the "
        f"old ones ({old['gap']:+.3f}); if not, the generator lost its power")
    # A pair-broken model should be near chance once XOR is the only signal.
    assert now["broken"] < 0.65


# --------------------------------------------------------------------------- #
# scoring and baselines
# --------------------------------------------------------------------------- #
def test_score_subset_ranks_oracle_above_random(blocks):
    """Sanity floor: if a random subset scored as well as the true drivers, the
    scoring function could not tell any two selections apart."""
    X, y, t = blocks
    tr, te = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
    oracle = H.score_subset(X[tr], y[tr], X[te], y[te], t["driver_cols"],
                            task="classification", num_boost_round=100, seed=0)
    rand = H.score_subset(X[tr], y[tr], X[te], y[te],
                          next(H.random_subset_baseline(len(t["driver_cols"]),
                                                        X.shape[1], 1,
                                                        np.random.default_rng(3))),
                          task="classification", num_boost_round=100, seed=0)
    assert oracle["auc"] > rand["auc"]


@pytest.mark.parametrize("selector", ["gain", "split", "l1"])
def test_baselines_return_exactly_k_features(blocks, selector):
    """Matched cardinality is the entire basis of the E3 comparison; a baseline
    returning a different k would make the head-to-head meaningless."""
    X, y, _ = blocks
    k = 7
    if selector == "l1":
        sel = H.select_by_l1(X, y, k, task="classification", seed=0)
    else:
        sel = H.select_by_importance(X, y, k, importance_type=selector,
                                     task="classification", num_boost_round=50, seed=0)
    assert len(sel) == k
    assert all(0 <= c < X.shape[1] for c in sel)


def test_aggregate_reports_mean_and_sd():
    agg = H.aggregate([{"a": 1.0, "b": 2}, {"a": 3.0, "b": 4}])
    assert agg["a"] == (2.0, 1.0)
    assert agg["b"] == (3.0, 1.0)
