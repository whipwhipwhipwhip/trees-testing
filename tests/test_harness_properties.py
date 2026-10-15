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

import math

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


# --------------------------------------------------------------------------- #
# GRRF: the guided per-feature penalty
# --------------------------------------------------------------------------- #
def test_grrf_gamma_zero_is_plain_regularized(blocks):
    """gamma=0 must reduce exactly to the unguided penalty regardless of guide.

    lambda_i = (1-gamma)*lambda_0 + gamma*Imp'_i collapses to lambda_0 at
    gamma=0, so a guide passed with gamma=0 must have no effect at all. If this
    drifts, every gamma>0 result is measured against the wrong baseline.
    """
    X, y, _ = blocks
    guide = H.importance_guide(X, y, task="classification", num_boost_round=50, seed=0)
    plain, _, _ = H.fit_and_select(X, y, penalty=0.05, task="classification",
                                   num_boost_round=100, seed=0, extra_params=SCOPE)
    guided0, _, _ = H.fit_and_select(X, y, penalty=0.05, task="classification",
                                     num_boost_round=100, seed=0,
                                     extra_params=H.grrf_params(guide, 0.0))
    assert plain == guided0


def test_grrf_exempts_the_feature_the_guide_rates_highest(blocks):
    """At gamma=1, lambda_i == Imp'_i, so the top-rated feature has lambda 1.0
    (no penalty at all) and must be selected. This is the mechanism that stops an
    arbitrary early split from deciding which features become penalty-exempt."""
    X, y, _ = blocks
    guide = H.importance_guide(X, y, task="classification", num_boost_round=200, seed=0)
    best_feature = int(np.argmax(guide))
    sel, _, _ = H.fit_and_select(X, y, penalty=0.01, task="classification",
                                 num_boost_round=200, seed=0,
                                 extra_params=H.grrf_params(guide, 1.0))
    assert best_feature in sel


def test_grrf_guide_of_wrong_length_fails_loudly(blocks):
    """A guide that silently mismatched the feature count would misalign every
    penalty -- feature i penalized by feature j's score."""
    X, y, _ = blocks
    with pytest.raises(Exception, match="unused_feature_penalty_guide"):
        H.fit_and_select(X, y, penalty=0.1, task="classification",
                         num_boost_round=5, seed=0,
                         extra_params=H.grrf_params(np.ones(X.shape[1] - 1), 0.5))


def test_importance_guide_is_normalised_to_unit_max(blocks):
    X, y, _ = blocks
    g = H.importance_guide(X, y, task="classification", num_boost_round=50, seed=0)
    assert g.shape == (X.shape[1],)
    assert g.min() >= 0.0 and abs(g.max() - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# candidate rule: the second mechanism of the paper's Algorithm 1
# --------------------------------------------------------------------------- #
# Whole-tree comparisons need fits that are reproducible run to run; LightGBM
# otherwise picks row- or column-wise histograms by timing them.
DET = {"deterministic": True, "force_row_wise": True, "num_threads": 1}


def _fit_trees(X, y, penalty, rounds, seed=0, **params):
    """Full tree structures, compared whole: two models can share |F| while
    splitting on different features in different places."""
    _, _, b = H.fit_and_select(X, y, penalty=penalty, task="classification",
                               num_boost_round=rounds, seed=seed,
                               extra_params={**DET, **SCOPE, **params})
    return b.dump_model()["tree_info"]


@pytest.mark.parametrize("penalty", [1.0, 0.05])
def test_candidates_at_least_m_is_identical_to_rule_off(blocks, penalty):
    """Drawing every challenger masks nothing, so K >= M must reproduce the rule
    switched off exactly. Also guards the clamp: Random::Sample returns NO indices
    when asked for more than it has, which would silently admit no challengers."""
    X, y, _ = blocks
    m = X.shape[1]
    off = _fit_trees(X, y, penalty, 60, unused_feature_penalty_candidates=-1)
    assert _fit_trees(X, y, penalty, 60, unused_feature_penalty_candidates=m) == off
    assert _fit_trees(X, y, penalty, 60, unused_feature_penalty_candidates=10 * m) == off


def test_candidates_zero_means_ceil_sqrt_m(blocks):
    """0 is documented as the paper's K = ceil(sqrt(M)). Also checks that it does
    something, since two no-ops would compare equal too."""
    X, y, _ = blocks
    k = math.ceil(math.sqrt(X.shape[1]))
    auto = _fit_trees(X, y, 0.05, 60, unused_feature_penalty_candidates=0)
    assert auto == _fit_trees(X, y, 0.05, 60, unused_feature_penalty_candidates=k)
    assert auto != _fit_trees(X, y, 0.05, 60, unused_feature_penalty_candidates=-1)


def test_challengers_are_drawn_at_random(blocks):
    """With F empty and K=1, the first split has one candidate and it is a random
    draw, so the root feature must vary with the seed. With the rule off the root
    is always the single best feature. If this fails, either no draw is happening
    or it is not seeded from `seed`."""
    X, y, _ = blocks

    def roots(**params):
        return {_fit_trees(X, y, None, 1, seed=s, num_leaves=2, **params)[0]
                ["tree_structure"]["split_feature"] for s in range(40)}

    assert len(roots()) == 1
    assert len(roots(unused_feature_penalty_candidates=1)) >= 10


def test_incumbents_are_never_masked():
    """The reason the rule is not just feature_fraction_bynode.

    One strong feature among 199 noise features, fitted with stumps and no gain
    penalty. Once the strong feature is in F the rule must offer it at every node,
    so it should win essentially every later stump. A symmetric
    feature_fraction_bynode at the same average rate masks it most of the time,
    so there it wins only a small share -- that is the approximation this rule
    exists to replace, and why a constant fraction cannot stand in for it.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 200))
    y = (X[:, 0] + 0.3 * rng.normal(size=3000) > 0).astype(int)
    k = math.ceil(math.sqrt(X.shape[1]))

    def share_after_admission(**params):
        roots = [t["tree_structure"].get("split_feature") for t in
                 _fit_trees(X, y, None, 150, num_leaves=2, learning_rate=0.02, **params)]
        assert 0 in roots, "strong feature never admitted, so this test has no power"
        after = roots[roots.index(0):]
        return sum(f == 0 for f in after) / len(after)

    assert share_after_admission(unused_feature_penalty_candidates=k) > 0.95
    assert share_after_admission(feature_fraction_bynode=k / X.shape[1]) < 0.5


@pytest.mark.parametrize("scope", ["ensemble", "tree"])
def test_candidate_rule_keeps_boosting_prefix_consistent(blocks, scope):
    """selection_curve's assumption, re-checked with the rule on. The rule draws at
    every node; if its generator were reseeded or advanced by anything that
    depends on the total round count, the first r trees of a long run would stop
    matching a short run, and every convergence curve would be wrong."""
    X, y, _ = blocks
    params = dict(unused_feature_penalty_scope=scope, unused_feature_penalty_candidates=0)
    assert _fit_trees(X, y, 0.3, 80, **params)[:30] == _fit_trees(X, y, 0.3, 30, **params)


def test_candidate_rule_owns_per_node_sampling(blocks):
    """With the rule on, feature_fraction_bynode is documented as ignored. If both
    applied, the incumbent guarantee would be broken by the symmetric sampler."""
    X, y, _ = blocks
    on = dict(unused_feature_penalty_candidates=0)
    assert (_fit_trees(X, y, 0.3, 60, feature_fraction_bynode=0.3, **on)
            == _fit_trees(X, y, 0.3, 60, **on))


def test_candidate_rule_survives_per_iteration_config_resets(blocks):
    """reset_parameter drives ResetConfig() whenever a value changes. That path
    restores the column sampler's config fraction, so it must re-apply the bynode
    override, and it must not clear F or reseed the challenger draw.

    Alternating feature_fraction_bynode forces a reset on every iteration, and
    since the rule owns bynode the model must not change. The same callback
    without the rule shows it has teeth, so this cannot pass vacuously.
    """
    import lightgbm as lgb
    X, y, _ = blocks
    base = {"objective": "binary", "verbosity": -1, "seed": 0, **DET, **SCOPE,
            "unused_feature_penalty": 0.3, "feature_fraction_bynode": 0.5}

    def trees(callbacks=None, **params):
        b = lgb.train({**base, **params}, lgb.Dataset(X, y), num_boost_round=50,
                      callbacks=callbacks)
        return b.dump_model()["tree_info"]

    def flip():
        return [lgb.reset_parameter(feature_fraction_bynode=[0.5, 0.4] * 25)]

    assert trees(flip()) != trees(), "callback should change the model with the rule off"
    on = dict(unused_feature_penalty_candidates=0)
    assert trees(flip(), **on) == trees(**on)


def test_milestone_params_keep_scope_whenever_f_is_read():
    """run_milestones._params used to return {} at lambda=1.0, which was harmless
    while only the penalty read F. The candidate rule reads F too, so the
    rule-only arm (lambda=1.0, rule on) silently ran under `tree` scope."""
    import trees_testing.run_milestones as M
    assert M._params(None) == {} and M._params(1.0) == {}
    assert M._params(1.0, candidates=-1) == {}
    assert M._params(0.1) == {M.SCOPE_PARAM: M.SCOPE}
    assert M._params(1.0, candidates=23) == {M.SCOPE_PARAM: M.SCOPE, M.CANDIDATES_PARAM: 23}
    assert M._params(0.1, candidates=0) == {M.SCOPE_PARAM: M.SCOPE, M.CANDIDATES_PARAM: 0}


def test_candidates_below_minus_one_fails_loudly(blocks):
    X, y, _ = blocks
    with pytest.raises(Exception, match="unused_feature_penalty_candidates"):
        H.fit_and_select(X, y, penalty=0.1, task="classification",
                         num_boost_round=5, seed=0,
                         extra_params={"unused_feature_penalty_candidates": -2})


# --------------------------------------------------------------------------- #
# real-data ground truth (network; skipped if unavailable)
# --------------------------------------------------------------------------- #
def test_madelon_ground_truth_is_recoverable():
    """Madelon's published composition is 20 real + 480 probes, but the indices
    were never released. We recover them from the correlation structure, so this
    asserts that recovery agrees with the published counts AND with a completely
    independent signal (gain importance). If those two ever disagree, the truth
    set is unsafe and every E6 number scored against it is meaningless.

    Also pins the block structure: single-linkage chains these into one giant
    block, which silently makes block_coverage 1.00 for every selector.
    """
    import trees_testing.real_data as rd
    pytest.importorskip("sklearn.datasets")
    try:
        X, y, t = rd.load_madelon()
    except Exception as e:                        # offline / OpenML down
        pytest.skip(f"Madelon unavailable: {type(e).__name__}")

    assert X.shape == (2600, 500)
    assert len(t["relevant_cols"]) == 20
    assert len(t["noise_cols"]) == 480

    # Independent route to the same set: top-20 by GAIN importance. (Split
    # importance is a much noisier ranking here and agrees on only ~17/20 --
    # use the same measure the exploration and E6 baselines use.)
    top20 = H.select_by_importance(X, y, 20, importance_type="gain",
                                   task="classification", num_boost_round=300, seed=0)
    overlap = len(top20 & set(t["relevant_cols"]))
    assert overlap >= 18, (
        f"correlation-derived truth and importance ranking agree on only "
        f"{overlap}/20 features; the ground truth is not trustworthy")

    # block structure must actually discriminate, not collapse to one block
    sizes = sorted(len(b) for b in t["blocks"])
    assert t["n_blocks"] >= 4, f"blocks collapsed: {sizes}"
    assert max(sizes) <= 8, f"a block swallowed most of the real features: {sizes}"
    assert sum(sizes) == 20
