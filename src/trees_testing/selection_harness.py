"""
Experiment harness for evaluating a redundancy-penalizing LightGBM feature
selector driven by the `unused_feature_penalty` parameter.

Design notes
------------
* The selected feature set F is defined exactly as in the regularized-tree
  paper: a feature is "in F" iff it is used in >= 1 split anywhere in the model.
  In LightGBM that is `booster.feature_importance(importance_type='split') > 0`.

* The model-fitting step is *pluggable* via a `fit_fn`. The default points at
  your LightGBM build; tests inject a mock so the metrics/flow can be verified
  without LightGBM installed. This also lets you swap in any other tree lib.

* CONFOUNDER WARNING (LightGBM-specific): F accumulates across ALL boosting
  rounds, so |F| is driven by num_boost_round as much as by the penalty. This
  is not merely a nuisance to control for -- whether |F| *stops* growing is
  the property that decides if the method selects features at all. Under
  `unused_feature_penalty_scope="tree"` it never does; under `"ensemble"` it
  does. `selection_curve` measures exactly that, and is the primary check.

* PENALTY CONVENTION (settled, do not re-derive): the fork hard-checks
  0.0 <= unused_feature_penalty <= 1.0 with default 1.0. 1.0 == vanilla,
  SMALLER == stronger. This matches the paper's lambda. Passing None omits
  the parameter, which is identical to passing 1.0.

* A NOTE ON WHAT THESE METRICS CANNOT SEE: every metric here that scores a
  selection is computed on the union of features used across all trees. That
  union is blind to trees collapsing to a single feature each, which is the
  actual failure mode at small lambda. Always report `tree_diagnostics`
  alongside any selection metric.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# selection extraction
# --------------------------------------------------------------------------- #
def used_features(split_importance: np.ndarray) -> set:
    """Columns used in >= 1 split == the selected set F."""
    return set(int(i) for i in np.where(np.asarray(split_importance) > 0)[0])


# --------------------------------------------------------------------------- #
# metrics: plain relevant/irrelevant (Milestone 2, interaction main effects)
# --------------------------------------------------------------------------- #
def recovery_metrics(selected: set, relevant_cols, noise_cols) -> dict:
    """Precision/recall/F1 of recovering the planted relevant set, plus a
    direct count of noise features wrongly selected."""
    relevant = set(relevant_cols)
    noise = set(noise_cols)
    sel = set(selected)

    tp = len(sel & relevant)
    precision = tp / len(sel) if sel else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_selected": len(sel),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "noise_selected": len(sel & noise),
        "relevant_recovered": tp,
        "n_relevant": len(relevant),
    }


# --------------------------------------------------------------------------- #
# metrics: block-aware (Milestone 1 -- the headline test)
# --------------------------------------------------------------------------- #
def block_metrics(selected: set, blocks, noise_cols, driver_cols=None) -> dict:
    """Score selection against a redundant-block structure.

    The key quantities:
      block_coverage         -- fraction of blocks with >=1 member selected
                                (analogue of recall; want 1.0)
      members_per_covered    -- mean #selected members among COVERED blocks
                                (the redundancy number; ideal == 1.0, vanilla > 1)
      mean_members_per_block -- mean #selected members across ALL blocks
      noise_selected         -- noise features wrongly chosen (want 0)
      n_selected             -- total |F|  (ideal == n_blocks)

    Pass `driver_cols` for `drivers_chosen`: of the blocks covered, how many were
    represented by the TRUE driver rather than one of its correlated copies.
    Structure and quality are different things -- a selector can score a perfect
    members_per_covered == 1.0 while consistently picking a noisy proxy over the
    feature that actually generates y, which costs real accuracy against an
    oracle and is invisible to every other number here.
    """
    sel = set(selected)
    noise = set(noise_cols)

    covered = 0
    members_in_covered = []
    members_all = []
    for block in blocks:
        k = len(sel & set(block))
        members_all.append(k)
        if k > 0:
            covered += 1
            members_in_covered.append(k)

    out = {
        "n_blocks": len(blocks),
        "n_selected": len(sel),
        "block_coverage": covered / len(blocks) if blocks else 0.0,
        "members_per_covered": float(np.mean(members_in_covered)) if members_in_covered else 0.0,
        "mean_members_per_block": float(np.mean(members_all)) if members_all else 0.0,
        "noise_selected": len(sel & noise),
    }
    if driver_cols is not None:
        drivers = set(driver_cols)
        out["drivers_chosen"] = len(sel & drivers)
        out["driver_rate"] = (len(sel & drivers) / covered) if covered else 0.0
    return out


# --------------------------------------------------------------------------- #
# metrics: interaction integrity (XOR probe)
# --------------------------------------------------------------------------- #
def interaction_metrics(selected: set, xor_pairs, main_cols, noise_cols) -> dict:
    """Did selection keep BOTH halves of every interaction pair?

    A pair is 'intact' only if both members are selected; 'broken' if exactly
    one is (worst case -- you've kept a feature that is useless without its
    partner); 'dropped' if neither is.
    """
    sel = set(selected)
    intact = broken = dropped = 0
    for pair in xor_pairs:
        k = len(sel & set(pair))
        if k == len(pair):
            intact += 1
        elif k == 0:
            dropped += 1
        else:
            broken += 1
    return {
        "n_pairs": len(xor_pairs),
        "pairs_intact": intact,
        "pairs_broken": broken,     # the dangerous outcome
        "pairs_dropped": dropped,
        "main_effects_kept": len(sel & set(main_cols)),
        "n_main_effects": len(main_cols),
        "noise_selected": len(sel & set(noise_cols)),
    }


# --------------------------------------------------------------------------- #
# pluggable model fitting
# --------------------------------------------------------------------------- #
def _default_lightgbm_fit(X, y, penalty, param_name, task,
                          num_boost_round, extra_params, seed):
    """Fit one LightGBM model and return its per-feature split importance.

    Requires your LightGBM build that understands `param_name`. Held here behind
    a function so the rest of the harness has no hard LightGBM dependency.
    """
    import lightgbm as lgb

    if task == "classification":
        objective = "binary" if len(np.unique(y)) == 2 else "multiclass"
    else:
        objective = "regression"

    params = {
        "objective": objective,
        "verbosity": -1,
        "seed": seed,
        # NOTE: leave feature_fraction at default (1.0) for the core tests so
        # the penalty is the only thing restricting features. Milestone 4
        # toggles it on purpose.
    }
    if objective == "multiclass":
        params["num_class"] = int(len(np.unique(y)))
    if penalty is not None:
        params[param_name] = penalty
    params.update(extra_params or {})

    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    imp = booster.feature_importance(importance_type="split")
    return np.asarray(imp), booster


def fit_and_select(
    X, y,
    penalty=None,
    *,
    param_name="unused_feature_penalty",
    task="classification",
    num_boost_round=100,
    extra_params=None,
    seed=0,
    fit_fn=None,
):
    """Fit a model and return (selected_set, split_importance, fitted_object).

    `fit_fn(X, y, penalty, param_name, task, num_boost_round, extra_params, seed)`
    must return (split_importance_array, fitted_object). Defaults to LightGBM.
    """
    fit_fn = fit_fn or _default_lightgbm_fit
    imp, obj = fit_fn(X, y, penalty, param_name, task,
                      num_boost_round, extra_params, seed)
    return used_features(imp), np.asarray(imp), obj


# --------------------------------------------------------------------------- #
# confounder map: |F| over (penalty x rounds)
# --------------------------------------------------------------------------- #
def penalty_round_map(X, y, penalties, rounds_grid, *, task="classification",
                      param_name="unused_feature_penalty", seed=0, fit_fn=None):
    """Return a dict {(penalty, n_rounds): |F|} so you can SEE that round count
    moves the selected-feature count and must be held fixed when comparing."""
    out = {}
    for p in penalties:
        for r in rounds_grid:
            sel, _, _ = fit_and_select(
                X, y, penalty=p, param_name=param_name, task=task,
                num_boost_round=r, seed=seed, fit_fn=fit_fn)
            out[(p, r)] = len(sel)
    return out


# --------------------------------------------------------------------------- #
# matched-cardinality baselines (the "isn't this just feature_fraction?" test)
# --------------------------------------------------------------------------- #
def evaluate_subset(X_tr, y_tr, X_te, y_te, feature_subset, *, task="classification",
                    num_boost_round=100, seed=0, fit_fn=None):
    """Train a fresh model on ONLY `feature_subset` and score on the test split.

    Leakage-safe by construction: caller passes already-split train/test, and
    the subset is applied to both. Returns accuracy (classification) or R^2-ish
    negative-MSE proxy (regression). Uses the same fit_fn plumbing.
    """
    cols = sorted(feature_subset)
    if not cols:
        return float("nan")
    fit_fn = fit_fn or _default_lightgbm_fit

    # reuse the fit path purely to train; then predict via the returned object
    _, booster = fit_fn(X_tr[:, cols], y_tr, None, "unused_feature_penalty",
                        task, num_boost_round, None, seed)
    pred = booster.predict(X_te[:, cols])
    if task == "classification":
        if pred.ndim == 1:               # binary -> prob of class 1
            yhat = (pred >= 0.5).astype(int)
        else:                            # multiclass -> argmax
            yhat = pred.argmax(axis=1)
        return float(np.mean(yhat == y_te))
    else:
        mse = float(np.mean((pred - y_te) ** 2))
        var = float(np.var(y_te)) + 1e-12
        return 1.0 - mse / var           # R^2


def random_subset_baseline(k, n_features, n_repeats=20, rng=None):
    """Yield `n_repeats` random feature subsets of size k -- the sanity baseline.
    If your selected set of size k can't beat a random size-k set, the selection
    logic isn't earning its keep."""
    rng = rng or np.random.default_rng(0)
    for _ in range(n_repeats):
        yield set(int(i) for i in rng.choice(n_features, size=k, replace=False))


# --------------------------------------------------------------------------- #
# structural diagnostics -- what the union-of-features metrics cannot see
# --------------------------------------------------------------------------- #
def _tree_split_features(node, out):
    """Collect split feature indices from one dumped tree_structure."""
    if "split_feature" in node:
        out.add(int(node["split_feature"]))
        _tree_split_features(node["left_child"], out)
        _tree_split_features(node["right_child"], out)


def per_tree_features(booster) -> list:
    """List of sets: the features split on in each individual tree.

    `dump_model` reports `split_feature` in the original (real) column index
    space, the same space `feature_importance` and our ground truth use.
    """
    dumped = booster.dump_model()
    trees = []
    for info in dumped["tree_info"]:
        feats = set()
        _tree_split_features(info["tree_structure"], feats)
        trees.append(feats)
    return trees


def tree_diagnostics(booster) -> dict:
    """Per-tree structure. Catches the failure that |F| is blind to.

    At small lambda the penalty can drive every tree down to ONE feature: once
    the first split lands, nothing else clears the multiplicative bar. The model
    then has no interactions at all, yet the union of features across trees can
    still look like a healthy, well-chosen set. `mean_features_per_tree` near
    1.0 means the trees are degenerate regardless of how good |F| looks.
    """
    trees = per_tree_features(booster)
    sizes = np.array([len(t) for t in trees]) if trees else np.array([0])
    return {
        "n_trees": len(trees),
        "mean_features_per_tree": float(sizes.mean()),
        "min_features_per_tree": int(sizes.min()),
        "max_features_per_tree": int(sizes.max()),
        "degenerate": bool(sizes.mean() < 1.5),   # trees are effectively stumps
    }


# --------------------------------------------------------------------------- #
# the primary check: does |F| converge?
# --------------------------------------------------------------------------- #
def selection_curve(booster) -> np.ndarray:
    """|F| after each successive tree -- i.e. |F| as a function of num_boost_round.

    Exact and cheap: boosting builds tree t from trees 1..t-1 only, so the first
    r trees of an R-round model ARE the model you would get by training for r
    rounds (given fixed seed, no bagging, no early stopping). One fit therefore
    yields the whole curve, instead of one fit per grid point.
    """
    seen, curve = set(), []
    for feats in per_tree_features(booster):
        seen |= feats
        curve.append(len(seen))
    return np.asarray(curve, dtype=int)


def convergence_report(curve, n_features=None, tail_frac=0.5) -> dict:
    """Has |F| stopped growing, and did it settle anywhere useful?

    This is THE property that separates "a feature selector" from "a thing that
    slows down how fast features get used". If |F| is still climbing at the end
    of training there is no selected feature set -- only an artifact of where
    you stopped counting.

    `tail_frac=0.5` asks: did |F| grow at all over the final half of training?

    Pass `n_features` to distinguish the two ways of not growing. Stopping
    because every feature is now in F is not selection, it is saturation, and
    reporting it as "converged" flatters a model that selected nothing --
    vanilla LightGBM converges in exactly that sense. `selected` is the flag
    that means converged AND to a proper subset.
    """
    curve = np.asarray(curve)
    if curve.size == 0:
        return {"final": 0, "converged": False, "saturated": False,
                "selected": False, "tail_growth": 0, "plateau_at": 0}
    tail_start = int(curve.size * (1.0 - tail_frac))
    tail_growth = int(curve[-1] - curve[tail_start])
    final = int(curve[-1])
    converged = tail_growth == 0
    saturated = bool(n_features is not None and final >= n_features)
    return {
        "final": final,
        "converged": converged,
        "saturated": saturated,
        "selected": bool(converged and not saturated),
        "tail_growth": tail_growth,
        # first round at which |F| reached its final value
        "plateau_at": int(np.argmax(curve >= final)) + 1,
    }


# --------------------------------------------------------------------------- #
# competing selectors -- the "does this earn its keep?" comparison set
# --------------------------------------------------------------------------- #
def select_by_importance(X, y, k, *, importance_type="gain", task="classification",
                         num_boost_round=200, seed=0):
    """Top-k features of a STOCK LightGBM model, by importance.

    This is the baseline that actually matters. It costs one ordinary training
    run and no C++ changes. If the regularized selector cannot beat it at
    matched k, the fork is not buying anything.

    Note it has no notion of redundancy: among a block of correlated copies,
    importance is split across the members rather than concentrated on one, so
    this is exactly where a redundancy-aware method should be able to win.
    """
    import lightgbm as lgb

    objective = ("binary" if len(np.unique(y)) == 2 else "multiclass") \
        if task == "classification" else "regression"
    params = {"objective": objective, "verbosity": -1, "seed": seed}
    if objective == "multiclass":
        params["num_class"] = int(len(np.unique(y)))
    booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_boost_round)
    imp = np.asarray(booster.feature_importance(importance_type=importance_type))
    return set(int(i) for i in np.argsort(-imp)[:k])


def select_by_l1(X, y, k, *, task="classification", seed=0):
    """Top-k by |coefficient| of an L1-penalized linear model.

    The standard non-tree feature selector, and a fair external competitor.
    Handles redundancy by construction (L1 picks one of a correlated group),
    but cannot see interactions or nonlinearity.
    """
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    if task == "classification":
        # sklearn 1.8 deprecated penalty="l1" in favour of l1_ratio=1; support both
        # so this keeps working either side of that release.
        try:
            model = LogisticRegression(l1_ratio=1, solver="liblinear", C=0.1,
                                       random_state=seed, max_iter=2000)
            model.fit(Xs, y)
        except TypeError:
            model = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                       random_state=seed, max_iter=2000)
            model.fit(Xs, y)
        coef = np.abs(model.coef_).max(axis=0)
    else:
        model = Lasso(alpha=0.01, random_state=seed, max_iter=5000)
        model.fit(Xs, y)
        coef = np.abs(model.coef_)
    return set(int(i) for i in np.argsort(-coef)[:k])


# --------------------------------------------------------------------------- #
# scoring: never report a selection without its downstream cost
# --------------------------------------------------------------------------- #
def score_subset(X_tr, y_tr, X_te, y_te, feature_subset, *, task="classification",
                 num_boost_round=200, seed=0):
    """Train a stock model on ONLY `feature_subset`; score on the held-out split.

    Returns accuracy AND AUC for binary tasks. AUC is the one to compare on:
    accuracy is coarse and can hide real differences between selections that
    a threshold-free measure exposes.

    Leakage-safe by construction -- the caller passes an already-made split and
    the subset is applied to both halves. Deliberately uses a STOCK model (no
    penalty) so this measures the quality of the SELECTION, not of the
    regularized fit.
    """
    import lightgbm as lgb

    cols = sorted(feature_subset)
    if not cols:
        return {"accuracy": float("nan"), "auc": float("nan"), "r2": float("nan")}

    objective = ("binary" if len(np.unique(y_tr)) == 2 else "multiclass") \
        if task == "classification" else "regression"
    params = {"objective": objective, "verbosity": -1, "seed": seed}
    if objective == "multiclass":
        params["num_class"] = int(len(np.unique(y_tr)))
    booster = lgb.train(params, lgb.Dataset(X_tr[:, cols], label=y_tr),
                        num_boost_round=num_boost_round)
    pred = booster.predict(X_te[:, cols])

    if task != "classification":
        mse = float(np.mean((pred - y_te) ** 2))
        var = float(np.var(y_te)) + 1e-12
        return {"accuracy": float("nan"), "auc": float("nan"), "r2": 1.0 - mse / var}

    if pred.ndim == 1:
        yhat = (pred >= 0.5).astype(int)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_te, pred))
    else:
        yhat = pred.argmax(axis=1)
        auc = float("nan")
    return {"accuracy": float(np.mean(yhat == y_te)), "auc": auc, "r2": float("nan")}


# --------------------------------------------------------------------------- #
# replication: one seed is an anecdote
# --------------------------------------------------------------------------- #
def aggregate(rows) -> dict:
    """Collapse a list of per-seed metric dicts into {key: (mean, sd)}.

    Everything reported should go through this. A single-seed number cannot be
    told apart from noise, and several of these metrics are visibly seed-
    sensitive.
    """
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float, bool))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = (float(arr.mean()), float(arr.std()))
    return out


def fmt_agg(agg, keys, prec=2) -> str:
    """Render selected keys of an `aggregate` result as 'mean±sd' columns."""
    parts = []
    for k in keys:
        if k not in agg:
            continue
        m, s = agg[k]
        parts.append(f"{m:.{prec}f}±{s:.{prec}f}")
    return "  ".join(parts)
