"""
Evaluation of the `unused_feature_penalty` LightGBM fork against synthetic data
with known ground truth.

    uv run python -m trees_testing.run_milestones            # everything
    uv run python -m trees_testing.run_milestones --quick    # fewer seeds/rounds
    uv run python -m trees_testing.run_milestones E3         # one experiment

The question this has to answer is not "does the penalty change anything" -- it
demonstrably does -- but "is what it does worth a C++ fork". So the design rule
throughout is:

  1. Never report a selection without its downstream cost. A small |F| that
     loses accuracy is not a result.
  2. Never report a single seed. Several of these metrics are seed-sensitive
     enough that one draw is indistinguishable from noise.
  3. Always compare against the cheap alternative. Top-k by importance from a
     stock LightGBM model costs one ordinary training run. If the fork cannot
     beat that at matched |F|, it is not earning its keep, and E3 is built to
     say so plainly if that is the answer.
  4. Report structure, not just the union of features. |F| is computed across
     all trees and is blind to trees collapsing to one feature each.

Experiments
-----------
  E1  convergence     -- does |F| stop growing? (the property that makes this a
                         feature selector at all; tree scope fails it)
  E2  selection       -- redundant blocks: is the selected set the RIGHT set,
                         and what does it cost on held-out data?
  E3  justification   -- matched-|F| head-to-head vs stock importance, L1,
                         random, all-features and the oracle. The headline.
  E4  interactions    -- does the penalty destroy XOR pairs? (self-validating:
                         refuses to report if the probe has no power)
  E5  high-dimensional-- p > n, where feature selection actually matters
"""

from __future__ import annotations

import argparse
import numpy as np

import trees_testing.synthetic as syn
import trees_testing.selection_harness as H

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
PENALTY_PARAM = "unused_feature_penalty"
SCOPE_PARAM = "unused_feature_penalty_scope"

# Settled from the fork's own config: 0.0 <= lambda <= 1.0, default 1.0.
# 1.0 == vanilla, SMALLER == stronger. Matches the paper's lambda.
VANILLA_PENALTY = 1.0

# Log-spaced, because all the dynamic range lives near zero: on the block design
# nothing distinguishable happens between 1.0 and ~0.1. A linear sweep spends
# most of its points in the dead zone.
PENALTY_SWEEP = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

# "ensemble" is the paper's semantics and the only scope under which |F|
# converges. "tree" is the fork's default and is kept in E1 for contrast.
SCOPE = "ensemble"

NUM_BOOST_ROUND = 400      # HOLD FIXED across every compared model
N_SEEDS = 5
TASK = "classification"


def _params(penalty, scope=SCOPE):
    """extra_params for a regularized fit. None penalty => stock LightGBM."""
    if penalty is None or penalty == VANILLA_PENALTY:
        return {}
    return {SCOPE_PARAM: scope}


def _split(n, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[: n // 2], idx[n // 2:]


def _hdr(tag, title):
    print("\n" + "=" * 78)
    print(f"{tag} -- {title}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# E1: does |F| converge?
# --------------------------------------------------------------------------- #
def e1_convergence(rounds=None, n_seeds=None):
    rounds = rounds or max(NUM_BOOST_ROUND, 800)
    n_seeds = n_seeds or N_SEEDS
    _hdr("E1", "convergence: does |F| stop growing?")
    print("""  If |F| is still climbing when training ends, there is no selected feature
  set -- only an artifact of where you stopped counting. This is the property
  that separates a feature selector from a thing that merely slows features
  down. Measured as the growth in |F| over the FINAL HALF of training.
""")
    X, y, t = syn.make_redundant_blocks(
        n_samples=3000, block_sizes=(5, 5, 4, 1, 1), n_noise=30,
        corr=0.95, task=TASK, random_state=0)
    print(f"  design: {t['n_blocks']} blocks / {len(t['relevant_cols'])} relevant / "
          f"{len(t['noise_cols'])} noise / {t['n_features']} total.  ideal |F| = {t['n_blocks']}")
    print(f"  {rounds} rounds, {n_seeds} seeds\n")

    print(f"  {'scope':<9} {'lambda':>7} | {'|F| final':>11}  {'tail growth':>11}  "
          f"{'outcome':>10}  {'feats/tree':>10}")
    print("  " + "-" * 74)
    for scope in ("tree", "ensemble"):
        for pen in (1.0, 0.1, 0.01):
            rows = []
            for s in range(n_seeds):
                _, _, booster = H.fit_and_select(
                    X, y, penalty=pen, param_name=PENALTY_PARAM, task=TASK,
                    num_boost_round=rounds, seed=s,
                    extra_params=_params(pen, scope))
                rep = H.convergence_report(H.selection_curve(booster),
                                           n_features=t["n_features"])
                rep.update(H.tree_diagnostics(booster))
                rows.append(rep)
            a = H.aggregate(rows)
            # Three distinct outcomes, only one of which is feature selection.
            if a["selected"][0] == 1.0:
                outcome = "SELECTED"
            elif a["saturated"][0] == 1.0:
                outcome = "saturated"
            elif a["converged"][0] == 0.0:
                outcome = "growing"
            else:
                outcome = "mixed"
            print(f"  {scope:<9} {pen:>7} | {a['final'][0]:>6.1f}±{a['final'][1]:<4.1f} "
                  f"{a['tail_growth'][0]:>6.1f}±{a['tail_growth'][1]:<4.1f} "
                  f"{outcome:>10}  {a['mean_features_per_tree'][0]:>10.2f}")

    print("""
  READ: only 'SELECTED' is a feature selector -- |F| stopped growing AND landed
  on a proper subset. 'saturated' means it stopped only because every feature is
  now in F, which is what vanilla does and is not selection. 'growing' means |F|
  is still climbing, so any |F| you quote is really a statement about
  num_boost_round. Under `tree` scope no lambda reaches SELECTED.
  'feats/tree' near 1.0 is the separate failure: trees collapsed to single-
  feature stumps, so the model has no interactions left however good |F| looks.""")


# --------------------------------------------------------------------------- #
# E2: is the selected set the RIGHT set, and what does it cost?
# --------------------------------------------------------------------------- #
def e2_selection(n_seeds=None):
    n_seeds = n_seeds or N_SEEDS
    _hdr("E2", "selection quality on redundant blocks, with downstream cost")
    print("""  A perfect redundancy-aware selector keeps exactly one member of each
  correlated block and no noise: |F| == n_blocks, members/block == 1.00,
  noise == 0. Accuracy and AUC come from a STOCK model retrained on the
  selected columns only, so this scores the SELECTION, not the penalized fit.
""")
    rows_by_pen = {p: [] for p in PENALTY_SWEEP}
    oracle, allfeat = [], []

    for s in range(n_seeds):
        X, y, t = syn.make_redundant_blocks(
            n_samples=3000, block_sizes=(5, 5, 4, 1, 1), n_noise=30,
            corr=0.95, task=TASK, random_state=s)
        tr, te = _split(len(y), s)
        sc = lambda cols: H.score_subset(X[tr], y[tr], X[te], y[te], cols,
                                         task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s)
        oracle.append(sc(t["driver_cols"]))
        allfeat.append(sc(range(X.shape[1])))
        for pen in PENALTY_SWEEP:
            sel, _, booster = H.fit_and_select(
                X[tr], y[tr], penalty=pen, param_name=PENALTY_PARAM, task=TASK,
                num_boost_round=NUM_BOOST_ROUND, seed=s, extra_params=_params(pen))
            m = H.block_metrics(sel, t["blocks"], t["noise_cols"],
                                driver_cols=t["driver_cols"])
            m.update(sc(sel))
            m.update(H.tree_diagnostics(booster))
            rows_by_pen[pen].append(m)

    print(f"  {'lambda':>7} | {'|F|':>9}  {'members/blk':>11}  {'noise':>9}  "
          f"{'drivers/5':>9}  {'AUC':>11}  {'feats/tree':>10}")
    print("  " + "-" * 86)
    for pen in PENALTY_SWEEP:
        a = H.aggregate(rows_by_pen[pen])
        print(f"  {pen:>7} | {a['n_selected'][0]:>4.1f}±{a['n_selected'][1]:<4.1f} "
              f"{a['members_per_covered'][0]:>6.2f}±{a['members_per_covered'][1]:<4.2f} "
              f"{a['noise_selected'][0]:>4.1f}±{a['noise_selected'][1]:<4.1f} "
              f"{a['drivers_chosen'][0]:>9.1f} "
              f"{a['auc'][0]:>6.3f}±{a['auc'][1]:<4.3f} "
              f"{a['mean_features_per_tree'][0]:>10.2f}")
    ao, aa = H.aggregate(oracle), H.aggregate(allfeat)
    print(f"\n  reference   oracle (5 true drivers) AUC {ao['auc'][0]:.3f}±{ao['auc'][1]:.3f}"
          f"   all 46 features AUC {aa['auc'][0]:.3f}±{aa['auc'][1]:.3f}")
    print("""
  READ: the operating point you want has members/blk == 1.00 AND AUC level with
  the oracle. If AUC falls away before members/blk reaches 1.00, the penalty is
  discarding signal rather than redundancy, and there is no good lambda.
  'drivers/5' is the quality behind the structure: picking one member per block
  is not the same as picking the RIGHT one. Every block member correlates at
  0.95 with its driver, so a copy is a slightly noisy proxy -- if drivers/5 sits
  below 5, that gap explains any shortfall against the oracle, and no amount of
  tuning lambda will close it because the penalty has no way to tell a driver
  from its copy.""")


# --------------------------------------------------------------------------- #
# E3: the justification test
# --------------------------------------------------------------------------- #
def e3_justification(n_seeds=None):
    n_seeds = n_seeds or N_SEEDS
    _hdr("E3", "does this beat the cheap alternative? (matched |F|)")
    print("""  THE test. The regularized model picks k features; every competitor is then
  asked for its own best k, and all are scored by retraining a stock model on
  those k columns. Same k, same data, same split, same rounds -- the only
  difference is WHICH k features.

  'stock gain' is the one that matters: top-k by importance from an ordinary
  LightGBM run, no fork required. Beating random is not evidence of anything;
  beating stock gain is. Redundancy is where the fork should win, because
  importance gets divided among correlated copies rather than concentrated.
""")
    lam = 0.01
    rows = {n: [] for n in ("regularized", "stock gain", "stock split", "L1", "random", "all", "oracle")}
    ks, correctness = [], {"regularized": [], "stock gain": [], "L1": []}

    for s in range(n_seeds):
        X, y, t = syn.make_redundant_blocks(
            n_samples=3000, block_sizes=(5, 5, 4, 1, 1), n_noise=30,
            corr=0.95, task=TASK, random_state=s)
        tr, te = _split(len(y), s)
        sc = lambda cols: H.score_subset(X[tr], y[tr], X[te], y[te], cols,
                                         task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s)

        sel, _, _ = H.fit_and_select(
            X[tr], y[tr], penalty=lam, param_name=PENALTY_PARAM, task=TASK,
            num_boost_round=NUM_BOOST_ROUND, seed=s, extra_params=_params(lam))
        k = len(sel)
        ks.append(k)

        cand = {
            "regularized": sel,
            "stock gain": H.select_by_importance(X[tr], y[tr], k, importance_type="gain",
                                                 task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s),
            "stock split": H.select_by_importance(X[tr], y[tr], k, importance_type="split",
                                                  task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s),
            "L1": H.select_by_l1(X[tr], y[tr], k, task=TASK, seed=s),
            "random": next(H.random_subset_baseline(k, X.shape[1], 1,
                                                    np.random.default_rng(s))),
            "all": set(range(X.shape[1])),
            "oracle": set(t["driver_cols"]),
        }
        for name, cols in cand.items():
            rows[name].append(sc(cols))
            if name in correctness:
                correctness[name].append(H.block_metrics(cols, t["blocks"], t["noise_cols"]))

    print(f"  matched k = {np.mean(ks):.1f}±{np.std(ks):.1f}   ({n_seeds} seeds, lambda={lam})\n")
    print(f"  {'selector':<14} | {'AUC':>13}  {'accuracy':>13}  {'blocks hit':>10}  {'noise':>6}")
    print("  " + "-" * 74)
    for name in ("oracle", "regularized", "stock gain", "stock split", "L1", "random", "all"):
        a = H.aggregate(rows[name])
        extra = ""
        if name in correctness:
            c = H.aggregate(correctness[name])
            extra = f"{c['block_coverage'][0]:>10.2f}  {c['noise_selected'][0]:>6.1f}"
        print(f"  {name:<14} | {a['auc'][0]:>6.3f}±{a['auc'][1]:<6.3f} "
              f"{a['accuracy'][0]:>6.3f}±{a['accuracy'][1]:<6.3f}  {extra}")

    # Paired against EVERY competitor, not just the expected one. Same seed and
    # split on both sides of each difference, so the comparison is paired and the
    # SE is of the difference rather than of two independent means.
    reg = np.array([r["auc"] for r in rows["regularized"]])
    print(f"\n  paired per-seed AUC difference, regularized minus each competitor:")
    verdicts = {}
    for name in ("stock gain", "stock split", "L1"):
        d = reg - np.array([r["auc"] for r in rows[name]])
        stderr = d.std() / np.sqrt(n_seeds)
        sep = abs(d.mean()) > 2 * stderr
        verdicts[name] = (d.mean(), sep)
        mark = ("WINS" if d.mean() > 0 else "LOSES") if sep else "tie"
        print(f"    vs {name:<12} {d.mean():+.4f} (SE {stderr:.4f})  "
              f"won {int((d > 0).sum())}/{n_seeds}   -> {mark}")

    beaten = [n for n, (m, sep) in verdicts.items() if sep and m > 0]
    tied = [n for n, (m, sep) in verdicts.items() if not sep]
    lost = [n for n, (m, sep) in verdicts.items() if sep and m < 0]
    print()
    if lost:
        print(f"  => NOT JUSTIFIED here: beaten by {', '.join(lost)}, which cost no fork.")
    elif beaten and not tied:
        print(f"  => justified on this design: beats {', '.join(beaten)}.")
    elif beaten:
        print(f"  => partially justified: beats {', '.join(beaten)}, but only ties\n"
              f"     {', '.join(tied)}. A tie with a standard method means the fork is not\n"
              f"     yet buying anything THAT method cannot; find a design where it does\n"
              f"     (this one's signal is linear, which suits L1).")
    else:
        print("  => no separation from any competitor. Not justified on this design.")
    if n_seeds < 10:
        print(f"     (n_seeds={n_seeds} is thin; re-run with --seeds 20 before believing it)")


# --------------------------------------------------------------------------- #
# E4: interaction safety, with a power check
# --------------------------------------------------------------------------- #
def e4_interactions(n_seeds=None):
    n_seeds = n_seeds or N_SEEDS
    _hdr("E4", "interaction safety (XOR pairs)")
    print("""  Each XOR pair is two features individually uninformative but jointly
  determining. A penalty that judges features on marginal gain may drop the
  second half before the tree ever discovers the interaction.

  This probe first checks it CAN detect that. If breaking a pair costs no
  accuracy on this data, `pairs_broken` is unfalsifiable and the result is
  suppressed rather than reported.
""")
    X, y, t = syn.make_xor_interaction(
        n_samples=4000, n_xor_pairs=3, n_main_effects=0, n_noise=25,
        task=TASK, random_state=0)
    tr, te = _split(len(y), 0)
    sc = lambda cols: H.score_subset(X[tr], y[tr], X[te], y[te], cols,
                                     task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=0)["auc"]
    power = syn.xor_probe_power(t, sc)
    print(f"  power check: all pairs intact AUC {power['intact']:.3f} vs one half of each\n"
          f"  pair removed AUC {power['broken']:.3f}  ->  gap {power['gap']:+.3f}")
    if not power["has_power"]:
        print("\n  !! PROBE HAS NO POWER -- breaking pairs costs nothing measurable here.\n"
              "     Any pairs_broken count below would be meaningless. Raise xor_weight or\n"
              "     lower n_main_effects/n_noise until the gap exceeds 0.05. SUPPRESSED.")
        return
    print("  probe has power; results below are meaningful.\n")

    print(f"  {'lambda':>7} | {'|F|':>9}  {'intact':>6}  {'broken':>6}  {'dropped':>7}  {'AUC':>13}")
    print("  " + "-" * 74)
    for pen in PENALTY_SWEEP:
        rows = []
        for s in range(n_seeds):
            Xs, ys, ts = syn.make_xor_interaction(
                n_samples=4000, n_xor_pairs=3, n_main_effects=0, n_noise=25,
                task=TASK, random_state=s)
            trs, tes = _split(len(ys), s)
            sel, _, _ = H.fit_and_select(
                Xs[trs], ys[trs], penalty=pen, param_name=PENALTY_PARAM, task=TASK,
                num_boost_round=NUM_BOOST_ROUND, seed=s, extra_params=_params(pen))
            m = H.interaction_metrics(sel, ts["xor_pairs"], ts["main_cols"], ts["noise_cols"])
            m["n_selected"] = len(sel)
            m.update(H.score_subset(Xs[trs], ys[trs], Xs[tes], ys[tes], sel,
                                    task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s))
            rows.append(m)
        a = H.aggregate(rows)
        print(f"  {pen:>7} | {a['n_selected'][0]:>4.1f}±{a['n_selected'][1]:<4.1f} "
              f"{a['pairs_intact'][0]:>6.2f}  {a['pairs_broken'][0]:>6.2f}  "
              f"{a['pairs_dropped'][0]:>7.2f}  {a['auc'][0]:>6.3f}±{a['auc'][1]:<6.3f}")
    print("""
  READ: 'broken' > 0 is the dangerous outcome -- half a pair kept, which is a
  feature that is useless without its partner. 'dropped' (both halves gone) is
  cleaner but still costs the signal; watch AUC fall as it rises. Whichever
  lambda E2/E3 favour must be checked here before it can be recommended.""")


# --------------------------------------------------------------------------- #
# E5: high-dimensional, p > n
# --------------------------------------------------------------------------- #
def e5_high_dim(n_seeds=None):
    n_seeds = n_seeds or N_SEEDS
    _hdr("E5", "high-dimensional (p > n), where selection actually matters")
    print("""  Everything above is n >> p, which is the regime where feature selection is
  least consequential. The paper's setting is 100-10,000 features. Here p=600,
  n=400, of which 6 blocks carry all the signal -- so a selector has to find
  ~6 needles among 600, with fewer rows than columns.
""")
    lam = 0.01
    rows = {n: [] for n in ("regularized", "stock gain", "L1", "all", "oracle")}
    ks = []
    for s in range(n_seeds):
        X, y, t = syn.make_redundant_blocks(
            n_samples=400, block_sizes=(4, 4, 3, 3, 1, 1), n_noise=584,
            corr=0.9, task=TASK, random_state=s)
        tr, te = _split(len(y), s)
        sc = lambda cols: H.score_subset(X[tr], y[tr], X[te], y[te], cols,
                                         task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s)
        sel, _, _ = H.fit_and_select(
            X[tr], y[tr], penalty=lam, param_name=PENALTY_PARAM, task=TASK,
            num_boost_round=NUM_BOOST_ROUND, seed=s, extra_params=_params(lam))
        k = max(len(sel), 1)
        ks.append(k)
        for name, cols in {
            "regularized": sel,
            "stock gain": H.select_by_importance(X[tr], y[tr], k, importance_type="gain",
                                                 task=TASK, num_boost_round=NUM_BOOST_ROUND, seed=s),
            "L1": H.select_by_l1(X[tr], y[tr], k, task=TASK, seed=s),
            "all": set(range(X.shape[1])),
            "oracle": set(t["driver_cols"]),
        }.items():
            r = sc(cols)
            r["n_selected"] = len(cols)
            # Diagnose WHERE a selector goes wrong here, not just that it does:
            # picking pure noise is a different failure from picking the wrong
            # member of a real block, and they have different fixes.
            r.update(H.block_metrics(cols, t["blocks"], t["noise_cols"],
                                     driver_cols=t["driver_cols"]))
            rows[name].append(r)

    print(f"  n=400, p=600, 6 blocks (16 relevant cols).  matched k = "
          f"{np.mean(ks):.1f}±{np.std(ks):.1f}   ({n_seeds} seeds)\n")
    print(f"  {'selector':<14} | {'|F|':>6}  {'AUC':>13}  {'accuracy':>13}  "
          f"{'blocks hit':>10}  {'noise picked':>12}")
    print("  " + "-" * 86)
    for name in ("oracle", "regularized", "stock gain", "L1", "all"):
        a = H.aggregate(rows[name])
        print(f"  {name:<14} | {a['n_selected'][0]:>6.1f}  "
              f"{a['auc'][0]:>6.3f}±{a['auc'][1]:<6.3f} "
              f"{a['accuracy'][0]:>6.3f}±{a['accuracy'][1]:<6.3f}  "
              f"{a['block_coverage'][0]:>10.2f}  {a['noise_selected'][0]:>12.1f}")
    print("""
  READ: with p > n, 'all' should degrade noticeably -- that gap is the headroom
  a selector has to win back. If 'regularized' closes it and 'stock gain' does
  not, that is the strongest case the fork can make.

  If instead 'regularized' trails while picking lots of noise, the mechanism to
  suspect is lock-in: under ensemble scope, whatever enters F first is exempt
  from the penalty for the rest of training. With p > n the earliest splits are
  chosen on very noisy gain estimates, so a spurious feature can be admitted
  early and then permanently preferred, while genuine signal stays penalized.
  Boosting is far more exposed to this than the paper's random forests, which
  re-randomise candidate features at every node and so keep exploring.""")


EXPERIMENTS = {
    "E1": e1_convergence,
    "E2": e2_selection,
    "E3": e3_justification,
    "E4": e4_interactions,
    "E5": e5_high_dim,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=list(EXPERIMENTS),
                    help="experiments to run (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds and rounds -- for checking the plumbing, not results")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    global NUM_BOOST_ROUND, N_SEEDS
    if args.quick:
        NUM_BOOST_ROUND, N_SEEDS = 100, 2
    if args.seeds:
        N_SEEDS = args.seeds

    print(f">>> fork param '{PENALTY_PARAM}', scope '{SCOPE}', "
          f"{NUM_BOOST_ROUND} rounds, {N_SEEDS} seeds")
    if args.quick:
        print(">>> QUICK MODE: plumbing check only, numbers are not results.")

    for name in args.which:
        key = name.upper()
        if key not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; pick from {list(EXPERIMENTS)}")
        EXPERIMENTS[key](n_seeds=N_SEEDS)
    print("\nDone.")


if __name__ == "__main__":
    main()
