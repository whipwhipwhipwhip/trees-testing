"""The headline figure: every regularized-trees configuration against the top-k frontier.

E6, E8 and E9 each compare one carried-forward setting with top-k gain importance at
that setting's own k. Those verdicts cannot be compared across settings, because
top-k's AUC falls steeply as k grows: a win at k~100 clears a far lower bar than a
loss at k~20. Plotting top-k as a curve over k and every setting as a point removes
that problem.

The bar a selector has to clear at |F| features is not top-k at exactly k = |F|: if
top-12 is both smaller and more accurate than a 100-feature selection, nobody should
want the 100. So the headline line is the best top-k with AT MOST |F| features (a
running maximum of the top-k curve), and a selector is only worth having if it sits
above that. The raw top-k curve is drawn too, because several random-forest settings
do beat it at their own k -- just not the envelope.

    uv run python -m trees_testing.frontier --seeds 10        # compute, then plot
    uv run python -m trees_testing.frontier --frontier-only   # recompute top-k curves only
    uv run python -m trees_testing.frontier --plot-only       # replot from results/

Computing needs the forked LightGBM. Plotting reads only results/frontier.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import trees_testing.selection_harness as H
from trees_testing.run_milestones import (E9_LAMBDAS, GRRF_GAMMA, NUM_BOOST_ROUND,
                                          PENALTY_PARAM, SCOPE, TASK, VANILLA_PENALTY,
                                          _params, _rf_params, _split)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "frontier.json"
FIGURES = ROOT / "docs"

FRONTIER_KS = [2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
METHODS = ["penalty only", "rule only", "penalty + rule (paper)", "GRRF"]


def _settings(p):
    """(method, label, penalty, K, guided) for every configuration E6/E8/E9 tried."""
    k_paper = int(np.ceil(np.sqrt(p)))
    k_sweep = sorted({1, max(1, round(k_paper / 4)), max(1, round(k_paper / 2)), k_paper})
    out = [("penalty only", f"lam={lam}", lam, None, False) for lam in E9_LAMBDAS]
    out += [("rule only", f"K={k}", VANILLA_PENALTY, k, False) for k in k_sweep]
    out += [("penalty + rule (paper)", f"lam={lam} K={k_paper}", lam, k_paper, False)
            for lam in E9_LAMBDAS]
    out += [("GRRF", f"lam={lam} gamma={GRRF_GAMMA}", lam, None, True) for lam in E9_LAMBDAS]
    return out


def compute(n_seeds, frontier_only=False):
    import lightgbm as lgb
    import trees_testing.real_data as rd

    X, y, t = rd.load_madelon()
    n, p = X.shape
    real = set(t["relevant_cols"])
    rfp = _rf_params(p)
    # (params with the candidate rule off, params with it on). Under RF the rule
    # replaces per-node sampling rather than stacking on it, as in E9.
    learners = {
        "GBDT": ({}, {}),
        "RF": (rfp, {k: v for k, v in rfp.items() if k != "feature_fraction_bynode"}),
    }
    settings = _settings(p)

    def score(s, cols):
        tr, te = _split(n, s)
        r = H.score_subset(X[tr], y[tr], X[te], y[te], cols, task=TASK,
                           num_boost_round=NUM_BOOST_ROUND, seed=s)
        return {"k": len(cols), "auc": r["auc"], "real": len(set(cols) & real)}

    if frontier_only:
        out = json.loads(RESULTS.read_text())
        assert out["n_seeds"] == n_seeds, "frontier and points must share seeds"
    else:
        out = {"n_seeds": n_seeds, "n_features": p, "num_boost_round": NUM_BOOST_ROUND,
               "frontier": {}, "points": []}
    for learner, (off, on) in learners.items():
        # Top-k frontier: one stock model per seed, every k a prefix of its ranking.
        curve = {k: [] for k in FRONTIER_KS}
        for s in range(n_seeds):
            tr, _ = _split(n, s)
            stock = lgb.train({"objective": "binary", "verbosity": -1, "seed": s, **off},
                              lgb.Dataset(X[tr], label=y[tr]),
                              num_boost_round=NUM_BOOST_ROUND)
            order = np.argsort(-np.asarray(stock.feature_importance("gain")))
            for k in FRONTIER_KS:
                curve[k].append(score(s, set(int(i) for i in order[:k])))
            print(f"  {learner} frontier, seed {s} done", flush=True)
        out["frontier"][learner] = {str(k): v for k, v in curve.items()}
        if frontier_only:
            continue

        for method, label, pen, k_cand, guided in settings:
            rows = []
            for s in range(n_seeds):
                tr, _ = _split(n, s)
                extra = {**(on if k_cand is not None else off),
                         **_params(pen, candidates=k_cand)}
                if guided:
                    # GBDT importance guides both learners, as in E6: it is the
                    # stronger guide, so this favours GRRF.
                    guide = H.importance_guide(X[tr], y[tr], task=TASK,
                                               num_boost_round=NUM_BOOST_ROUND, seed=s)
                    extra.update(H.grrf_params(guide, GRRF_GAMMA, SCOPE))
                sel, _, b = H.fit_and_select(X[tr], y[tr], penalty=pen,
                                             param_name=PENALTY_PARAM, task=TASK,
                                             num_boost_round=NUM_BOOST_ROUND,
                                             seed=s, extra_params=extra)
                r = score(s, sel)
                r["converged"] = bool(H.convergence_report(
                    H.selection_curve(b), n_features=p)["selected"])
                rows.append(r)
            out["points"].append({"learner": learner, "method": method,
                                  "setting": label, "seeds": rows})
            a = H.aggregate([{"k": r["k"], "auc": r["auc"]} for r in rows])
            print(f"  {learner:<5} {method:<23} {label:<18} "
                  f"|F|={a['k'][0]:6.1f}  AUC={a['auc'][0]:.3f}", flush=True)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
# Categorical slots 1-3 of the reference palette, light and dark steps. Only three
# hues clear the colour-vision checks for all pairs in a scatter, so the fourth
# method (GRRF) is drawn in neutral ink with hollow markers. Method is also encoded
# by marker shape, so identity never rests on colour alone.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#52514e"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c3c2b7"]},
}
MARKERS = ["o", "s", "D", "^"]


def _mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    return xs.mean(), xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0.0


def plot(theme="light"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(RESULTS.read_text())
    c = THEMES[theme]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": c["ink2"],
                         "axes.labelcolor": c["ink2"], "xtick.color": c["ink2"],
                         "ytick.color": c["ink2"], "text.color": c["ink"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=c["surface"])

    # Only the GBDT ranking is drawn: it is the cheap alternative to beat. The RF
    # ranking is in results/ and is no better at any k that matters.
    ks = sorted(int(k) for k in data["frontier"]["GBDT"])
    raw = np.array([np.mean([r["auc"] for r in data["frontier"]["GBDT"][str(k)]])
                    for k in ks])
    envelope = np.maximum.accumulate(raw)
    k_best = ks[int(np.argmax(raw))]
    for ax, learner in zip(axes, ("GBDT", "RF")):
        ax.set_facecolor(c["surface"])
        ax.grid(True, color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        ax.plot(ks, raw, color=c["ink2"], linewidth=1.2, linestyle=":", zorder=3)
        ax.step(ks, envelope, where="post", color=c["ink"], linewidth=2, zorder=3)
        if learner == "GBDT":
            ax.annotate(f"best top-k by gain using ≤ |F| features\n"
                        f"(one stock GBDT run; peaks at k = {k_best})",
                        (480, envelope[-1]), xytext=(0, 5), textcoords="offset points",
                        color=c["ink"], fontsize=9, fontweight="bold", ha="right", va="bottom")
            ax.annotate("top-k at exactly k = |F|", (200, raw[ks.index(200)]),
                        xytext=(0, -14), textcoords="offset points", color=c["ink2"],
                        fontsize=9, ha="center")

        for i, method in enumerate(METHODS):
            pts = [pt for pt in data["points"]
                   if pt["learner"] == learner and pt["method"] == method]
            if not pts:
                continue
            xy = sorted((np.mean([r["k"] for r in pt["seeds"]]),
                         *_mean_se([r["auc"] for r in pt["seeds"]])) for pt in pts)
            x, yv, ye = map(np.array, zip(*xy))
            ax.plot(x, yv, color=c["series"][i], linewidth=1, alpha=0.6, zorder=4)
            hollow = method == "GRRF"
            ax.errorbar(x, yv, yerr=ye, fmt=MARKERS[i], color=c["series"][i],
                        markersize=6.5, markeredgewidth=1.5,
                        markerfacecolor=c["surface"] if hollow else c["series"][i],
                        markeredgecolor=c["series"][i] if hollow else c["surface"],
                        elinewidth=1, capsize=0, zorder=5)

        ax.set_ylim(0.5, envelope[-1] + 0.055)   # headroom for the envelope label
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200", "500"])
        ax.set_xlabel("features selected, |F| (log scale)")
        ax.set_title(f"regularized trees on {'gradient boosting' if learner == 'GBDT' else 'a random forest'}",
                     color=c["ink"], fontsize=11, loc="left")
    axes[0].set_ylabel("test AUC, stock GBDT on selected features")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MARKERS[i], linestyle="", markersize=6.5,
                      markeredgewidth=1.5, color=c["series"][i],
                      markerfacecolor=c["surface"] if m == "GRRF" else c["series"][i],
                      markeredgecolor=c["series"][i] if m == "GRRF" else c["surface"])
               for i, m in enumerate(METHODS)]
    fig.legend(handles, METHODS, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), labelcolor=c["ink2"])
    fig.suptitle("For every configuration, a smaller top-k by gain importance scores higher",
                 x=0.06, ha="left", color=c["ink"], fontsize=13, fontweight="bold")
    fig.text(0.06, 0.885, f"MADELON, {data['n_seeds']} seeds, mean ± SE. "
             "Points along a line are one method's sweep of λ (or K). "
             "Any k from 10 to 20 gives top-k AUC 0.92–0.93.",
             color=c["ink2"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"frontier-{theme}.png"
    fig.savefig(path, dpi=160, facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--frontier-only", action="store_true",
                    help="recompute the top-k curves and keep the stored points")
    args = ap.parse_args()
    if not args.plot_only:
        compute(args.seeds, frontier_only=args.frontier_only)
    for theme in THEMES:
        plot(theme)


if __name__ == "__main__":
    main()
