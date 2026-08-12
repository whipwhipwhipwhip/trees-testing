"""
Real benchmark datasets with recoverable ground truth.

The synthetic generators in `synthetic.py` have exact ground truth but their
signal is *linear*, which quietly hands the advantage to L1 -- in E3 the fork
beats tree-importance baselines decisively yet only ties L1 logistic. A tie
with a free standard method is the ceiling on any justification argument, and
it is an artifact of the design rather than a property of the method.

MADELON fixes that without giving up ground truth.

Why this dataset specifically
-----------------------------
Madelon is from the NIPS 2003 feature selection challenge (Guyon et al.) and is
one of the benchmark families used in the regularized-trees paper itself. Its
construction is *exactly* the structure this fork targets:

  * 5 informative features -- points on the vertices of a 5-dimensional
    hypercube, so the signal is XOR-like and genuinely NONLINEAR;
  * 15 redundant features -- random linear combinations of those 5;
  * 480 "probes" -- pure noise, added deliberately as known-irrelevant.

That gives three things at once, which no other candidate offers together:

  1. A real, recognised benchmark rather than data we made up.
  2. Ground truth that is *verifiable from the data itself* (below), not
     trusted from a README.
  3. Nonlinearity, which removes L1's structural advantage. Measured on the
     half-split used here: L1 recovers 3/20 of the real features and scores
     AUC 0.579, near chance, while a stock LightGBM on all features reaches
     0.867. Whatever the fork does on Madelon, a tie with L1 will not be the
     reason it fails.

Recovering the ground truth
---------------------------
The challenge published the counts (20 real, 480 probes) but not the indices.
They are recoverable exactly, because the 15 redundant features are linear
combinations of the 5 informative ones while the probes are independent noise:
the real features are precisely those with a strong correlation partner.

Applying `|corr| > 0.5 with at least one other feature` yields exactly 20
columns, and those 20 are also the top 20 by gain importance -- two independent
routes agreeing on the same set. `load_madelon` asserts the count, so if the
upstream data ever changes the run fails loudly instead of scoring against a
silently wrong truth set.
"""

from __future__ import annotations

import numpy as np

MADELON_OPENML_ID = 1485

# Documented composition of the challenge dataset.
MADELON_N_REAL = 20          # 5 informative + 15 redundant linear combinations
MADELON_N_FEATURES = 500


def _correlation_blocks(X, cols, threshold=0.5):
    """Group `cols` into blocks by absolute correlation.

    The redundant features are linear combinations of the informative ones, so
    members of a group correlate at ~0.99 while probes correlate with nothing.
    A block is the analogue of one `make_redundant_blocks` block: a perfect
    redundancy-aware selector keeps one member of each.

    AVERAGE linkage, not single linkage. Single linkage chains: because each
    redundant feature is a combination of SEVERAL informative ones, weak
    cross-group links exist, and one such link is enough to merge two groups.
    That collapses 18 of the 20 into a single block and makes `block_coverage`
    trivially 1.00 for any selector -- the metric stops discriminating at all.
    Average linkage requires groups to be close on the whole, and recovers
    blocks of sizes [2, 2, 3, 4, 4, 5].
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    sub = np.abs(np.corrcoef(X[:, cols].T))
    np.fill_diagonal(sub, 1.0)
    dist = np.clip(1.0 - sub, 0.0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(int(cols[idx]))
    return [sorted(v) for v in groups.values()]


def load_madelon(threshold=0.5, cache=True):
    """Fetch Madelon and derive its ground truth. Returns (X, y, truth).

    `truth` matches the shape the synthetic generators return, so every existing
    metric (`block_metrics`, `recovery_metrics`) applies unchanged:

      relevant_cols : the 20 real features
      noise_cols    : the 480 probes
      blocks        : correlation blocks among the real features
      n_blocks      : number of such blocks -- the ideal |F|
      n_features    : 500

    Requires network access on first call; scikit-learn caches it afterwards.
    """
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=MADELON_OPENML_ID, as_frame=False,
                     parser="liac-arff", cache=cache)
    X = np.asarray(d.data, dtype=np.float64)
    # labels come through as strings ('1'/'2'); map to 0/1
    classes = np.unique(d.target)
    y = (np.asarray(d.target) == classes[-1]).astype(int)

    if X.shape[1] != MADELON_N_FEATURES:
        raise RuntimeError(
            f"expected {MADELON_N_FEATURES} Madelon features, got {X.shape[1]}; "
            "the upstream dataset changed and the ground truth below is unsafe")

    C = np.corrcoef(X.T)
    np.fill_diagonal(C, 0.0)
    has_partner = (np.abs(C) > threshold).any(axis=1)
    relevant = np.where(has_partner)[0]

    if len(relevant) != MADELON_N_REAL:
        raise RuntimeError(
            f"correlation rule recovered {len(relevant)} real features, expected "
            f"{MADELON_N_REAL}. Ground truth is not trustworthy -- do not score "
            "against it until this is understood.")

    blocks = _correlation_blocks(X, relevant, threshold)
    noise = sorted(set(range(X.shape[1])) - set(int(i) for i in relevant))
    truth = {
        "relevant_cols": [int(i) for i in relevant],
        "noise_cols": noise,
        "blocks": blocks,
        "n_blocks": len(blocks),
        "driver_cols": None,          # which member is "informative" is not published
        "n_features": int(X.shape[1]),
        "name": "madelon",
    }
    return X, y, truth
