"""
Synthetic data generators with *known* ground truth, for evaluating a
redundancy-penalizing feature selector (e.g. LightGBM's `unused_feature_penalty`).

The whole point of testing on synthetic data is that here we know the right
answer, so we can measure *correctness* of selection directly -- not just
downstream accuracy loss (which is all you can measure on real data, because
the true relevant feature set is unknown).

Three generators, one per milestone / probe:

  make_redundant_blocks   -> Milestone 1: does the penalty collapse each
                             correlated block to ~one representative?
  make_relevant_irrelevant-> Milestone 2: recovery of planted relevants vs noise
                             (sweep the penalty against this).
  make_xor_interaction    -> Interaction stress test: does the penalty wrongly
                             kill the second half of an interaction pair?

Every generator returns an (X, y, truth) triple where `truth` is a dict with
the ground-truth column bookkeeping AFTER the column shuffle, so you can score
selections directly against it.

Pure numpy. No LightGBM dependency.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _standardize(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-12)


def _correlated_copy(base: np.ndarray, corr: float, rng: np.random.Generator) -> np.ndarray:
    """Return a column correlated with `base` at approximately `corr`.

    base is assumed ~standardized. copy = corr*base + sqrt(1-corr^2)*eps.
    """
    eps = rng.standard_normal(base.shape[0])
    return corr * base + np.sqrt(max(0.0, 1.0 - corr**2)) * eps


def _labels_from_eta(eta: np.ndarray, task: str, rng: np.random.Generator,
                     noise_std: float = 0.0):
    """Turn a linear predictor into labels/targets.

    classification: Bernoulli via logistic link (stochastic, realistic).
    regression:     eta plus gaussian noise.
    """
    if task == "classification":
        p = 1.0 / (1.0 + np.exp(-eta))
        return rng.binomial(1, p).astype(int)
    elif task == "regression":
        return eta + noise_std * rng.standard_normal(eta.shape[0])
    raise ValueError(f"unknown task: {task}")


def _shuffle_columns(X: np.ndarray, groups: dict, rng: np.random.Generator):
    """Shuffle feature columns and remap every index in `groups` accordingly.

    Position must not leak structure, so we permute columns and carry the
    bookkeeping through the permutation. `groups` maps name -> list-of-int or
    list-of-list-of-int (block structure); both are remapped.
    """
    n_features = X.shape[1]
    perm = rng.permutation(n_features)
    X_shuffled = X[:, perm]
    # old index -> new index
    inv = np.empty(n_features, dtype=int)
    inv[perm] = np.arange(n_features)

    def remap(obj):
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return [[int(inv[i]) for i in block] for block in obj]
        return [int(inv[i]) for i in obj]

    remapped = {k: remap(v) for k, v in groups.items()}
    return X_shuffled, remapped


# --------------------------------------------------------------------------- #
# Milestone 1: redundant blocks
# --------------------------------------------------------------------------- #
def make_redundant_blocks(
    n_samples: int = 3000,
    block_sizes=(4, 4, 3, 1, 1),
    n_noise: int = 25,
    corr: float = 0.95,
    task: str = "classification",
    weight_scale: float = 1.5,
    noise_std: float = 1.0,
    random_state=None,
):
    """Plant several *driver* features that carry signal; expand some of them
    into a block of correlated copies; pad with pure-noise features.

    Each entry of `block_sizes` is one block. block_size k means: 1 driver that
    actually generates the label + (k-1) correlated copies that carry the same
    information (redundant). A singleton block (size 1) is just a lone relevant
    feature with no copies.

    Ground truth (`truth` dict, post-shuffle indices):
      blocks        : list[list[int]] -- each inner list is one block's columns
      driver_cols   : list[int]       -- the columns that truly generate y
      noise_cols    : list[int]       -- pure-noise columns
      relevant_cols : list[int]       -- union of all block members
      n_blocks      : int

    A perfect redundancy-aware selector picks exactly ONE member from every
    block and zero noise: |F| == n_blocks.
    """
    rng = np.random.default_rng(random_state)
    block_sizes = list(block_sizes)
    n_blocks = len(block_sizes)

    # drivers: one per block, each standardized
    drivers = _standardize(rng.standard_normal((n_samples, n_blocks)))

    # signal comes ONLY from drivers
    weights = weight_scale * rng.choice([-1.0, 1.0], size=n_blocks) \
        * rng.uniform(0.6, 1.0, size=n_blocks)
    eta = drivers @ weights
    y = _labels_from_eta(eta, task, rng, noise_std=noise_std)

    # build feature columns block by block
    columns = []
    blocks = []          # list of column-index lists (pre-shuffle)
    driver_cols = []     # the driver column within each block (pre-shuffle)
    col = 0
    for b, size in enumerate(block_sizes):
        base = drivers[:, b]
        member_idx = []
        # driver first
        columns.append(base)
        driver_cols.append(col)
        member_idx.append(col)
        col += 1
        # correlated copies
        for _ in range(size - 1):
            columns.append(_correlated_copy(base, corr, rng))
            member_idx.append(col)
            col += 1
        blocks.append(member_idx)

    # noise
    noise_cols = []
    for _ in range(n_noise):
        columns.append(rng.standard_normal(n_samples))
        noise_cols.append(col)
        col += 1

    X = np.column_stack(columns)

    groups = {"blocks": blocks, "driver_cols": driver_cols, "noise_cols": noise_cols}
    X, groups = _shuffle_columns(X, groups, rng)

    relevant_cols = sorted([i for block in groups["blocks"] for i in block])
    truth = {
        "blocks": groups["blocks"],
        "driver_cols": groups["driver_cols"],
        "noise_cols": groups["noise_cols"],
        "relevant_cols": relevant_cols,
        "n_blocks": n_blocks,
        "n_features": X.shape[1],
    }
    return X, y, truth


# --------------------------------------------------------------------------- #
# Milestone 2: relevant vs irrelevant (no redundancy)
# --------------------------------------------------------------------------- #
def make_relevant_irrelevant(
    n_samples: int = 3000,
    n_relevant: int = 8,
    n_noise: int = 42,
    task: str = "classification",
    weight_scale: float = 1.5,
    noise_std: float = 1.0,
    random_state=None,
):
    """`n_relevant` independent informative features + `n_noise` pure-noise
    features. No redundancy -- this isolates relevant/irrelevant discrimination
    so you can sweep the penalty and watch recovery precision/recall/F1.

    Ground truth:
      relevant_cols : list[int]
      noise_cols    : list[int]
    """
    rng = np.random.default_rng(random_state)
    relevant = _standardize(rng.standard_normal((n_samples, n_relevant)))
    weights = weight_scale * rng.choice([-1.0, 1.0], size=n_relevant) \
        * rng.uniform(0.6, 1.0, size=n_relevant)
    eta = relevant @ weights
    y = _labels_from_eta(eta, task, rng, noise_std=noise_std)

    noise = rng.standard_normal((n_samples, n_noise))
    X = np.column_stack([relevant, noise])

    groups = {
        "relevant_cols": list(range(n_relevant)),
        "noise_cols": list(range(n_relevant, n_relevant + n_noise)),
    }
    X, groups = _shuffle_columns(X, groups, rng)
    truth = {
        "relevant_cols": sorted(groups["relevant_cols"]),
        "noise_cols": sorted(groups["noise_cols"]),
        "n_features": X.shape[1],
    }
    return X, y, truth


# --------------------------------------------------------------------------- #
# Interaction stress test: XOR pairs
# --------------------------------------------------------------------------- #
def make_xor_interaction(
    n_samples: int = 4000,
    n_xor_pairs: int = 2,
    n_main_effects: int = 0,
    n_noise: int = 25,
    task: str = "classification",
    weight_scale: float = 1.5,
    xor_weight: float = 4.0,
    random_state=None,
):
    """The failure mode probe. Each XOR pair is two features that are each
    *individually* uninformative about y but *jointly* determine it via XOR.

    A redundancy penalty that judges a feature on its marginal gain risks
    dropping the second member of a pair before the tree discovers the
    interaction. Both members must be retained or the signal is destroyed.

    THE PROBE MUST HAVE POWER. If main effects and label noise carry most of
    the signal, then destroying an interaction pair costs almost no accuracy,
    and the probe cannot distinguish "the penalty broke an interaction" from
    "the interaction never mattered" -- it can report `pairs_broken > 0` with
    no measurable consequence, which is not evidence of anything. Hence:

      * `n_main_effects` defaults to 0, so XOR is the ONLY signal;
      * `xor_weight` (default 4.0, vs the old 1.5) makes the label close to
        deterministic given an intact pair.

    Verify power for any given configuration with `xor_probe_power` below
    rather than trusting these defaults blindly.

    Ground truth:
      xor_pairs      : list[list[int]] -- each inner list is the 2 cols of a pair
      xor_cols       : flat list of all xor columns (all must be retained)
      main_cols      : list[int]       -- main-effect relevants
      noise_cols     : list[int]
      relevant_cols  : xor_cols + main_cols
    """
    rng = np.random.default_rng(random_state)
    columns = []
    eta = np.zeros(n_samples)
    xor_pairs = []
    col = 0

    for _ in range(n_xor_pairs):
        a = rng.standard_normal(n_samples)
        b = rng.standard_normal(n_samples)
        # XOR on signs: high when exactly one of a,b is positive
        contrib = np.where((a > 0) ^ (b > 0), 1.0, -1.0)
        eta += xor_weight * contrib
        columns.append(a); columns.append(b)
        xor_pairs.append([col, col + 1])
        col += 2

    main_cols = []
    if n_main_effects > 0:
        M = _standardize(rng.standard_normal((n_samples, n_main_effects)))
        w = weight_scale * rng.choice([-1.0, 1.0], size=n_main_effects)
        eta += M @ w
        for j in range(n_main_effects):
            columns.append(M[:, j])
            main_cols.append(col); col += 1

    noise_cols = []
    for _ in range(n_noise):
        columns.append(rng.standard_normal(n_samples))
        noise_cols.append(col); col += 1

    y = _labels_from_eta(eta, task, rng)
    X = np.column_stack(columns)

    groups = {"xor_pairs": xor_pairs, "main_cols": main_cols, "noise_cols": noise_cols}
    X, groups = _shuffle_columns(X, groups, rng)

    xor_cols = sorted([i for pair in groups["xor_pairs"] for i in pair])
    truth = {
        "xor_pairs": groups["xor_pairs"],
        "xor_cols": xor_cols,
        "main_cols": sorted(groups["main_cols"]),
        "noise_cols": sorted(groups["noise_cols"]),
        "relevant_cols": sorted(xor_cols + groups["main_cols"]),
        "n_features": X.shape[1],
    }
    return X, y, truth


def xor_probe_power(truth, score_fn):
    """Measure whether breaking an interaction pair actually costs anything.

    Compares a model given every relevant feature against one given the same
    features with ONE HALF of each XOR pair removed -- precisely the damage the
    penalty is suspected of doing. `score_fn(cols) -> float` trains and scores
    on a held-out split (see `selection_harness.score_subset`).

    If `gap` is not clearly positive, the probe is inert: any `pairs_broken`
    result it produces is unfalsifiable, because breaking pairs demonstrably
    costs nothing on this data. Fix the generator before reading the result.

    Returns {intact, broken, gap, has_power}.
    """
    intact_cols = set(truth["relevant_cols"])
    broken_cols = set(intact_cols)
    for pair in truth["xor_pairs"]:
        broken_cols.discard(pair[1])          # drop one half of every pair

    intact = float(score_fn(sorted(intact_cols)))
    broken = float(score_fn(sorted(broken_cols)))
    gap = intact - broken
    return {
        "intact": intact,
        "broken": broken,
        "gap": gap,
        "has_power": bool(gap > 0.05),
    }
