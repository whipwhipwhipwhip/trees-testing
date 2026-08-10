"""
Validation of the generators + metrics WITHOUT LightGBM, using mock selectors.
Confirms: (1) ground-truth bookkeeping survives the column shuffle, (2) drivers
actually carry the signal, (3) correlated copies hit target correlation,
(4) every metric returns the right numbers on hand-constructed selections,
(5) the pluggable harness runs end-to-end with an injected fit_fn.
"""
import numpy as np
import trees_testing.synthetic as syn
import trees_testing.selection_harness as H


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


print("== generators: ground-truth integrity ==")

# redundant blocks
X, y, t = syn.make_redundant_blocks(n_samples=4000, block_sizes=(4, 4, 3, 1, 1),
                                    n_noise=25, corr=0.95, random_state=1)
n_block_members = sum(len(b) for b in t["blocks"])
check("n_features == block members + noise",
      t["n_features"] == n_block_members + len(t["noise_cols"]))
check("n_features matches X", t["n_features"] == X.shape[1])
check("blocks sizes preserved (4+4+3+1+1=13 members)", n_block_members == 13)
check("no index overlap between relevant and noise",
      len(set(t["relevant_cols"]) & set(t["noise_cols"])) == 0)
check("one driver per block", len(t["driver_cols"]) == t["n_blocks"])

# each driver sits inside exactly one block
for d in t["driver_cols"]:
    in_blocks = sum(d in b for b in t["blocks"])
    check(f"driver {d} in exactly one block", in_blocks == 1)

# correlated copies really are correlated with their block's driver
worst = 1.0
for block, drv in zip(t["blocks"], t["driver_cols"]):
    for member in block:
        if member == drv:
            continue
        c = abs(np.corrcoef(X[:, member], X[:, drv])[0, 1])
        worst = min(worst, c)
check(f"all copies corr >0.9 with driver (worst={worst:.3f})", worst > 0.9)

# noise truly uncorrelated with y (point-biserial ~ 0)
noise_corrs = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in t["noise_cols"]]
check(f"noise weakly assoc with y (max={max(noise_corrs):.3f})", max(noise_corrs) < 0.1)

print("\n== generators: relevant/irrelevant ==")
X2, y2, t2 = syn.make_relevant_irrelevant(n_samples=4000, n_relevant=8,
                                          n_noise=42, random_state=2)
check("8 relevant + 42 noise = 50", t2["n_features"] == 50)
check("relevant/noise disjoint",
      len(set(t2["relevant_cols"]) & set(t2["noise_cols"])) == 0)
rel_corrs = [abs(np.corrcoef(X2[:, j], y2)[0, 1]) for j in t2["relevant_cols"]]
noise_corrs2 = [abs(np.corrcoef(X2[:, j], y2)[0, 1]) for j in t2["noise_cols"]]
check(f"relevant more associated with y than noise "
      f"(min rel={min(rel_corrs):.3f} vs max noise={max(noise_corrs2):.3f})",
      min(rel_corrs) > max(noise_corrs2))

print("\n== generators: XOR interaction ==")
X3, y3, t3 = syn.make_xor_interaction(n_samples=6000, n_xor_pairs=2,
                                      n_main_effects=3, n_noise=25, random_state=3)
# each xor member alone should be ~uninformative; the pair jointly informative
for pair in t3["xor_pairs"]:
    a, b = pair
    ca = abs(np.corrcoef((X3[:, a] > 0).astype(int), y3)[0, 1])
    cb = abs(np.corrcoef((X3[:, b] > 0).astype(int), y3)[0, 1])
    # joint XOR feature
    joint = ((X3[:, a] > 0) ^ (X3[:, b] > 0)).astype(int)
    cj = abs(np.corrcoef(joint, y3)[0, 1])
    check(f"xor pair {pair}: each member weak (a={ca:.3f},b={cb:.3f}) "
          f"joint strong (={cj:.3f})", ca < 0.1 and cb < 0.1 and cj > 0.3)

print("\n== metrics: hand-constructed selections ==")

# block_metrics: ideal selection = one member per block, no noise
blocks = t["blocks"]
ideal = {b[0] for b in blocks}                      # one per block
m = H.block_metrics(ideal, blocks, t["noise_cols"])
check("ideal: full block coverage", m["block_coverage"] == 1.0)
check("ideal: 1.0 members per covered block", m["members_per_covered"] == 1.0)
check("ideal: |F| == n_blocks", m["n_selected"] == t["n_blocks"])
check("ideal: no noise", m["noise_selected"] == 0)

# vanilla-like selection = grab ALL members of every block + a few noise
greedy = set(t["relevant_cols"]) | set(t["noise_cols"][:3])
m2 = H.block_metrics(greedy, blocks, t["noise_cols"])
check("greedy: still full coverage", m2["block_coverage"] == 1.0)
check("greedy: members_per_covered > 1 (redundant)", m2["members_per_covered"] > 1.0)
check("greedy: picks up noise", m2["noise_selected"] == 3)
check("greedy selects many more than ideal", m2["n_selected"] > m["n_selected"])

# recovery_metrics: perfect recovery
rm = H.recovery_metrics(set(t2["relevant_cols"]), t2["relevant_cols"], t2["noise_cols"])
check("perfect recovery: precision=1", rm["precision"] == 1.0)
check("perfect recovery: recall=1", rm["recall"] == 1.0)
check("perfect recovery: f1=1", rm["f1"] == 1.0)
check("perfect recovery: no noise", rm["noise_selected"] == 0)

# half the relevants + some noise
partial = set(t2["relevant_cols"][:4]) | set(t2["noise_cols"][:4])
rm2 = H.recovery_metrics(partial, t2["relevant_cols"], t2["noise_cols"])
check("partial recall = 0.5", abs(rm2["recall"] - 0.5) < 1e-9)
check("partial precision = 0.5", abs(rm2["precision"] - 0.5) < 1e-9)
check("partial noise_selected = 4", rm2["noise_selected"] == 4)

# interaction_metrics: intact vs broken
both = {i for pair in t3["xor_pairs"] for i in pair} | set(t3["main_cols"])
im = H.interaction_metrics(both, t3["xor_pairs"], t3["main_cols"], t3["noise_cols"])
check("all pairs intact", im["pairs_intact"] == t3["xor_pairs"].__len__())
check("no broken pairs", im["pairs_broken"] == 0)
check("all main effects kept", im["main_effects_kept"] == im["n_main_effects"])

# break one pair by dropping its second member
broken_sel = set(both)
broken_sel.discard(t3["xor_pairs"][0][1])
im2 = H.interaction_metrics(broken_sel, t3["xor_pairs"], t3["main_cols"], t3["noise_cols"])
check("breaking a pair registers as broken", im2["pairs_broken"] == 1)
check("remaining pair still intact", im2["pairs_intact"] == 1)

print("\n== harness: end-to-end with a MOCK fit_fn ==")

# A mock 'redundancy-aware' selector: returns split importance that uses one
# member per block + main effects + xor pairs, ignoring noise. Lets us confirm
# fit_and_select + used_features wire up correctly without LightGBM.
def mock_fit_redundancy_aware(X, y, penalty, param_name, task,
                              num_boost_round, extra_params, seed):
    imp = np.zeros(X.shape[1])
    # pretend the model learned to use exactly the 'ideal' columns
    for c in ideal:
        imp[c] = 10
    return imp, None

sel, imp, _ = H.fit_and_select(X, y, penalty=0.5, task="classification",
                               fit_fn=mock_fit_redundancy_aware)
check("used_features recovers the mock's selection", sel == ideal)
mm = H.block_metrics(sel, blocks, t["noise_cols"])
check("mock harness -> ideal block metrics", mm["members_per_covered"] == 1.0)

print("\nALL CHECKS PASSED")
