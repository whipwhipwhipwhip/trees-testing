# Does "Feature Selection via Regularized Trees" work in gradient boosting?

**No — and the reason is structural.** This repo implements Deng & Runger's
[regularized trees](https://arxiv.org/abs/1201.1587) as a native LightGBM parameter — a ~170-line C++
fork of the tree learner — and evaluates it against planted ground truth and a standard benchmark.

The method beats its matched baseline under a **random forest** on 10 of 10 seeds, and loses under
**gradient boosting** on 10 of 10 — same data, same penalty, same metric, only the base learner
changed.

| selector | base learner | \|F\| | test AUC | true features found | vs matched baseline |
| --- | --- | ---: | ---: | ---: | --- |
| RRF | random forest | 108.0 | 0.866 ± 0.008 | 18.0 / 20 | **wins 10/10** (+0.0120, SE 0.0019) |
| RRF | gradient boosting | 31.7 | 0.850 ± 0.022 | 8.0 / 20 | **loses 0/10** (−0.0461, SE 0.0071) |
| top-k gain importance | — | 15.0 | **0.928 ± 0.004** | 14.7 / 20 | best result overall |

<sub>MADELON (NIPS 2003 feature selection challenge), 10 seeds, paired per-seed differences, matched
cardinality. Every row scored identically: a stock LightGBM retrained on the selected columns only.</sub>

## Three findings

**1. Regularized trees is a random-forest method and does not transfer to boosting.**
A random forest re-randomises the candidate feature set at every node and bootstraps each tree, so the
accumulated feature set *F* is built from many independent looks. Boosting offers every feature at
every node and commits greedily: *F* is fixed early by a few noisy splits, and the penalty can only
*reorder* gains — it has no mechanism to evict a feature already admitted. Swapping only the base
learner flips the verdict, which isolates that as the cause rather than a story told after the fact.

**2. Top-k by gain importance is a brutally strong baseline.** One training run and an `argsort` beat
every configuration tested here, including the guided variant (GRRF) built specifically to fix the
failure above. Very few papers report this comparison at matched cardinality; it is worth reporting.

**3. The selected feature set does not converge unless *F* spans the whole ensemble.** The paper
accumulates *F* across all trees. Reset it per tree and |F| never stops growing — λ only changes the
*rate* at which features enter, so any |F| you quote is really a statement about `num_boost_round`.
This was a real bug in the fork's first implementation, found by reading the paper closely, and it is
the kind of error that silently produces publishable-looking numbers.

## Reproduce

```bash
uv sync
uv run pytest tests/ -q                                  # 20 harness self-tests
uv run python -m trees_testing.run_milestones E8 --seeds 10   # the headline result
uv run python -m trees_testing.run_milestones --seeds 20      # everything (~40 min)
```

Requires the forked LightGBM (see [Fork](#the-fork)). MADELON is fetched from OpenML on first use.

## The experiments

| | question | verdict |
| --- | --- | --- |
| **E1** | Does \|F\| stop growing? | Only under `ensemble` scope. Under `tree` scope, never. |
| **E2** | Is the selected set the *right* set? | Structurally perfect on synthetic blocks; picks a correlated copy over the true driver ~1 block in 4. |
| **E3** | Does it beat top-k importance? | On synthetic linear designs, yes: +0.0264 AUC, 18/20 seeds. Ties L1. |
| **E4** | Does it destroy interactions? | Yes. At the λ that selects cleanly, XOR AUC falls 0.953 → 0.584. |
| **E5** | What happens when p > n? | Fails badly: 5.2 of 7 picks are pure noise, worse than using all 600 features. |
| **E6** | A real benchmark (MADELON)? | Loses to top-k importance at every λ. GRRF improves it a lot, still loses. |
| **E7** | Is it at least faster? | No. The penalized fit is ~16% *slower* — it rescales gains already computed. |
| **E8** | Is boosting the problem? | Yes. Positive control: the verdict flips under a random forest. |

## What makes the evaluation trustworthy

The interesting engineering here is in the harness, not the method. Feature selection is unusually
easy to fool yourself about, so the harness is built to fail loudly:

- **The XOR probe suppresses its own results** when it lacks the power to detect what it claims to
  measure. Originally, destroying an interaction pair cost 0.008 accuracy — `pairs_broken` was
  unfalsifiable. It now measures intact-vs-broken AUC first (gap +0.512 after the fix) and refuses to
  report if the gap is too small to mean anything.
- **"Converged" is separated from "saturated."** Vanilla LightGBM's |F| also stops growing — because
  every feature is in it. Conflating that with selection would make the do-nothing baseline look like
  a selector. E1 reports `SELECTED` / `saturated` / `growing` as distinct outcomes.
- **Tree structure is reported alongside |F|.** Every selection metric is computed on the union of
  features across trees, which is blind to trees collapsing into single-feature stumps — the actual
  failure mode at small λ (measured: mean 1.01 features/tree). `tree_diagnostics` catches it.
- **Baselines are matched on cardinality and paired on seed**, and include the cheap alternative that
  the method has to beat, not just random subsets.
- **The harness's own assumptions are tested.** `selection_curve` derives the whole |F|-vs-rounds
  curve from one fit, relying on boosting being prefix-consistent. A test asserts that against
  independently trained models — if it ever breaks, every convergence result breaks with it.
- **Ground truth is verified, not trusted.** MADELON's real-feature indices were never published. They
  are recovered from correlation structure (redundant features correlate ~0.99 with their source;
  probes correlate with nothing), which yields exactly 20 columns — the same 20 that top gain
  importance. Two independent routes agreeing, asserted in the test suite.

## Layout

```
src/trees_testing/
  synthetic.py           # generators with planted ground truth (blocks, relevant/irrelevant, XOR)
  real_data.py           # MADELON + ground-truth recovery
  selection_harness.py   # metrics, baselines, convergence, tree diagnostics, GRRF plumbing
  run_milestones.py      # E1-E8
tests/
  test_harness_properties.py   # 20 tests, incl. the harness's own assumptions
```

## The fork

`unused_feature_penalty` and friends live in a LightGBM fork (`../LightGBM_RegularisedTrees`,
+172 lines across `src/treelearner/serial_tree_learner.{h,cpp}` and `include/LightGBM/config.h`):

| parameter | meaning |
| --- | --- |
| `unused_feature_penalty` | λ ∈ (0,1]. Split gain of a not-yet-used feature is scaled by this. 1.0 = off. |
| `unused_feature_penalty_scope` | `tree` \| `ensemble`. Whether *F* resets per tree. **`ensemble` is the paper's semantics**; `tree` is the default for backwards compatibility and does not converge. |
| `unused_feature_penalty_gamma` | γ ∈ [0,1] for GRRF: λᵢ = (1−γ)λ₀ + γ·Imp′ᵢ |
| `unused_feature_penalty_guide` | per-feature Imp′ᵢ = Impᵢ / maxⱼ Impⱼ from a preliminary model |

`rebuild-dev.sh` in that repo builds only the C++ shared library and drops it into this project's
venv — seconds rather than a full wheel rebuild.

## Limitations

- One real benchmark. MADELON is well-suited to the question (nonlinear, redundant, 96% noise,
  recoverable ground truth) but conclusions from a single dataset should be held loosely.
- The RF comparison uses LightGBM's `boosting=rf`, which is not identical to the paper's
  implementation — notably it retains LightGBM's leaf-wise growth.
- Deng & Runger's original paper *does* cover regularized boosted trees, so the contribution here is
  narrower than "nobody tried boosting": it is the matched-cardinality comparison against importance
  ranking, the ensemble-scope requirement, and the controlled base-learner swap.
- The cached-split staleness described in the fork's design notes is still unaddressed.

## References

- Deng & Runger, [Feature Selection via Regularized Trees](https://arxiv.org/abs/1201.1587) (2012)
- Deng & Runger, [Gene Selection with Guided Regularized Random Forest](https://arxiv.org/abs/1209.6425) (2013)
- Guyon et al., NIPS 2003 feature selection challenge (MADELON)
