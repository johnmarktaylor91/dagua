# r71 FMMM Over-Dispersion: Documented Residual (2026-06-12)

## Verdict: graph-dependent multilevel-architecture gap (no uniform fix; not chaos)

Two diagnosis rounds (gem/fmmm codex + fmmm-v2 codex, both HIGH effort) converged on the
same conclusion: dagua's FMMM fidelity path over-disperses (~1.4x the seeded OGDF reference
ensemble) on ~185 combos, and there is NO uniform parameter/convergence fix.

## Root cause (identified, not closed)

- `classic_fmmm_steps*` with `fidelity_mode=True` dispatches to
  `_layout_ogdf_fmmm_small_fidelity`, which is a **single-level** force layout.
- OGDF's FMMM is **multilevel** (multipole): it applies fixed iterations PER COARSENING
  LEVEL via `get_max_mult_iter`, with the coarse levels removing most of the seed-dependent
  jitter before the fine level.
- dagua's single-level path runs a fixed iteration budget at the finest level only -> it
  retains more seed-dependent variance -> systematic over-dispersion.

## Why not fixed in r71

The only uniform candidate (routing through dagua's existing native multilevel ops path)
improved 3 of 4 anchor graphs but REGRESSED rgg_100:

| graph | OGDF disp | dagua single-level (stored) | native-multilevel candidate |
|---|---:|---:|---:|
| random_dag_50 | 0.0875 | 0.1321 | 0.1094 |
| transformer_layer | 0.0762 | 0.1642 | 0.0637 |
| multiscale_skip_cascade | 0.0856 | 0.1866 | 0.1214 |
| rgg_100 | 0.0177 | 0.0024 | 0.0431 |

The existing native multilevel ops path is NOT a faithful port of OGDF's specific
multilevel scheme (coarsening criteria, per-level iteration counts, multipole force
approximation) -- so swapping to it trades one mismatch for another.

## Ruled out
- Not chaos: the OGDF reference is reproducible (W_R ~0.38, far from the ~1.41 max;
  E_cross overshoots reference self-variation by ~18x). Genuine port difference.
- Not stopping-criterion: OGDF runner sets FixedIterations (threshold stopping inactive).
- Not cooling: OGDF default coolTemperature=false; no missing terminal anneal.
- Not post-pass: dagua already applies OGDF's 10 cooldown + 20 fine-tune.

## To close (future work, out of r71 scope)
A faithful port of OGDF FMMM's multilevel scheme to the dagua fidelity path:
coarsening (solar-system/star merging) + per-level `get_max_mult_iter` iteration budgets +
multipole force approximation. This is a substantial new implementation (~the size of the
original FMMM port), not a parameter tweak.

## Disposition for the report
FMMM residual combos: ~23% quality-equivalent (equal drawing), remainder labeled
`FMMM_MULTILEVEL_ARCHITECTURE_RESIDUAL` -- identified root cause, deferred fix.
