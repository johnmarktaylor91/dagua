# Sprint 23 Area E: Outerplanar + multi-component finishers

## Mandate

outerplanar_dag_20 (-0.74 vs igraph_sugiyama 73.16) and
multi_component_80 (-0.64 vs graphviz_dot 75.10) are tiny graphs where
the gradient pipeline is already close to ceiling but missing the last
~1 point because the outer-face / component-tile-permutation isn't
optimal.

Sprint-21a overlap_jitter and sprint-22b global_depth_align both
touched this area; the remaining gap is small enough that targeted
picker-safe permutation search should close them. These are insurance
bets -- low risk, modest individual lift, but they cumulatively help
push best-or-tied past 96%.

## Research questions

1. Score the current dagua output and competitor outputs on
   outerplanar_dag_20 and multi_component_80. Per-metric breakdown
   to identify which metric the competitor wins on.

2. Implement targeted polish for each:
   - outerplanar_dag_20: try outer-face rotation (the planar
     embedding's outer face has multiple valid choices; pick the
     one that maximizes composite)
   - multi_component_80: try inter-component reflection / rotation
     / row-major-vs-column-major tile arrangement; pick best

3. Empirically validate that the polish doesn't regress similar-
   structure graphs:
   - planar_60, sierpinski_42 (planar protected wins)
   - disconnected_encoder_residual, parallel_cycles_4x5 (multi-
     component, where sprint-22b/22d already win)

## Output spec

File: `.project-context/research/sprint_23_finish_line/E_outerplanar_multicomp__<agent>.md`

Sections:
- TL;DR
- Per-metric diagnosis on the two targets
- Algorithm sketches for each (Python pseudocode, ~50 LOC each)
- Empirical table including protected-win regression checks
- Picker decision: ship as polish candidates with what gate

## Constraints

- READ-ONLY on dagua/
- HEAD = sprint-22e finalize commit `d27fced`
- These are LOW-RISK insurance bets; estimated LOC each ~80
- Reference sprint-21a's overlap_jitter and sprint-22b's
  global_depth_align as precedent for this kind of narrow polish

## Word budget

1500-2500 words.
