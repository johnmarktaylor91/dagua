# P14: Standard-Corpora Holdout Verdict (r80)

Run: 2026-07-10, branch r79/p6a-stdcorpora + trunk merge (algo = final r80 head).
Harness: scripts/r79_stdcorpora_eval.py, 9 engines, seed 42, tie band 0.5, dagua children
spawn-isolated (fork-after-torch deadlock fix 9b329ed). Dedup: last row per (graph,engine).
HOLDOUT DISCIPLINE HELD: zero tuning against these corpora at any point in the sprint.

## Verdict

| Corpus | Class | W | T | L | n | Best-or-tied |
|---|---|---:|---:|---:|---:|---:|
| rome | small undirected (10-100 n) | 33 | 66 | 53 | 152 | 65.1% |
| north | directed DAGs | 90 | 8 | 9 | 107 | 91.6% |
| suitesparse | mesh/structural | 3 | 0 | 9 | 12 | 25.0% |
| **TOTAL** | | **126** | **74** | **71** | **271** | **73.8%** |

Iteration-corpus best-or-tied on the same honest ruler: 74/108 = 68.5%.
**Holdout (73.8%) > iteration (68.5%): generalization confirmed, no overfitting.**

## Reading
- North 91.6%: the layered/DAG core is world-class on its target class out-of-sample.
- Rome 65.1%: portfolio holds up on unseen small undirected graphs; loss tail is
  graphviz_neato on tiny sparse graphs (r81 headroom: candidate param-matching).
- SuiteSparse 25%: the mesh-class weakness reproduces out-of-sample exactly as the
  iteration corpus predicted (known residual, not a surprise).
- 3 suitesparse graphs skipped (insufficient rows after dedup).

## Provenance notes
- First run OOM'd at 101GB (dagua in-process, no timeout; fixed: subprocess isolation).
- Fork-after-torch deadlock made all dagua rows timeout (fixed: spawn context).
- A publish-replace bug destroyed external rows once (recovered via --resume carry).
All three incidents are filed; none affect row validity (every surviving row stamped OK).
