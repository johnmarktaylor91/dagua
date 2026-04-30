<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 19 ADVERSARIAL DIFF for **neulay** family. The original NeuLay
paper code was just cloned -- now we can do real diff.

## Inputs

**NeuLay reference paper code (READ ALL):**
- `/home/jtaylor/projects/_references/NeuLay/old_code/NeuLay-2.py` -- main implementation
- `/home/jtaylor/projects/_references/NeuLay/old_code/FDL.py` -- force-directed baseline
- `/home/jtaylor/projects/_references/NeuLay/NeuLay_pyg.ipynb` -- PyTorch Geometric notebook (read as JSON if needed)
- `/home/jtaylor/projects/_references/NeuLay/README.md`

Paper: Both, Dehmamy, Yu, Barabasi (2023) "Accelerating Network Layouts
Using Graph Neural Networks", Nature Communications 14:1560.
GitHub: https://github.com/csabath95/NeuLay

**Dagua side (READ ALL):**
- `dagua/layout/ops/neulay.py`
- `dagua/layout/ops/pipelines/neulay.py`
- `dagua/eval/competitors/neulay_competitor.py` (for variant params)

**Note**: The dagua/eval competitor adapter expects an installed
upstream `neulay` Python package, which is currently unavailable. The
cloned repo at `/home/jtaylor/projects/_references/NeuLay` is the
authoritative source -- compare against it directly.

## What to do

**DIAGNOSIS-ONLY.** Produce ONE document:
`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_neulay.md`

Cover:

1. **Overall flow**: NeuLay has two phases (GCN reparameterization +
   direct refinement). Verify dagua follows same structure.
2. **GCN architecture**: hidden dim, num layers, activation, output
   head dimension. NeuLay reference vs dagua impl.
3. **Energy / loss function**: NeuLay uses a specific energy formulation
   (typically attractive proportional to spring + repulsive
   proportional to inverse-power). Diff every term.
4. **Optimizer + LR**: sklearn vs PyTorch differences. NeuLay reference
   uses what optimizer? What schedule?
5. **Early stopping / patience** (dagua has _PATIENCE=10; reference?)
6. **Initial coordinates** (random init scheme)
7. **Latent dimensions** (dagua uses _LATENT_DIM=10; reference?)
8. **Pair sampling for repulsion** (dagua has pair-query-radius and
   refresh interval; reference logic?)
9. **Number of training steps** (gcn_steps, linear_steps defaults)
10. **RNG** (torch vs numpy)
11. **Hyperparameter alignment table**
12. **Ranked fix list**
13. **Installation feasibility**: can we install the cloned NeuLay
    repo via `pip install -e /home/jtaylor/projects/_references/NeuLay`
    or similar? Document any missing deps.
14. **Recommended Round 20 fix scope**

## Constraints

DIAGNOSIS ONLY. No file edits. No commits.
</task>

<scope_constraints>
DIAGNOSIS-ONLY. Read-only.
</scope_constraints>

<verification_loop>
File ROUND_19_DIFF_neulay.md exists, exhaustive, with line:line refs.
</verification_loop>
