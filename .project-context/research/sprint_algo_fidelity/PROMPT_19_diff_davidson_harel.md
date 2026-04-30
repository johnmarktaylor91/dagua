<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 19 ADVERSARIAL DIFF for **davidson_harel** family.

The user wants every last divergence between dagua's davidson_harel
implementation and the igraph C reference, line by line, brutally
exhaustive. The previous Round 13 fix landed energy weights + move
schedule but the family is still partial_match (median 0.238). The
user is NOT satisfied with "good enough" -- they want every remaining
divergence catalogued.

## Inputs

**Dagua side:**
- `dagua/layout/ops/davidson_harel.py` (entire file, all classes)
- `dagua/layout/ops/pipelines/davidson_harel.py`

**igraph reference (READ ALL):**
- `/home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c` (entire file, ~700 lines)
- The dh_energy function, applyEnergy, the move-acceptance loop,
  the no_tries=30 candidate-direction sweep, fine-tuning pass

**Existing analysis:**
- `.project-context/research/sprint_algo_fidelity/ROUND_12_BLOCKED.md`
- `eval_output/algo_fidelity/round_13/SUMMARY.md`

## What to do

**This is a DIAGNOSIS-ONLY round.** Do NOT edit any source files.
No commits.

Produce ONE document:
`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_davidson_harel.md`

Format -- be brutally exhaustive:

```
# Davidson-Harel: line-by-line dagua-vs-igraph diff

## 1. Overall structure
... compare igraph_layout_davidson_harel main flow vs dagua pipeline.
... call out anything that's structurally different (phases, ordering, state).

## 2. Energy function
For each energy term:
- Term: <name>
- igraph: <C source line:expression>
- dagua: <python file:line:expression>
- Mathematically identical? (Y/N) — if N, what differs precisely
- Numerically identical? (Y/N — different units, scaling, sign?)
- Severity: HIGH/MEDIUM/LOW (HIGH = changes shape outcome, LOW = doesn't)

(Do this for every term: node-node distance, border, edge length,
edge crossing count, node-edge distance, plus any energy term in
either side that the other side doesn't have.)

## 3. Move-acceptance loop
Per-step comparison:
- Outer loop count
- Per-node inner loop
- Candidate direction generation (igraph: 30 shuffled circular dirs;
  dagua: ?)
- Move radius schedule (igraph: width/2 -> cool_fact decay; dagua: ?)
- Acceptance criterion (igraph: dE < 0 OR exp(-dE/move_radius) > rand;
  dagua: ?)
- Fine-tuning phase (igraph: maxiter*0.01 trailing iterations;
  dagua: ?)

## 4. RNG / determinism
- Initial layout RNG (igraph: RNG_UNIF on seed; dagua: torch.rand on seed?)
- Move-direction RNG
- Move-acceptance RNG
- Anywhere RNG semantics could diverge between numpy and torch

## 5. Edge cases
- Disconnected components handling
- Self-loops handling
- Multi-edges handling
- Empty graph

## 6. Hyperparameter alignment table
| Param | igraph default | dagua default | Match? | Notes |
|---|---|---|---|---|
| ... full table

## 7. Ranked fix list
Numbered, ranked by expected RMSD impact:
1. <Most impactful divergence> — file:line in dagua, file:line in igraph,
   proposed fix in <X> lines, expected median delta
2. ...
N. <Smallest divergence>

## 8. Recommended Round 20 fix scope
Combine top-K levers into one Round 20 prompt.
```

This document is the input for Round 20's actual code fix.

## Constraints

- DIAGNOSIS ONLY. No file edits. No commits.
- Read the whole davidson_harel.c file (~700 lines) and the whole
  dagua davidson_harel.py file (~1500 lines).
- Be exhaustive -- the user explicitly wants "every last little
  divergence". A short report is failure; produce 2000+ words if
  needed.
- Cite specific line numbers on both sides for every claim.

## End state
File: `ROUND_19_DIFF_davidson_harel.md`. No code changes. No commit.
</task>

<scope_constraints>
DIAGNOSIS-ONLY. NO file edits. NO commits.
Allowed: read files, run `grep`, run `wc`, run `git log`, write
ONE markdown report at the path above.
</scope_constraints>

<verification_loop>
File ROUND_19_DIFF_davidson_harel.md exists, has at least 8 sections
per the format, has line:line references throughout, has a ranked
fix list with at least 5 items.
</verification_loop>
