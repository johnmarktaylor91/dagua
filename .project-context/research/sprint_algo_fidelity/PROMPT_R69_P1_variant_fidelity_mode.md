<task>
NOTE (R69 P1b re-dispatch): A prior run of this task correctly HARD-STOPPED because
`linlog.py` delegated its fidelity path to the reference. That has now been FIXED
(commit a700ccd: real in-pipeline linlog port, bit-exact, no delegation). So linlog is
now safe to patch -- linlog variants take `fidelity_mode=True`. Proceed with the full
task below. Still honor the hard-stop rule if you find ANY OTHER pipeline that delegates.

You are patching dagua's benchmark variant registry so the fidelity benchmark
exercises the BIT-EXACT reimplementation ports, not dagua's default fast tensor
codepath.

## Repo context
- Project: /home/jtaylor/projects/dagua (PyTorch graph-layout engine).
- File to edit: `dagua/eval/variants.py`. It defines 118 `classic_*` reimplementation
  variants via the `_variant(...)` helper (defined at variants.py:61).
- `_variant()` signature (positional):
    _variant(variant_id, base_engine, display_name, reimpl_params, original_engine,
             original_params, is_true_original, is_stochastic, is_heavy, max_nodes=None)
  The 4th arg `reimpl_params` (a dict, e.g. `{"rounds": 50}`) is dagua's reimpl config.
  The 5th arg `original_engine` (e.g. `"igraph_davidson_harel"`, `"graphviz_neato"`,
  `"ogdf_fmmm"`, `"fa2_default"`, `"tsne_graph"`, `"linlog"`, `"umap_*"`, `"neulay_*"`,
  `"nx_*"`, `"cytoscape_fcose"`) is the REFERENCE adapter the variant is validated against.

## The bug
Only 9 of 118 `classic_*` variants set `"fidelity_mode"` in their `reimpl_params`.
The other 109 run dagua's DEFAULT (fast, non-matching) layout, so the fidelity
benchmark has been comparing the wrong implementation -> 100% PARTIAL verdicts.
Every pipeline in `dagua/layout/ops/pipelines/*.py` already SUPPORTS `fidelity_mode`
(verified: all 24 reference it). The variants just don't opt in.

## What to do
For EVERY `classic_*` variant in variants.py that does NOT already have
`"fidelity_mode"` in its `reimpl_params` dict, add the CORRECT `fidelity_mode` value
so the pipeline routes to its bit-exact, reference-matching codepath.

### Deriving the correct value (do NOT blindly guess -- VERIFY against pipeline code)
For each variant:
1. Read its `original_engine` (5th positional arg) to know which reference family it
   targets.
2. Open the matching pipeline `dagua/layout/ops/pipelines/<base>.py` (map base_engine
   `classic_drl` -> `drl.py`, `classic_neato` -> `neato.py`, `classic_fmmm` -> `fmmm.py`,
   etc.) and read EXACTLY which `fidelity_mode` values route to the bit-exact path
   (look for `if fidelity_mode ==`, string comparisons, `is True`, helper predicates
   like `_is_graphviz_dot_fidelity_mode`).
3. Set `reimpl_params["fidelity_mode"]` to the value that makes that pipeline match
   THIS variant's reference family.

Starting heuristic (CONFIRM each against the pipeline; correct if the code disagrees):
- reference `graphviz_*`  -> `"graphviz"`
- reference `igraph_*`    -> `"igraph"`  (use `True` if the pipeline only knows one mode)
- reference `ogdf_*`      -> `"ogdf"` if the pipeline accepts it, else `True`
- reference `fa2_*`, `sgd2_*`, `tsne_*`, `linlog*`, `umap_*`, `neulay_*`, `nx_*` -> `True`
- reference `cytoscape_fcose`, `gephi_yifanhu`, or any engine with NO bit-exact port
  in `dagua/layout/ops/pipelines/` -> DO NOT add fidelity_mode; record it as "no-port".

Existing reference examples already in the file (match this style):
  `{"maxiter": 200, "epsilon": 0.0001, "pack": True, "fidelity_mode": "graphviz"}`  (sfdp)
  `{"barycenter_passes": 24, "rank_sep": 1.0, "node_sep": 1.0, "fidelity_mode": "igraph"}` (sugiyama)
  `{"steps": 200, "fidelity_mode": True}`  (fmmm graphviz fdp fidelity)
</task>

<constraints>
## HARD constraint -- Python-only (the product's whole selling point)
dagua must stay pip-installable with ZERO non-Python dependencies. The ONLY thing
that violates this is requiring a SEPARATE NON-PYTHON PROGRAM. Therefore:
- A fidelity pipeline MUST NOT shell out to an external binary: NO `subprocess`,
  `os.system`, `Popen`, `check_output`, NO invoking the graphviz `dot`/`neato`/`sfdp`/
  `fdp` binaries, NO `pygraphviz`, NO Java/OGDF process.
- Importing COMMON PYTHON LIBRARIES is FINE and expected: numpy, scipy, sklearn,
  networkx, torch. (e.g. tsnet.py importing `sklearn.manifold._t_sne._joint_probabilities`
  is acceptable -- it is a Python numerical primitive, not an external program.)

## HARD constraint -- no fidelity-output delegation to the reference
The fidelity codepath must COMPUTE positions itself. It must NEVER call the reference
adapter / reference layout function to PRODUCE the reimplementation's output positions
(e.g. must not call `igraph .layout_*()`, must not call the graphviz binary, must not
return `sklearn.manifold.TSNE().fit_transform()` wholesale). Using a stateless numerical
primitive (distances -> affinity matrix) is allowed; delegating the actual layout is not.
This is a documented project rule (5 prior incidents R51/R57/R58). If while patching you
discover ANY pipeline whose fidelity path delegates layout output to the reference,
STOP and report it -- do not silently "fix" by patching the variant.

## Scope discipline
- ONLY add `"fidelity_mode"` keys to `reimpl_params` dicts. Do NOT change param
  semantics, display names, pairings, `original_engine`, or any other arg.
- Leave the 9 variants that already have `fidelity_mode` UNCHANGED unless one is
  provably wrong vs its pipeline (if so, note it explicitly).
- Do not touch `dagua/layout/_archive/` (frozen oracles).
</constraints>

<verification_loop>
1. `python -c "import dagua.eval.variants"` -- must import clean.
2. Confirm count: `grep -c '"fidelity_mode"' dagua/eval/variants.py` should rise from 9
   to (118 minus the no-port count).
3. Grep the touched pipelines for delegation red flags and confirm NONE shell out:
   `grep -rnE "subprocess|os\.system|Popen|check_output|pygraphviz" dagua/layout/ops/pipelines/`
   (expected: empty.)
4. Smoke test: pick 2 patched engines that should be deterministic-ish (e.g. a graphviz
   one and an igraph one). Layout one small graph (10-20 nodes) WITH the new fidelity_mode
   and WITHOUT it; confirm the outputs DIFFER (proving fidelity_mode actually re-routes).
   If a pipeline accepts fidelity_mode but produces IDENTICAL output to default, flag that
   pipeline as "fidelity_mode is a no-op" in the report -- that is a real finding.
</verification_loop>

<output>
1. Edit `dagua/eval/variants.py` in place with the fidelity_mode additions.
2. Write a mapping report to
   `eval_output/fidelity_report_r69/p1_variant_fidelity_mapping.md` -- a markdown table
   with columns: variant_id | original_engine (reference) | pipeline file | chosen
   fidelity_mode | verified-routes-to-bit-exact-path (Y/N/no-op/no-port). Group "no-port"
   and "no-op" variants in a clearly-labeled section at the bottom.
3. Commit to the current branch (`develop` -- this is the active fidelity sprint branch;
   do NOT create a new branch). Commit message:
   `feat(eval): R69 P1 -- opt all reimpl variants into fidelity_mode (bit-exact ports)`
   with a summary of how many variants were patched and any no-port/no-op findings.
</output>

<default_follow_through_policy>
Proceed autonomously with the most reasonable low-risk interpretation. Only STOP and
report (do not guess) if: (a) you find a pipeline whose fidelity path delegates layout
output to the reference (the hard constraint above), or (b) a pipeline shells out to a
non-Python binary. For ordinary ambiguity (which exact fidelity_mode string an engine
wants), pick the value the pipeline code supports and note your choice in the report --
the downstream 5-seed Procrustes check will empirically catch any mis-mapping.
</default_follow_through_policy>
