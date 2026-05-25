<task>
R34 RECOVER NeuLay reference. R33 refaudit confirmed `neulay` and `NeuLay`
packages are NOT installable in this env. /tmp/graph-drawing has the NeuLay-2.py
script but it's not a Python package and has hard-coded dataset loading.

User said: "leave NOTHING on the table". Recover NeuLay.

## Your job

### Phase A: Inspect the script

- Read `/tmp/graph-drawing/NeuLay-2.py` (or wherever the original NeuLay script
  lives in that clone). Identify the layout function call signature, deps, and
  side effects.

### Phase B: Build an importable wrapper

Create `dagua/eval/competitors/neulay_wrapper.py` (or similar) that:
1. Imports the NeuLay-2.py module logic by factoring out hard-coded loaders
2. Exposes `def layout_neulay_reference(edge_index, num_nodes, seed, **kwargs) -> torch.Tensor`
   that returns positions
3. Is fully side-effect-free at import time (no top-level dataset reads)

If the script uses code that's NOT in the env (e.g., specific torch_geometric
features), document the gap and provide a near-equivalent.

### Phase C: Wire the reference

Update `dagua/eval/competitors/neulay_competitor.py` (or create one) so the
`neulay` reference engine points to your wrapper instead of the broken stub.

### Phase D: Verify

```python
from dagua.eval.competitors import get_competitor
c = get_competitor('neulay')
print(c.available())  # should now be True
from dagua.eval.graphs import get_test_graphs
tg = [t for t in get_test_graphs() if t.name == 'linear_3layer_mlp'][0]
r = c.layout(tg.graph, seed=42)
print('shape:', r.pos.shape if r.pos is not None else 'NONE', 'err:', r.error)
```

Then bounded live_compare:
```bash
python scripts/algo_fidelity_live_compare.py classic_neulay neulay --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_34/neulay_recover/post
```

Expected: real RMSD numbers (not insufficient_data) for the 6 neulay variants.

## Implementation

Use commit-safe wrapper. Commits:
1. `feat(eval): round 34 neulay -- importable wrapper from NeuLay-2.py`
2. `feat(eval): round 34 neulay -- wire reference to wrapper`

## Output
`eval_output/algo_fidelity/round_34/neulay_recover/SUMMARY.md`.
</task>

<completeness_contract>
Either: (a) working NeuLay wrapper producing positions, OR (b) explicit
documented blocker (e.g., script uses unavailable upstream dep that can't be
installed).
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Read deeply.
</default_follow_through_policy>
