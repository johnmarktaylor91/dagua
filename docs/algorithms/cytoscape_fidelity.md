# Cytoscape Family Fidelity

Reference packages installed in this build:

- `cytoscape` 3.34.0
- `cytoscape-cose-bilkent` 4.1.0
- `cytoscape-cise` 2.0.1
- `cytoscape-avsdf` 1.0.0

## Results

Run:

```bash
python scripts/verify_cytoscape_fidelity.py
```

Expected tier summary:

| Algorithm | Reference | Tier | Named residual |
| --- | --- | --- | --- |
| `avsdf` | `cytoscape-avsdf` | bit-exact target | none expected; deterministic AVSDF order and circle placement are ported natively |
| `cose` | Cytoscape core `cose` | distributional | spring update clipping/temperature stage |
| `cose_bilkent` | `cytoscape-cose-bilkent` | distributional partial | compound gravity and disconnected tiling stage |
| `cise` | `cytoscape-cise` | distributional partial | inter-cluster force relaxation stage |

`avsdf` is the only deterministic algorithm in this batch. The native pipeline
ports the adjacent-vertex-smallest-degree-first order, local crossing
postprocess, and circle placement directly from `avsdf-base`.

The three spring/compound layouts are distinct from `fcose`. They are registered
as separate algorithms and expose Cytoscape-compatible option names, but the
current native implementation intentionally stops at a documented partial:
legacy CoSE force semantics are represented by a native CoSE step, while the
deep Cytoscape compound machinery from `cose-base`/`cise-base` remains the named
first-divergent stage.
