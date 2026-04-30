# Round 19 Adversarial Diff -- DRL

Status: DIAGNOSIS ONLY
Family: drl
Date: 2026-04-30

## Files Read

- Dagua ops: `dagua/layout/ops/drl.py:1-1247`
- Dagua pipeline: `dagua/layout/ops/pipelines/drl.py:1-128`
- Dagua variants: `dagua/eval/variants.py:1310-1362`
- igraph layout entry/defaults: `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:240-480`
- igraph graph loop/update/energy/cutting: `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp:126-207`, `371-437`, `571-810`, `816-888`, `909-975`, `981-1135`, `1257-1268`
- igraph density grid: `/home/jtaylor/projects/_references/igraph/src/layout/drl/DensityGrid.cpp:66-256`
- igraph headers: `drl_graph.h:46-130`, `drl_layout.h:43-55`, `DensityGrid.h:46-77`, `drl_Node.h:46-66`
- Round 14: `eval_output/algo_fidelity/round_14/SUMMARY.md:1-62`, `.project-context/research/sprint_algo_fidelity/ROUND_14_DIAGNOSIS.md:35-68`, `.project-context/research/sprint_algo_fidelity/ROUND_14_RESIDUAL.md:32-43`

Note: the requested `drl_Node.cpp` file does not exist in this igraph checkout. `drl_Node.h` is a data-only `Node` class (`drl_Node.h:46-66`). The per-node update logic is in `drl_graph.cpp::update_node_pos` and `Solve_Analytic` (`drl_graph.cpp:909-975`, `1064-1135`).

## Executive Diff

The largest confirmed divergences are not the visible preset table for `default`; they are execution semantics:

1. **Effective phase sweep counts differ.** igraph calls `update_nodes()` before stage control on every `ReCompute()` call (`drl_graph.cpp:610-611` before `624-808`). This creates an initialization-parameter sweep even when `init_iterations=0`, plus extra boundary/final sweeps. Dagua runs exactly the declared per-phase iteration counts (`drl.py:914-975`, `1171-1198`) and therefore omits these sweeps.
2. **Fine-density simmer state differs.** igraph's fine-density bins are empty when simmer starts and are filled progressively during the first simmer sweep (`drl_graph.cpp:752-755`, `DensityGrid.cpp:141-156`, `247-255`). Dagua keeps buckets populated from the beginning because `_DensityGrid.add_node()` always updates `buckets` (`drl.py:483-488`, `1157-1159`), so simmer fine repulsion sees all nodes immediately.
3. **Coarse density kernel/boundary behavior differs.** igraph uses a separable product falloff and hard boundary penalty/throw (`DensityGrid.cpp:81-85`, `104-110`, `174-177`, `212-215`). Dagua uses a radial tent kernel and clamps out-of-view coordinates (`drl.py:430-433`, `448-455`, `469-481`).
4. **Node acceptance differs.** igraph always chooses between the analytic and random candidates only (`drl_graph.cpp:964-973`). Dagua also compares against the old coordinate and can reject both candidates (`drl.py:752-764`, `817-823`). Round 14 tested this, improved median RMSD by `0.017400`, and reverted because it missed the `0.030000` threshold (`ROUND_14_DIAGNOSIS.md:70-88`).
5. **Edge cutting differs.** igraph erases only from the current node's neighbor map (`drl_graph.cpp:1130-1133`). Dagua erases symmetrically from both endpoints (`drl.py:665-667`), changing later directed traversals.
6. **`REFINE` and `FINAL` presets have direct value mismatches.** Dagua `refine.init.damping_mult=1.0` and `cooldown_temperature=250.0` (`drl.py:204-211`), but igraph has `0` and `200` (`drl_layout.cpp:342-361`). Dagua `final.expansion=(50, 2000, 2, 1)` (`drl.py:213-220`), but igraph has `(50, 50, .1, .25)` (`drl_layout.cpp:374-403`).

## Pipeline Structure

### Dagua

Dagua builds a four-op pipeline: prepare params/adjacency, initialize positions, run all phases, finalize dtype/device (`pipelines/drl.py:38-44`). The public wrapper validates inputs, creates `LayoutProblem(seed=seed)`, uses a CPU execution plan, and returns final `float32` positions (`pipelines/drl.py:49-125`).

The phase solver builds one `_DensityGrid`, adds all nodes, seeds `random.Random(problem.seed)`, initializes cut state, then runs exactly six named phase specs in order: init, liquid, expansion, cooldown, crunch, simmer (`drl.py:1157-1178`). Each `DRLPhaseStep` loops `for _ in range(self._phase.iterations)` and then loops nodes in ascending order (`drl.py:914-975`).

### igraph

igraph entry validates non-negative damping and positive weights, constructs `drl::graph`, calls `init_parms(options)`, optionally reads seed coordinates, then calls `draw_graph(res)` (`drl_layout.cpp:435-480`). The graph constructor copies options into initial and phase schedule fields, builds symmetric neighbor maps, and initializes the density server (`drl_graph.cpp:126-207`).

The main loop is `while (ReCompute())` (`drl_graph.cpp:1257-1260`). `ReCompute()` always calls `update_nodes()` before stage-control logic (`drl_graph.cpp:610-611`). Stage control then mutates `temperature`, `attraction`, `damping_mult`, `min_edges`, `cut_off_length`, `fineDensity`, and `STAGE` (`drl_graph.cpp:624-808`).

### Effective Sweep Count Table

These counts are derived from igraph's `update_nodes()` position before the stage switch (`drl_graph.cpp:610-611`) and the stage conditions (`drl_graph.cpp:624-808`), compared with dagua's exact `range(iterations)` loop (`drl.py:914-975`). The igraph `stage 0` count includes one initialization-parameter sweep plus the liquid sweeps.

| Template | igraph effective sweeps | dagua effective sweeps | Divergence |
| --- | ---: | ---: | --- |
| DEFAULT | 755 = stage0 201, expansion 200, cooldown 201, crunch 51, simmer 101, final stage6 1 | 750 = init 0, liquid 200, expansion 200, cooldown 200, crunch 50, simmer 100 | Dagua misses 5 sweeps: init, cooldown boundary, crunch boundary, simmer boundary, final stage6 |
| COARSEN | 755 | 750 | Same sweep-count divergence as DEFAULT |
| COARSEST | 905 = stage0 201, expansion 200, cooldown 201, crunch 201, simmer 101, final stage6 1 | 900 | Same + longer crunch |
| REFINE | 155 = stage0 1, expansion 50, cooldown 51, crunch 51, simmer 1, final stage6 1 | 150 = init 0, liquid 0, expansion 50, cooldown 50, crunch 50, simmer 0 | Dagua misses init/cooldown/crunch/simmer/final sweeps |
| FINAL | 180 = stage0 1, expansion 50, cooldown 51, crunch 51, simmer 26, final stage6 1 | 175 = init 0, liquid 0, expansion 50, cooldown 50, crunch 50, simmer 25 | Same sweep-count divergence |

## Per-Phase Defaults: DEFAULT Template

igraph reference: `drl_layout.cpp:246-277`. Dagua reference: `drl.py:176-185`.

| Phase | Field | igraph | dagua | Match |
| --- | --- | ---: | ---: | --- |
| init | iterations, temperature, attraction, damping | 0, 2000, 10, 1.0 | 0, 2000.0, 10.0, 1.0 | Yes as stored; effective sweep differs |
| liquid | iterations, temperature, attraction, damping | 200, 2000, 10, 1.0 | 200, 2000.0, 10.0, 1.0 | Yes |
| expansion | iterations, temperature, attraction, damping | 200, 2000, 2, 1.0 | 200, 2000.0, 2.0, 1.0 | Yes |
| cooldown | iterations, temperature, attraction, damping | 200, 2000, 1, .1 | 200, 2000.0, 1.0, 0.1 | Yes as stored; cooling timing differs |
| crunch | iterations, temperature, attraction, damping | 50, 250, 1, .25 | 50, 250.0, 1.0, 0.25 | Yes as stored; effective sweep differs |
| simmer | iterations, temperature, attraction, damping | 100, 250, .5, 0 | 100, 250.0, 0.5, 0.0 | Yes as stored; effective sweep differs |

## Per-Phase Parameter Table for All Templates

`edge_cut` is `32.0 / 40.0 = 0.8` in both igraph and dagua for all templates (`drl_layout.cpp:243`, `drl.py:177-220`).

### DEFAULT

| Phase | igraph `(iters,temp,attr,damp)` | dagua `(iters,temp,attr,damp)` | Match |
| --- | --- | --- | --- |
| init | `(0, 2000, 10, 1.0)` `drl_layout.cpp:247-250` | `(0, 2000.0, 10.0, 1.0)` `drl.py:179` | Yes |
| liquid | `(200, 2000, 10, 1.0)` `drl_layout.cpp:252-255` | `(200, 2000.0, 10.0, 1.0)` `drl.py:180` | Yes |
| expansion | `(200, 2000, 2, 1.0)` `drl_layout.cpp:257-260` | `(200, 2000.0, 2.0, 1.0)` `drl.py:181` | Yes |
| cooldown | `(200, 2000, 1, .1)` `drl_layout.cpp:262-265` | `(200, 2000.0, 1.0, 0.1)` `drl.py:182` | Yes |
| crunch | `(50, 250, 1, .25)` `drl_layout.cpp:267-270` | `(50, 250.0, 1.0, 0.25)` `drl.py:183` | Yes |
| simmer | `(100, 250, .5, 0)` `drl_layout.cpp:272-275` | `(100, 250.0, 0.5, 0.0)` `drl.py:184` | Yes |

### COARSEN

| Phase | igraph | dagua | Match |
| --- | --- | --- | --- |
| init | `(0, 2000, 10, 1.0)` `drl_layout.cpp:279-282` | `(0, 2000.0, 10.0, 1.0)` `drl.py:188` | Yes |
| liquid | `(200, 2000, 2, 1.0)` `drl_layout.cpp:284-287` | `(200, 2000.0, 2.0, 1.0)` `drl.py:189` | Yes |
| expansion | `(200, 2000, 10, 1.0)` `drl_layout.cpp:289-292` | `(200, 2000.0, 10.0, 1.0)` `drl.py:190` | Yes |
| cooldown | `(200, 2000, 1, .1)` `drl_layout.cpp:294-297` | `(200, 2000.0, 1.0, 0.1)` `drl.py:191` | Yes |
| crunch | `(50, 250, 1, .25)` `drl_layout.cpp:299-302` | `(50, 250.0, 1.0, 0.25)` `drl.py:192` | Yes |
| simmer | `(100, 250, .5, 0)` `drl_layout.cpp:304-307` | `(100, 250.0, 0.5, 0.0)` `drl.py:193` | Yes |

### COARSEST

| Phase | igraph | dagua | Match |
| --- | --- | --- | --- |
| init | `(0, 2000, 10, 1.0)` `drl_layout.cpp:311-314` | `(0, 2000.0, 10.0, 1.0)` `drl.py:197` | Yes |
| liquid | `(200, 2000, 2, 1.0)` `drl_layout.cpp:316-319` | `(200, 2000.0, 2.0, 1.0)` `drl.py:198` | Yes |
| expansion | `(200, 2000, 10, 1.0)` `drl_layout.cpp:321-324` | `(200, 2000.0, 10.0, 1.0)` `drl.py:199` | Yes |
| cooldown | `(200, 2000, 1, .1)` `drl_layout.cpp:326-329` | `(200, 2000.0, 1.0, 0.1)` `drl.py:200` | Yes |
| crunch | `(200, 250, 1, .25)` `drl_layout.cpp:331-334` | `(200, 250.0, 1.0, 0.25)` `drl.py:201` | Yes |
| simmer | `(100, 250, .5, 0)` `drl_layout.cpp:336-339` | `(100, 250.0, 0.5, 0.0)` `drl.py:202` | Yes |

### REFINE

| Phase | igraph | dagua | Match |
| --- | --- | --- | --- |
| init | `(0, 50, .5, 0)` `drl_layout.cpp:343-346` | `(0, 50.0, 0.5, 1.0)` `drl.py:206` | **No: damping** |
| liquid | `(0, 2000, 2, 1.0)` `drl_layout.cpp:348-351` | `(0, 2000.0, 2.0, 1.0)` `drl.py:207` | Yes |
| expansion | `(50, 500, .1, .25)` `drl_layout.cpp:353-356` | `(50, 500.0, 0.1, 0.25)` `drl.py:208` | Yes |
| cooldown | `(50, 200, 1, .1)` `drl_layout.cpp:358-361` | `(50, 250.0, 1.0, 0.1)` `drl.py:209` | **No: temp** |
| crunch | `(50, 250, 1, .25)` `drl_layout.cpp:363-366` | `(50, 250.0, 1.0, 0.25)` `drl.py:210` | Yes |
| simmer | `(0, 250, .5, 0)` `drl_layout.cpp:368-371` | `(0, 250.0, 0.5, 0.0)` `drl.py:211` | Yes |

### FINAL

| Phase | igraph | dagua | Match |
| --- | --- | --- | --- |
| init | `(0, 50, .5, 0)` `drl_layout.cpp:375-378` | `(0, 50.0, 0.5, 0.0)` `drl.py:215` | Yes |
| liquid | `(0, 2000, 2, 1.0)` `drl_layout.cpp:380-383` | `(0, 2000.0, 2.0, 1.0)` `drl.py:216` | Yes |
| expansion | `(50, 50, .1, .25)` `drl_layout.cpp:385-388` | `(50, 2000.0, 2.0, 1.0)` `drl.py:217` | **No: temp, attr, damping** |
| cooldown | `(50, 200, 1, .1)` `drl_layout.cpp:390-393` | `(50, 200.0, 1.0, 0.1)` `drl.py:218` | Yes |
| crunch | `(50, 250, 1, .25)` `drl_layout.cpp:395-398` | `(50, 250.0, 1.0, 0.25)` `drl.py:219` | Yes |
| simmer | `(25, 250, .5, 0)` `drl_layout.cpp:400-403` | `(25, 250.0, 0.5, 0.0)` `drl.py:220` | Yes |

## Density Grid Implementation

Dagua does implement a grid-based density proxy; it is **not** plain O(N^2) repulsion. However, it is not equivalent to igraph's density grid.

| Aspect | igraph | dagua | Impact |
| --- | --- | --- | --- |
| Constants | `GRID_SIZE=1000`, `VIEW_SIZE=4000.0`, `RADIUS=10`, `HALF_VIEW=2000`, `VIEW_TO_GRID=.25` (`drl_layout.h:43-55`) | `grid_size=1000`, `view_size=4000.0`, `radius=10`, `cell_width=4`, origin `-2000` (`drl.py:22-39`, `421-425`) | Nominal constants align |
| Kernel | `fall_off=(RADIUS - abs(i))/RADIUS * (RADIUS - abs(j))/RADIUS` (`DensityGrid.cpp:80-85`) | radial `sqrt(xx^2+yy^2)`, `clamp(1 - distance/radius)` (`drl.py:430-433`) | Major shape mismatch: diamond/separable product vs circular cone |
| Cell conversion | `(coord + HALF_VIEW + .5) * VIEW_TO_GRID` (`DensityGrid.cpp:101-102`, `168-169`, `202-203`) | `floor((coord - origin) / cell_width)` with no `+0.5` offset (`drl.py:448-455`) | Off-by-half-cell trajectory differences |
| Boundary density query | returns `10000` inside 10-cell boundary (`DensityGrid.cpp:97-110`) | clamps to nearest valid cell (`drl.py:448-455`, `503-507`) | Dagua permits edge-hugging instead of strong boundary repulsion |
| Add/subtract outside grid | throws runtime error when kernel start falls outside grid (`DensityGrid.cpp:174-177`, `212-215`) | truncates kernel to grid overlap (`drl.py:469-481`) | Different behavior near/outside view |
| Coarse density | square of cell density (`DensityGrid.cpp:130-132`) | square of cell density (`drl.py:503-507`) | Formula aligns after grid value differs |
| Fine density | 3x3 bins, `1e-4/(distance + 1e-50)` (`DensityGrid.cpp:112-125`) | 3x3 buckets, `1e-4/(distance + 1e-12)` (`drl.py:534-548`) | Different singular guard by 38 orders of magnitude |
| Fine bins lifecycle | bins are separate and are populated only via `fineAdd()` when `fineDensity=true` (`DensityGrid.cpp:141-156`, `247-255`) | `buckets` are updated on every `add_node()` from initialization onward (`drl.py:483-488`, `490-501`) | Major simmer divergence |
| Fine subtract identity | `fineSubtract()` does `pop_front()` from the cell deque, no node-id check (`DensityGrid.cpp:234-240`) | `remove_node()` removes exact node id from a set (`drl.py:490-501`) | Dagua is cleaner but not igraph-equivalent |

Biggest density divergence: the fine-density bins. In igraph, when crunch switches to simmer it sets `fineDensity=true` and `fine_first_add` is still true (`drl_graph.cpp:752-755`, `882-886`). The first simmer sweep therefore starts with empty fine bins and fills them progressively through `update_density()` and `fineAdd()` (`drl_graph.cpp:981-996`, `DensityGrid.cpp:247-255`). Dagua enters simmer with every node already present in `buckets` because the same bucket structure was maintained during all prior coarse phases (`drl.py:483-488`, `1157-1159`). This changes repulsion throughout the entire first simmer sweep and likely later sweeps if igraph's FIFO `pop_front()` behavior diverges from exact-id removal.

## Per-Node Update Step

igraph per-node update:

1. `update_nodes()` iterates nodes in ascending order because `MAX_PROCS=1` (`drl_layout.h:38-40`, `drl_graph.cpp:824-833`, `871-878`).
2. For each mutable node, `update_node_pos()` computes `jump_length=.010*temperature` (`drl_graph.cpp:917-918`).
3. It subtracts the old node from the density server, respecting first-add/fine-first-add flags (`drl_graph.cpp:920-924`, `DensityGrid.cpp:149-156`).
4. It computes energy at the old coordinate (`drl_graph.cpp:923-925`).
5. `Solve_Analytic()` computes weighted neighbor centroid and applies damping (`drl_graph.cpp:1064-1089`).
6. Edge cutting happens inside `Solve_Analytic()` after centroid calculation (`drl_graph.cpp:1094-1133`).
7. It samples a random perturbation from the analytic candidate with `(.5 - RNG_UNIF01()) * jump_length` per axis (`drl_graph.cpp:939-941`).
8. It computes random-candidate energy (`drl_graph.cpp:943-947`).
9. It temporarily restores the old position and sometimes re-adds it to density (`drl_graph.cpp:955-962`).
10. It chooses the lower-energy candidate between analytic and random only (`drl_graph.cpp:964-973`).
11. `update_density()` then subtracts old and adds new for the actual accepted coordinate (`drl_graph.cpp:981-996`).

Dagua per-node update:

1. Removes the node from `_DensityGrid` unconditionally (`drl.py:752`).
2. Computes current energy at the old coordinate (`drl.py:753-764`).
3. Computes weighted centroid and analytic coordinate (`drl.py:766-772`).
4. Cuts edges before candidate energy evaluation for expansion/cooldown (`drl.py:773-780`).
5. Samples random perturbation with `rng.uniform(-0.5, 0.5) * jump_length` (`drl.py:782-792`).
6. Computes analytic and random energies (`drl.py:794-815`).
7. Accepts analytic only if it beats old, then random only if it beats the current best (`drl.py:817-822`).
8. Adds the accepted coordinate back to the grid (`drl.py:822-823`).

Key per-node divergences:

- **Acceptance rule:** igraph ignores old energy during final choice; dagua can keep old position (`drl_graph.cpp:964-973` vs `drl.py:817-823`).
- **Density update timing:** igraph may re-add old before final accepted update depending on `first_add`/`fine_first_add`, then `update_density()` handles old-to-new transition (`drl_graph.cpp:955-996`). Dagua directly removes then directly re-adds accepted (`drl.py:752`, `822-823`).
- **Fine-density cell membership:** igraph's `fineSubtract()` pops the front entry from the old cell and `fineAdd()` pushes a `Node` copy (`DensityGrid.cpp:234-255`); dagua removes/adds exact node ids (`drl.py:490-501`, `483-488`).

## Edge Cutting

Round 14's edge-cut diagnosis is verified.

igraph:

- `Solve_Analytic()` returns without cutting if `min_edges == 99` or `CUT_END >= 39500` (`drl_graph.cpp:1094-1102`).
- It computes `num_connections = sqrt(neighbors[node_ind].size())` once for the current node (`drl_graph.cpp:1104`).
- It only considers cutting if the current node has at least `min_edges` neighbors (`drl_graph.cpp:1113-1116`).
- It scores neighbor distance from the current node's centroid, multiplied by current-node `sqrt(degree)` (`drl_graph.cpp:1118-1122`).
- It erases only from `neighbors[node_ind]` (`drl_graph.cpp:1130-1133`).

Dagua:

- `_maybe_cut_long_edge()` runs only in expansion and cooldown, with `cut_end > 0.0` (`drl.py:773-780`).
- It skips if `len(neighbors) < min_edges` (`drl.py:650-652`).
- It computes a weighted centroid (`drl.py:654`) and scores each neighbor by distance from centroid times `sqrt(len(adjacency[neighbor]))` (`drl.py:657-660`).
- It erases symmetrically from current node and worst neighbor (`drl.py:665-667`).

Additional edge-cut divergences beyond symmetric/asymmetric removal:

- **Degree factor source:** igraph uses `sqrt(current node degree)` (`drl_graph.cpp:1104`, `1121`); dagua uses `sqrt(worst neighbor degree)` per candidate (`drl.py:657-660`).
- **Cut-disable threshold:** igraph disables cutting when `CUT_END >= 39500` (`drl_graph.cpp:1099-1102`). Dagua uses `cut_end > 0.0` and never models the `>=39500` no-cut guard (`drl.py:773-780`, `1162-1168`).
- **Cut floor:** igraph clamps `cut_length_end` to at least `1.0` (`drl_graph.cpp:381-386`); dagua leaves `cut_end=0.0` possible and makes `cut_rate=0.0` when `cut_end <= 0.0` (`drl.py:1162-1168`).

Impact: symmetric removal is likely a large trajectory divergence because igraph's neighbor structure becomes directed after cuts. If node `u` cuts `v`, `u` stops attracting to `v`, but `v` may still attract to `u` later. Dagua removes both, weakening attraction in both update directions.

## Move Acceptance and Cooling

### Candidate Acceptance

igraph computes old energy but uses it only as `energies[0]` before replacing `energies[0]` with the analytic coordinate's energy? Actually the code stores old energy in `energies[0]` (`drl_graph.cpp:923-925`), then moves to analytic (`927-929`), but does not recompute analytic energy before the random candidate. Therefore `energies[0]` is the old-coordinate energy while `updated_pos[0]` is the analytic coordinate. The final choice assigns `updated_pos[0]` when `energies[0] < energies[1]` (`drl_graph.cpp:964-968`). This is stranger than Round 14's simplified wording: igraph compares **old energy** against **random energy**, but if old energy wins, it accepts the analytic coordinate.

Dagua computes all three consistently: old, analytic, random (`drl.py:753-764`, `794-815`) and accepts only improvements (`drl.py:817-823`). This is a major semantic mismatch. A literal igraph port should preserve the old-energy/analytic-position pairing unless evidence shows igraph's behavior is patched elsewhere.

### Schedule Timing

igraph:

- Stage control happens after `update_nodes()` (`drl_graph.cpp:610-611`, `624-808`).
- Liquid sets temperature/attraction/damping after the first init sweep (`drl_graph.cpp:624-657`).
- Expansion decreases attraction by `.05`, `min_edges` by `.05`, `cut_off_length` by `cut_rate`, and damping by `.005` before the next expansion update because of fall-through from stage 0 (`drl_graph.cpp:659-675`).
- Cooldown decreases temperature by `10`, cut length by `cut_rate*2`, and min edges by `.2` after the first cooldown update (`drl_graph.cpp:696-715`).
- Crunch has no intra-phase cooling (`drl_graph.cpp:738-760`).
- Simmer decreases temperature by `2` after each simmer update while above `50` (`drl_graph.cpp:763-785`).

Dagua:

- Phase control happens inside the phase loop before node updates (`drl.py:914-959`).
- Expansion decrements before every expansion sweep (`drl.py:916-933`), matching igraph's first expansion update after fall-through but not necessarily all boundary behavior.
- Cooldown decrements before the first cooldown sweep (`drl.py:934-950`), while igraph's first cooldown sweep uses the initial cooldown temperature (`drl_graph.cpp:686-688`, then next `update_nodes()` before `701-712`).
- Simmer decrements before the first simmer sweep (`drl.py:951-958`), while igraph's first simmer sweep uses `simmer.temperature` (`drl_graph.cpp:749-755`, then next `update_nodes()` before `766-770`).
- Dagua clamps cut length with `max(cut_end, ...)` (`drl.py:927-945`); igraph subtracts in expansion without an immediate clamp (`drl_graph.cpp:671`) and only guards cooldown subtraction with `cut_off_length > cut_length_end` before subtracting (`drl_graph.cpp:706-709`), which can overshoot.

## RNG

igraph:

- The legacy `rand_seed` is ignored in `init_parms(int rand_seed, ...)` (`drl_graph.cpp:376-378`) and the `srand()` call is commented as unnecessary in igraph (`drl_graph.cpp:428-430`).
- `init_parms(const options*)` passes `rand_seed=0.0`, not the user seed (`drl_graph.cpp:433-437`).
- Random moves use igraph's global RNG via `RNG_UNIF01()` twice per node update (`drl_graph.cpp:939-941`).
- Initial positions default to `Node` constructor `(0,0)` unless `use_seed` calls `read_real(res)` (`drl_Node.h:60-63`, `drl_layout.cpp:473-476`).

Dagua:

- Initializes positions with Python `random.Random(seed).random()` into `[0,1)` coordinates (`drl.py:390-407`).
- Seeds a separate Python RNG with the same `problem.seed` for perturbations (`drl.py:1161`).
- Uses `rng.uniform(-0.5, 0.5)` twice per node update (`drl.py:782-792`).
- Public wrapper default seed is `42` (`pipelines/drl.py:49-56`).

Impact: default unseeded igraph DRL starts from zeros unless the caller supplies seed coordinates, then uses igraph's global RNG for perturbations. Dagua always random-initializes positions from Python RNG. This is a top-tier divergence unless the comparator intentionally passes seed coordinates into igraph. Round 14 did not resolve this; it only noted stochastic residuals (`SUMMARY.md:35-62`).

## Hyperparameter Alignment

| Hyperparameter | igraph | dagua | Alignment |
| --- | --- | --- | --- |
| `edge_cut` default | `32.0/40.0` (`drl_layout.cpp:243`) | `32.0/40.0` (`drl.py:178`, etc.) | Match |
| `initial_min_edges` | `20` (`drl_graph.cpp:137`) | `20.0` (`drl.py:1094`, `1169`) | Match |
| `cut_base` | `40000.0 * (1-edge_cut)` (`drl_graph.cpp:379-381`) | `40000.0 * (1-edge_cut)` (`drl.py:1098`, `1162`) | Match |
| `cut_length_end` floor | min `1.0` (`drl_graph.cpp:383-386`) | no floor when `cut_end <= 0.0` (`drl.py:1162-1168`) | Diverges for extreme edge_cut |
| `cut_off_start` | `4.0 * cut_length_end` (`drl_graph.cpp:388-392`) | `4.0 * cut_end` (`drl.py:1095`, `1163`) | Match for default; diverges with floor |
| `cut_rate` | `(start-end)/400 = 3*end/400` (`drl_graph.cpp:388-392`) | `3*cut_end/400` (`drl.py:1096-1097`, `1164-1168`) | Match for default |
| expansion attraction decrement | `.05` until `>1` (`drl_graph.cpp:665-667`) | `0.05` floor `1.0` (`drl.py:917-921`) | Mostly match |
| expansion min_edges decrement | `.05` until `>12` (`drl_graph.cpp:668-670`) | `0.05` floor `12.0` (`drl.py:922-926`) | Mostly match |
| expansion damping decrement | `.005` until `>.1` (`drl_graph.cpp:672-674`) | `0.005` floor `0.1` (`drl.py:929-933`) | Mostly match |
| cooldown temp decrement | `10` after update, until `>50` (`drl_graph.cpp:701-704`) | `10` before update, floor `50` (`drl.py:934-939`) | Timing diverges |
| cooldown cut decrement | subtract `cut_rate*2` if current `> end` (`drl_graph.cpp:706-709`) | `max(cut_end, current - 2*cut_rate)` (`drl.py:940-945`) | Clamp semantics diverge |
| cooldown min_edges decrement | `.2` until `>1` (`drl_graph.cpp:710-712`) | `.2` floor `1.0` (`drl.py:946-950`) | Timing diverges |
| simmer temp decrement | `2` after update, until `>50` (`drl_graph.cpp:766-770`) | `2` before update, floor `50` (`drl.py:951-958`) | Timing diverges |
| attraction factor | `attraction^4 * 2e-2` (`drl_graph.cpp:1009-1012`) | `attraction**4 * 0.02` (`drl.py:599-601`) | Match |
| energy stage powers | STAGE 0: distance^4; STAGE 1: distance^2; stages >=2: distance^1 (`drl_graph.cpp:1028-1037`) | init/liquid power 4, expansion power 2, later power 1 (`drl.py:551-557`, `603-608`) | Stored phase mapping aligns |
| jump length | `.010 * temperature` (`drl_graph.cpp:917-918`) | `0.01 * temperature` (`drl.py:54-61`, `784`) | Match |
| coarse density constants | `1000`, `4000`, `10` (`drl_layout.h:43-55`) | `1000`, `4000`, `10` (`drl.py:22-39`) | Match |
| coarse density kernel | separable product (`DensityGrid.cpp:81-85`) | radial cone (`drl.py:430-433`) | Diverges |
| fine density guard | `1e-50` (`DensityGrid.cpp:123-125`) | `1e-12` (`drl.py:47-60`, `545-548`) | Diverges |
| node order | ascending for `MAX_PROCS=1` (`drl_graph.cpp:824-833`, `871-878`) | ascending `range(num_nodes)` (`drl.py:960-973`) | Match |
| variants exposed | five variants default/coarsen/coarsest/refine/final all route to `classic_drl` vs `igraph_drl` (`variants.py:1310-1362`) | same | Variant surface aligns |

## Ranked Fix List

1. **Match igraph's effective `ReCompute()` sweep schedule.** Add the init-parameter sweep, boundary sweeps, and final stage6 sweep, or rewrite the phase runner around igraph's stage-control order. This affects every DRL variant.
2. **Fix igraph candidate semantics literally.** The code appears to compare old-coordinate energy against random-coordinate energy, then accept analytic coordinate if old energy wins (`drl_graph.cpp:923-929`, `943-947`, `964-973`). Round 14 tested a partial version but not this exact behavior.
3. **Rework fine-density lifecycle.** Separate coarse density grid from fine bins and leave fine bins empty until `fineDensity` starts; fill progressively like igraph (`DensityGrid.cpp:141-156`, `247-255`).
4. **Use igraph's separable density kernel.** Replace radial cone with product falloff (`DensityGrid.cpp:81-85` vs `drl.py:430-433`).
5. **Implement igraph cell conversion and boundary penalty.** Include `+0.5`, `VIEW_TO_GRID`, the 10-cell `10000` penalty, and throw/guard behavior instead of clamp/truncate (`DensityGrid.cpp:101-110`, `168-177`, `202-215`).
6. **Make edge cutting asymmetric.** Erase only `adjacency[node][worst]`, not the reciprocal entry (`drl_graph.cpp:1130-1133` vs `drl.py:665-667`).
7. **Fix edge-cut score degree factor.** Use current-node `sqrt(len(neighbors[node]))`, not candidate neighbor degree (`drl_graph.cpp:1104-1122` vs `drl.py:657-660`).
8. **Align `REFINE` preset.** Set init damping to `0.0` and cooldown temperature to `200.0` (`drl_layout.cpp:343-361` vs `drl.py:204-211`).
9. **Align `FINAL` preset.** Set expansion to `(50, 50.0, 0.1, 0.25)` (`drl_layout.cpp:385-388` vs `drl.py:217`).
10. **Investigate initialization contract in the comparator.** igraph defaults to zero positions unless `use_seed` is true (`drl_Node.h:60-63`, `drl_layout.cpp:473-476`); dagua random-initializes (`drl.py:390-407`). The intended fidelity target must decide whether dagua should accept external seed coordinates or mimic igraph's default zero start.
11. **Align RNG source/order.** Python `random.Random` cannot reproduce `RNG_UNIF01()` sequences (`drl_graph.cpp:939-941` vs `drl.py:782-792`). Exact fidelity requires comparator-level seeding or an igraph-compatible RNG stream.
12. **Match cooldown/simmer decrement timing and clamp semantics.** igraph decrements after sweeps; dagua decrements before sweeps (`drl_graph.cpp:696-770` vs `drl.py:934-958`).
13. **Match cut-end edge cases.** Add `cut_length_end >= 1.0` and `CUT_END >= 39500` cutting disable behavior (`drl_graph.cpp:381-386`, `1099-1102`).

## Recommended Round 20 Fix Scope

Recommended scope should be staged, because DRL has many coupled trajectory controls.

1. **First patch: schedule semantics only.** Port `ReCompute()` stage-control order into dagua so effective sweep counts match igraph, without touching density or edge cutting. Add tests that count per-stage sweeps for all five templates.
2. **Second patch: exact candidate acceptance.** Implement the literal igraph old-energy/analytic-position behavior and rerun the Round 14 small comparator. This is narrower than density and directly affects every node update.
3. **Third patch: preset table fixes for `REFINE` and `FINAL`.** These are unambiguous data fixes and should have targeted variant tests.
4. **Fourth patch: edge cutting.** Change asymmetric removal and current-node degree factor together; evaluate because this is likely invasive.
5. **Fifth patch: density-grid fidelity.** Split coarse density and fine bins, switch to the separable kernel, add boundary behavior, and test at the unit level before live compare.

Do **not** combine all of these in one Round 20 patch. The first recommended Round 20 deliverable is schedule + exact candidate semantics + preset fixes, with live compare before density-grid surgery.

## Assumptions

- I treated igraph `drl_graph.cpp` as the source of truth for per-node behavior because this checkout has no `drl_Node.cpp`.
- The effective sweep counts above assume `MAX_PROCS=1`, which is the compile-time igraph setting in `drl_layout.h:38-40`.
- I did not run comparators or modify implementation code; this is diagnosis-only.

## Dead Code / Removable Code

None identified as safely removable within this diagnosis scope.
