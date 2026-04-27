# Area C — Close-loss + Tie bucket lift analysis

## Question

The "BEST or TIED" share is 88% (82/93). The 11 graphs that don't
dominate but ARE within ±2.0 of best competitor are the highest-
leverage targets — each one that flips from close-loss/tie to
modest-win is +1 strict-dominate without changing anything visible.

For each graph in the close-loss [-2..-0.5] and tie [-0.5..+0.5]
buckets:

1. Identify the dominant losing metric component.
2. Identify what competitor does specifically that we don't.
3. Recommend the lowest-effort change that flips it.
4. Quantify expected delta + risk.

## How to enumerate

Run `/tmp/h2h_buckets_seeded.py` at HEAD = `97286e4` to get the full
bucket. For each graph in close-loss or tie:

```python
# Per-graph score breakdown
torch.manual_seed(0)
m_dagua = composite(full(layout(g, LayoutConfig(seed=42)),
                         g.edge_index, node_sizes=g.node_sizes))
torch.manual_seed(0)
m_comp = composite(full(competitor_pos, g.edge_index,
                        node_sizes=g.node_sizes))
# compare component by component
```

Components ranked by weight: dag_consistency 25, edge_length_cv 20,
depth_spearman 15, overlap 10, edge_straightness 10, crossing_rate 10,
angular_resolution 5, cluster_separation 5.

## Research targets

For each of the ~16 graphs in close-loss + tie:

- Which metric is dagua losing on?
- Which metric is dagua winning on (so the trade is bidirectional)?
- Is the loss in a metric component that polishing/tuning could fix?
- Or is it structural (algorithm choice)?
- What's the SINGLE simplest knob that would lift this graph by ~1pt?

## Output format

`.project-context/research/sprint_21_final_push/C_close_loss_lift__<your_agent_name>.md`

Include:
- TL;DR with the top 3 lowest-cost lifts and what each delivers
- Per-graph breakdown table:
  | graph | dagua | comp | delta | losing-metric | comp-strategy | recommendation | est-delta |
- Cluster the recommendations: which of the 16 graphs flip together
  with one knob? (E.g. "all 4 graphs lose on crossing_rate -- a
  smarter crossing reduction op flips them all")
- Risk: which protected wins are at risk for each recommendation

## Constraints

- READ-ONLY. Findings file only.
- Read `.project-context/research/sprint_21_final_push/CONTEXT.md` first.
- This is a SURVEY task -- breadth over depth. Hit all 16 graphs.
- Budget: 2000-3500 words for the full survey.
