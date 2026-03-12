# Visual Reset Brief

This is the short design brief for the next serious visual pass.

It exists so future work starts from the hard-won diagnosis instead of drifting back into vague polishing.

## What To Preserve

- The node-placement engine may be genuinely strong.
- The scalable multilevel architecture is real and valuable.
- The benchmark/visual-audit infrastructure is now strong enough to support disciplined iteration.

The reset should **not** throw away the placement story just because the current renders are weak.

## What Is Currently Failing

This is the blunt diagnosis.

- Text hierarchy is poor.
  - too much text
  - wrong emphasis
  - clutter instead of guidance
- Edge language is weak.
  - not enough semantic clarity
  - not enough visual confidence
  - routing style does not yet feel intentional
- Cluster/container treatment is muddy.
  - boxes feel heavy, decorative, or debug-like rather than structural
- The overall system looks like a themed debug export, not a designed visual language.

## The Reset Principles

The next pass should follow these principles:

1. Placement first.
   - judge node placement separately from styling
   - use `placement_dashboard.md` and benchmark metrics to protect the real asset

2. Minimal baseline first.
   - start from a restrained black/gray visual system
   - prove hierarchy, spacing, and readability before reintroducing stronger visual identity

3. Ruthless text discipline.
   - default view should show only what earns its place
   - labels must establish hierarchy, not sabotage it

4. Edge language must do semantic work.
   - connection type, directionality, skips, recurrence, and emphasis should be visually intentional

5. Clusters must clarify structure, not add soup.
   - containment should be instantly legible
   - nested containers should feel architectural, not ornamental

6. Evaluate stepwise.
   - simplest graphs first
   - then challenge graphs
   - then kitchen-sink graphs
   - then scale / showcase outputs

## Operational Workflow

Use this reset loop:

1. Consult `eval_output/report/placement_dashboard.md`
   - confirm whether the node-placement engine itself changed
2. Rebuild one or two visual-audit graphs only
   - do not start with the whole gallery
3. Compare against:
   - frozen baseline
   - competitor stepwise renders
   - decomposition views
4. Fix one visual language problem at a time
   - typography
   - edge language
   - containers
   - information density
5. Expand outward only after the simple graphs look intentional

## The Core Reminder

Do not confuse:

- weak visual language

with:

- weak layout engine

The next phase is not “make it prettier.”  
It is “design a visual system worthy of the engine.”
