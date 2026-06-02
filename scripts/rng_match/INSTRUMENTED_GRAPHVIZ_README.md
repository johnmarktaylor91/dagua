# Instrumented Graphviz 7.0.5

This directory contains the reproducible build assets for the permanent logging-only Graphviz reference at:

`~/tools/graphviz-7.0.5-instr/`

The source tree is:

`~/tools/graphviz-7.0.5-src/`

## Build

Run from the dagua repository root:

```bash
scripts/rng_match/build_instrumented_graphviz.sh
```

The script verifies the official 7.0.5 release tarball checksum, then builds from the matching GitLab `7.0.5` tag archive because the release tarball does not include CMake project files.

The local CMake recipe needs conda's GTS package for `sfdp` triangulation. Graphviz CMake also tries to build/install unrelated `gvmap` tools; those fail to link against GTS/rbtree in this environment, after the required `dot_builtins` binary and layout plugins are built. The script treats that as a known auxiliary-tool failure only after checking the required `dot_builtins`, `libgvplugin_neato_layout.so`, and `libgvplugin_core.so` artifacts exist.

## Trace Usage

Default runs are silent:

```bash
~/tools/graphviz-7.0.5-instr/bin/dot -Kfdp -Gseed=1 -Gstart=1 -Tplain path8.dot
```

Enable trace rows on stderr:

```bash
GV_TRACE=1 ~/tools/graphviz-7.0.5-instr/bin/dot -Kfdp -Gseed=1 -Gstart=1 -Tplain path8.dot
```

Write trace rows to a file:

```bash
GV_TRACE=1 GV_TRACE_FILE=/tmp/gv.trace ~/tools/graphviz-7.0.5-instr/bin/dot -Kfdp path8.dot
```

Trace rows use `%.17g` for floating-point values.

## Instrumentation Points

- `lib/fdpgen/tlayout.c`: `STEP` and `POSITION` rows for `fdp` iterations.
- `lib/fdpgen/xlayout.c`: `XLAYOUT` rows for overlap-removal iteration state.
- `lib/fdpgen/layout.c`: `FINALCC` and `FINALCC_COMPONENT` rows around component packing.
- `lib/common/utils.c`: `COMPUTE_BB` rows after per-component bounding-box computation.
- `lib/sfdpgen/spring_electrical.c`: `STEP` and `POSITION` rows for `sfdp` fast/slow spring-electrical iterations.

All logging is guarded by `GV_TRACE`; when unset, the added code returns before opening files or formatting rows.

## Veridical Proof

Stock binary:

`~/anaconda3/envs/py311/bin/dot`

Instrumented binary:

`~/tools/graphviz-7.0.5-instr/bin/dot`

Command shape:

```bash
dot -K{fdp,neato,sfdp} -Gseed=N -Gstart=N -Tplain graph.dot
```

Graphs: `path8`, `star8`, `grid3x3`, `cycle6`, `complete5`, `tree7`.

Seeds: `1`, `2`, `3`.

Result from the final check:

```text
total=54
exact_bytes=54
max_rmsd=0
result=PASS bit-for-bit identical
```

Trace smoke test:

```text
default_trace_rows=0
trace_rows=2776
output_identical_with_trace=yes
```
