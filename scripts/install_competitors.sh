#!/usr/bin/env bash
# Install all dependencies for dagua's competitor layout engines.
# Run with: sudo bash scripts/install_competitors.sh
# (sudo only needed for apt packages; pip/npm run as current user)
set -euo pipefail

echo "=== Installing dagua competitor dependencies ==="

# ─── System packages (apt) ───────────────────────────────────────────────────
echo ""
echo "--- System packages (requires sudo) ---"

# Graphviz: dot, neato, fdp, sfdp
apt-get install -y graphviz

# Node.js: needed for dagre + ELK (via subprocess)
if ! command -v node &>/dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
else
    echo "Node.js already installed: $(node --version)"
fi

# Build tools for native Python packages (igraph, scipy)
apt-get install -y build-essential cmake pkg-config

echo ""
echo "--- System packages done ---"

# ─── Python packages (pip) ───────────────────────────────────────────────────
echo ""
echo "--- Python packages (current user/env) ---"

# Core competitor dependencies
pip install --quiet networkx          # NetworkX spring + kamada_kawai layouts
pip install --quiet igraph             # igraph FR, sugiyama, DrL, GEM, Davidson-Harel
pip install --quiet pydot             # DOT file parsing for Graphviz interop
pip install --quiet scipy             # Sparse matrix ops, MDS, eigensolvers
pip install --quiet scikit-learn      # t-SNE, MDS for reference comparison tests

# Reference implementations for classic algorithm validation
pip install --quiet s-gd2             # Stress-SGD reference (Zheng 2018, C++ core)
pip install --quiet fa2-modified      # ForceAtlas2 reference (maintained fork)
pip install --quiet umap-learn        # UMAP embedding for graph layout
pip install --quiet ogdf-python       # OGDF bindings: GEM, LinLog, FM³, Maxent-Stress, DH

# Optional accelerators
pip install --quiet numba             # JIT for coarsening matching loop (~50x faster)

echo ""
echo "--- Python packages done ---"

# ─── Node.js packages (npm) ──────────────────────────────────────────────────
echo ""
echo "--- Node.js packages (global) ---"

# dagre: JavaScript DAG layout (used via subprocess)
npm install -g dagre 2>/dev/null || npm install -g dagre

# elkjs: Eclipse Layout Kernel for JavaScript (used via subprocess)
npm install -g elkjs 2>/dev/null || npm install -g elkjs

# Mermaid CLI: documentation-diagram reference renderer used by the
# cosmetic-feature gallery audit harness.
npm install -g @mermaid-js/mermaid-cli 2>/dev/null || npm install -g @mermaid-js/mermaid-cli

echo ""
echo "--- Node.js packages (project dev dependencies) ---"

# Tier-B cosmetic reference renderers. Keep these local to the project so
# adapter subprocesses can resolve them with Node's normal module lookup.
npm install --save-dev cytosnap d3 jsdom d3-graphviz canvas

echo ""
echo "--- Node.js packages done ---"

# ─── Java libraries (current user) ──────────────────────────────────────────
echo ""
echo "--- Java libraries (current user) ---"

GEPHI_TOOLKIT_DIR="${HOME}/.local/share/gephi-toolkit"
GEPHI_TOOLKIT_JAR="${GEPHI_TOOLKIT_DIR}/gephi-toolkit-0.10.0-all.jar"
mkdir -p "${GEPHI_TOOLKIT_DIR}"
if [[ -s "${GEPHI_TOOLKIT_JAR}" ]]; then
    echo "Gephi Toolkit already installed: ${GEPHI_TOOLKIT_JAR}"
else
    curl -fL --retry 1 \
        -o "${GEPHI_TOOLKIT_JAR}" \
        "https://repo1.maven.org/maven2/org/gephi/gephi-toolkit/0.10.0/gephi-toolkit-0.10.0-all.jar" \
        || rm -f "${GEPHI_TOOLKIT_JAR}"
fi

echo ""
echo "--- Java libraries done ---"

# ─── Verify installations ───────────────────────────────────────────────────
echo ""
echo "=== Verification ==="

echo -n "Graphviz (dot):    " && (dot -V 2>&1 | head -1) || echo "MISSING"
echo -n "Node.js:           " && (node --version) || echo "MISSING"
echo -n "dagre:             " && (node -e "require('dagre'); console.log('OK')" 2>/dev/null) || echo "MISSING"
echo -n "elkjs:             " && (node -e "require('elkjs'); console.log('OK')" 2>/dev/null) || echo "MISSING"
echo -n "Mermaid CLI:       " && (mmdc -h >/dev/null 2>&1 && echo "OK") || echo "MISSING"
echo -n "cytosnap:          " && (node -e "require('cytosnap'); console.log('OK')" 2>/dev/null) || echo "MISSING"
echo -n "d3/jsdom/canvas:   " && (node -e "require('d3'); require('jsdom'); require('canvas'); console.log('OK')" 2>/dev/null) || echo "MISSING"
echo -n "Gephi Toolkit:     " && ([[ -s "${GEPHI_TOOLKIT_JAR}" ]] && echo "${GEPHI_TOOLKIT_JAR}") || echo "MISSING"

echo ""
python3 -c "
deps = {
    'networkx': 'networkx',
    'igraph': 'igraph',
    'pydot': 'pydot',
    'scipy': 'scipy',
    'sklearn': 'sklearn',
    's_gd2': 's_gd2',
    'fa2': 'fa2',
    'umap': 'umap',
    'ogdf-python': 'ogdf',
    'numba': 'numba',
}
for name, module in deps.items():
    try:
        __import__(module)
        print(f'{name:18s} OK')
    except ImportError:
        print(f'{name:18s} MISSING')
"

echo ""
echo "=== Done. All competitor engines should now be available. ==="
echo ""
echo "To verify: python -c 'from dagua.eval.competitors import get_available_competitors; print([c.name for c in get_available_competitors()])'"
