"""JUNG ISOM competitor adapter via a Java subprocess."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.error import URLError
from urllib.request import urlretrieve

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_runtime_seed, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_SEED = 42
_DEFAULT_STEPS = 2000
_DEFAULT_WIDTH = 600
_DEFAULT_HEIGHT = 600
_JUNG_CACHE_DIR = Path("/tmp/dagua-isom-jung")
_JUNG_VERSION = "2.1.1"
_GUAVA_VERSION = "19.0"
_RUNNER_CLASS = "IsomReferenceRunner"
_RUNNER_SOURCE = _JUNG_CACHE_DIR / f"{_RUNNER_CLASS}.java"
_RUNNER_CLASS_FILE = _JUNG_CACHE_DIR / f"{_RUNNER_CLASS}.class"
_ARTIFACT_URLS = {
    "jung-algorithms-2.1.1.jar": (
        "https://repo1.maven.org/maven2/net/sf/jung/jung-algorithms/"
        f"{_JUNG_VERSION}/jung-algorithms-{_JUNG_VERSION}.jar"
    ),
    "jung-api-2.1.1.jar": (
        f"https://repo1.maven.org/maven2/net/sf/jung/jung-api/{_JUNG_VERSION}/"
        f"jung-api-{_JUNG_VERSION}.jar"
    ),
    "jung-graph-impl-2.1.1.jar": (
        "https://repo1.maven.org/maven2/net/sf/jung/jung-graph-impl/"
        f"{_JUNG_VERSION}/jung-graph-impl-{_JUNG_VERSION}.jar"
    ),
    "guava-19.0.jar": (
        f"https://repo1.maven.org/maven2/com/google/guava/guava/{_GUAVA_VERSION}/"
        f"guava-{_GUAVA_VERSION}.jar"
    ),
}
_JAVA_SOURCE = r"""
import edu.uci.ics.jung.algorithms.layout.ISOMLayout;
import edu.uci.ics.jung.algorithms.layout.util.RandomLocationTransformer;
import edu.uci.ics.jung.graph.SparseMultigraph;
import java.awt.Dimension;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.util.Locale;
import java.util.Random;

public class IsomReferenceRunner {
  static void seedMathRandom(long seed) throws Exception {
    Class<?> holder = Class.forName("java.lang.Math$RandomNumberGeneratorHolder");
    Field field = holder.getDeclaredField("randomNumberGenerator");
    field.setAccessible(true);
    Random random = (Random) field.get(null);
    random.setSeed(seed);
  }

  public static void main(String[] args) throws Exception {
    BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
    String header = reader.readLine();
    if (header == null) {
      throw new IllegalArgumentException("missing header");
    }
    String[] parts = header.trim().split("\\s+");
    int n = Integer.parseInt(parts[0]);
    int steps = Integer.parseInt(parts[1]);
    long seed = Long.parseLong(parts[2]);
    int width = Integer.parseInt(parts[3]);
    int height = Integer.parseInt(parts[4]);
    SparseMultigraph<Integer, Integer> graph = new SparseMultigraph<Integer, Integer>();
    for (int index = 0; index < n; index++) {
      graph.addVertex(index);
    }
    int edgeId = 0;
    String line;
    while ((line = reader.readLine()) != null) {
      line = line.trim();
      if (line.isEmpty()) {
        continue;
      }
      String[] edge = line.split("\\s+");
      graph.addEdge(edgeId, Integer.parseInt(edge[0]), Integer.parseInt(edge[1]));
      edgeId++;
    }
    ISOMLayout<Integer, Integer> layout = new ISOMLayout<Integer, Integer>(graph);
    layout.setSize(new Dimension(width, height));
    layout.setInitializer(new RandomLocationTransformer<Integer>(layout.getSize(), seed));
    seedMathRandom(seed);
    for (int step = 0; step < steps; step++) {
      layout.step();
    }
    for (int index = 0; index < n; index++) {
      System.out.printf(
          Locale.ROOT,
          "%d %.17g %.17g%n",
          index,
          layout.getX(index),
          layout.getY(index));
    }
  }
}
"""


def _classpath() -> str:
    """Return the JUNG runner classpath.

    Returns
    -------
    str
        Platform classpath containing the cache directory and required jars.
    """
    paths = [_JUNG_CACHE_DIR]
    paths.extend(_JUNG_CACHE_DIR / filename for filename in _ARTIFACT_URLS)
    return ":".join(str(path) for path in paths)


def _ensure_artifacts() -> Optional[str]:
    """Download JUNG runtime artifacts into the temporary cache.

    Returns
    -------
    str | None
        Error message when a download fails, otherwise ``None``.
    """
    _JUNG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in _ARTIFACT_URLS.items():
        target = _JUNG_CACHE_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        try:
            urlretrieve(url, target)
        except (OSError, URLError) as exc:
            return f"failed to download {filename}: {exc}"
    return None


def _ensure_runner() -> Optional[str]:
    """Compile the Java JUNG ISOM runner if needed.

    Returns
    -------
    str | None
        Error message when Java, jars, or compilation are unavailable;
        otherwise ``None``.
    """
    artifact_error = _ensure_artifacts()
    if artifact_error is not None:
        return artifact_error

    _RUNNER_SOURCE.write_text(_JAVA_SOURCE, encoding="utf-8")
    if (
        _RUNNER_CLASS_FILE.exists()
        and _RUNNER_CLASS_FILE.stat().st_mtime >= _RUNNER_SOURCE.stat().st_mtime
    ):
        return None
    try:
        result = subprocess.run(
            ["javac", "-classpath", _classpath(), str(_RUNNER_SOURCE)],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode != 0:
        return result.stderr[:500]
    return None


def _build_runner_input(
    graph: DaguaGraph,
    steps: int,
    seed: int,
    width: int,
    height: int,
) -> str:
    """Build the line-oriented Java runner input.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    steps : int
        Number of JUNG ``step()`` calls.
    seed : int
        Java ``Random`` seed for deterministic initialization and training points.
    width : int
        JUNG layout width.
    height : int
        JUNG layout height.

    Returns
    -------
    str
        Header plus one ``source target`` line per edge.
    """
    lines = [f"{graph.num_nodes} {int(steps)} {int(seed)} {int(width)} {int(height)}"]
    if graph.edge_index.numel() > 0:
        for edge_pos in range(graph.edge_index.shape[1]):
            source = int(graph.edge_index[0, edge_pos].item())
            target = int(graph.edge_index[1, edge_pos].item())
            lines.append(f"{source} {target}")
    return "\n".join(lines) + "\n"


@register
class IsomCompetitor(CompetitorBase):
    """Run JUNG ``ISOMLayout`` through a reproducible Java subprocess."""

    name = "isom_jung"
    max_nodes = 5_000
    supports_clusters = False
    variant_param_names = frozenset({"steps", "width", "height"})

    def available(self) -> bool:
        """Check whether the Java JUNG runner can be compiled.

        Returns
        -------
        bool
            ``True`` when Java and the required jars are available.
        """
        return _ensure_runner() is None

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run JUNG ISOM with default source parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int | None, default=None
            Java ``Random`` seed. ``None`` resolves to the benchmark runtime
            seed or ``42``.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[dict[str, object]] = None,
    ) -> CompetitorResult:
        """Run JUNG ISOM with optional runner parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int | None, default=None
            Java ``Random`` seed.
        variant_params : dict[str, object], optional
            Optional ``steps``, ``width``, and ``height`` overrides.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        runner_error = _ensure_runner()
        if runner_error is not None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=0.0,
                error=runner_error,
            )

        params = {} if variant_params is None else dict(variant_params)
        resolved_seed = get_runtime_seed(_DEFAULT_SEED) if seed is None else seed
        input_data = _build_runner_input(
            graph=graph,
            steps=int(params.get("steps", _DEFAULT_STEPS)),
            seed=_DEFAULT_SEED if resolved_seed is None else int(resolved_seed),
            width=int(params.get("width", _DEFAULT_WIDTH)),
            height=int(params.get("height", _DEFAULT_HEIGHT)),
        )

        start = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    "java",
                    "--add-opens",
                    "java.base/java.lang=ALL-UNNAMED",
                    "-classpath",
                    _classpath(),
                    _RUNNER_CLASS,
                ],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
            if result.returncode != 0:
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=elapsed,
                    error=result.stderr[:500],
                )
            pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) != 3:
                    continue
                index = int(parts[0])
                pos[index, 0] = float(parts[1])
                pos[index, 1] = float(parts[2])
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except subprocess.TimeoutExpired:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error="timeout",
            )
        except (FileNotFoundError, ValueError) as exc:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error=str(exc),
            )
