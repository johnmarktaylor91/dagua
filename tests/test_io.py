"""Tests for io.py — graph_from_json, graph_to_json, YAML, unified load/save, LLM-based construction."""

import json
import os
import tempfile
from unittest import mock

import pytest
import yaml

from dagua.graph import DaguaGraph
from dagua.io import (
    ImageAIConfig,
    _dict_to_cluster_style,
    _dict_to_edge_style,
    _dict_to_node_style,
    _extract_json_from_response,
    _get_llm_client,
    _prepare_image_for_llm,
    configure_image_ai,
    get_image_ai_config,
    graph_code_from_image,
    graph_dict_from_image,
    graph_script_from_dict,
    graph_from_image,
    graph_from_json,
    graph_from_yaml,
    graph_to_json,
    graph_to_yaml,
    load,
    save,
    theme_code_from_image,
    theme_dict_from_image,
    theme_from_image,
)
from dagua.styles import (
    ClusterStyle, EdgeStyle, GraphStyle, NodeStyle, Theme,
    DARK_THEME, DEFAULT_THEME_OBJ, get_theme,
)


# ─── TestGraphFromJson ─────────────────────────────────────────────────────


@pytest.mark.smoke
class TestGraphFromJson:
    """graph_from_json: dict, string, file, styles, clusters."""

    def test_basic(self):
        data = {
            "nodes": [
                {"id": "a", "label": "Node A"},
                {"id": "b", "label": "Node B"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        g = graph_from_json(data)
        assert g.num_nodes == 2
        assert g.node_labels == ["Node A", "Node B"]
        assert g.edge_index.shape == (2, 1)

    def test_with_styles(self):
        data = {
            "nodes": [
                {
                    "id": "x",
                    "label": "Styled",
                    "style": {"shape": "ellipse", "base_color": "#D55E00"},
                }
            ],
            "edges": [
                {
                    "source": "x",
                    "target": "y",
                    "style": {"color": "#FF0000", "style": "dashed"},
                }
            ],
        }
        g = graph_from_json(data)
        assert g.node_styles[0].shape == "ellipse"
        assert g.node_styles[0].base_color == "#D55E00"
        assert g.edge_styles[0].color == "#FF0000"
        assert g.edge_styles[0].style == "dashed"

    def test_clusters(self):
        data = {
            "nodes": [
                {"id": "a"},
                {"id": "b"},
                {"id": "c"},
            ],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}],
            "clusters": [
                {
                    "name": "group1",
                    "members": ["a", "b"],
                    "label": "Group 1",
                    "style": {"fill": "#E5E5E0"},
                }
            ],
        }
        g = graph_from_json(data)
        assert "group1" in g.clusters
        assert g.cluster_labels["group1"] == "Group 1"
        assert "group1" in g.cluster_styles
        assert g.cluster_styles["group1"].fill == "#E5E5E0"

    def test_direction(self):
        data = {"direction": "LR", "nodes": [{"id": "a"}]}
        g = graph_from_json(data)
        assert g.direction == "LR"

    def test_theme(self):
        data = {
            "nodes": [{"id": "a", "type": "input"}],
            "theme": {
                "default": {"base_color": "#56B4E9"},
                "input": {"base_color": "#009E73"},
            },
        }
        g = graph_from_json(data)
        assert isinstance(g._theme, Theme)
        assert g._theme.get_node_style("input").base_color == "#009E73"
        assert g._theme.get_node_style("default").base_color == "#56B4E9"

    def test_missing_optional_fields(self):
        """Minimal valid JSON — only node IDs."""
        data = {"nodes": [{"id": "a"}, {"id": "b"}]}
        g = graph_from_json(data)
        assert g.num_nodes == 2
        assert g.node_labels == ["a", "b"]  # ID used as label
        assert g.edge_index.shape[1] == 0

    def test_empty_graph(self):
        g = graph_from_json({})
        assert g.num_nodes == 0

    def test_unknown_style_keys_ignored(self):
        data = {
            "nodes": [
                {"id": "a", "style": {"shape": "rect", "bogus_key": 42, "another": "x"}}
            ]
        }
        g = graph_from_json(data)
        assert g.node_styles[0].shape == "rect"

    def test_json_string_input(self):
        data = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
        g = graph_from_json(json.dumps(data))
        assert g.num_nodes == 2

    def test_file_path_input(self):
        data = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            g = graph_from_json(path)
            assert g.num_nodes == 2
        finally:
            os.unlink(path)

    def test_node_types(self):
        data = {
            "nodes": [
                {"id": "a", "type": "input"},
                {"id": "b", "type": "output"},
                {"id": "c"},
            ]
        }
        g = graph_from_json(data)
        assert g.node_types == ["input", "output", "default"]

    def test_edge_labels(self):
        data = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b", "label": "flow"}],
        }
        g = graph_from_json(data)
        assert g.edge_labels == ["flow"]

    def test_padding_list_to_tuple(self):
        data = {
            "nodes": [{"id": "a", "style": {"padding": [10, 5]}}]
        }
        g = graph_from_json(data)
        assert g.node_styles[0].padding == (10, 5)


# ─── TestGraphToJson ───────────────────────────────────────────────────────


@pytest.mark.smoke
class TestGraphToJson:
    """graph_to_json: roundtrip and serialization."""

    def test_roundtrip_basic(self):
        original = {
            "nodes": [
                {"id": "a", "label": "Node A"},
                {"id": "b", "label": "Node B"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        g = graph_from_json(original)
        result = graph_to_json(g)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["edges"][0]["source"] == "a"
        assert result["edges"][0]["target"] == "b"

    def test_roundtrip_with_styles(self):
        original = {
            "nodes": [
                {"id": "x", "style": {"shape": "ellipse", "base_color": "#D55E00"}},
            ],
            "edges": [
                {"source": "x", "target": "y", "style": {"style": "dashed"}},
            ],
        }
        g = graph_from_json(original)
        result = graph_to_json(g)

        # Node style should include shape override
        x_node = [n for n in result["nodes"] if n["id"] == "x"][0]
        assert x_node["style"]["shape"] == "ellipse"

        # Edge style should include dashed
        assert result["edges"][0]["style"]["style"] == "dashed"

    def test_roundtrip_with_clusters(self):
        original = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}],
            "clusters": [
                {"name": "grp", "members": ["a", "b"], "label": "Group"}
            ],
        }
        g = graph_from_json(original)
        result = graph_to_json(g)

        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["name"] == "grp"
        assert result["clusters"][0]["label"] == "Group"
        assert set(result["clusters"][0]["members"]) == {"a", "b"}

    def test_direction_preserved(self):
        g = graph_from_json({"direction": "RL", "nodes": [{"id": "a"}]})
        result = graph_to_json(g)
        assert result["direction"] == "RL"

    def test_default_direction_omitted(self):
        g = graph_from_json({"nodes": [{"id": "a"}]})
        result = graph_to_json(g)
        assert "direction" not in result

    def test_json_dumps_works(self):
        g = graph_from_json({"nodes": [{"id": "a"}], "edges": []})
        result = graph_to_json(g)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["nodes"][0]["id"] == "a"


# ─── TestExtractJsonFromResponse ───────────────────────────────────────────


@pytest.mark.smoke
class TestExtractJsonFromResponse:
    """_extract_json_from_response: various LLM output formats."""

    def test_pure_json(self):
        text = '{"nodes": [{"id": "a"}]}'
        result = _extract_json_from_response(text)
        assert result["nodes"][0]["id"] == "a"

    def test_code_fenced(self):
        text = 'Here is the graph:\n```json\n{"nodes": [{"id": "b"}]}\n```'
        result = _extract_json_from_response(text)
        assert result["nodes"][0]["id"] == "b"

    def test_code_fenced_no_lang(self):
        text = '```\n{"nodes": [{"id": "c"}]}\n```'
        result = _extract_json_from_response(text)
        assert result["nodes"][0]["id"] == "c"

    def test_preamble_text(self):
        text = 'Sure! Here is the JSON:\n\n{"nodes": [{"id": "d"}], "edges": []}'
        result = _extract_json_from_response(text)
        assert result["nodes"][0]["id"] == "d"

    def test_invalid_raises_valueerror(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            _extract_json_from_response("This is not JSON at all")


# ─── TestStyleConverters ──────────────────────────────────────────────────


@pytest.mark.smoke
class TestStyleConverters:
    """_dict_to_*_style helpers."""

    def test_node_style(self):
        s = _dict_to_node_style({"shape": "ellipse", "font_size": 12.0})
        assert s.shape == "ellipse"
        assert s.font_size == 12.0

    def test_edge_style(self):
        s = _dict_to_edge_style({"color": "#FF0000", "width": 2.0})
        assert s.color == "#FF0000"
        assert s.width == 2.0

    def test_cluster_style(self):
        s = _dict_to_cluster_style({"fill": "#AABBCC", "padding": 20.0})
        assert s.fill == "#AABBCC"
        assert s.padding == 20.0

    def test_unknown_keys_filtered(self):
        s = _dict_to_node_style({"shape": "rect", "unknown_field": True})
        assert s.shape == "rect"
        assert not hasattr(s, "unknown_field")


# ─── TestGraphFromImage (mock-based) ──────────────────────────────────────


class TestGraphFromImage:
    """graph_from_image with mocked LLM calls."""

    def _mock_image_path(self):
        """Create a temp PNG file for testing."""
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG-like header
        f.close()
        return f.name

    def test_returns_dagua_graph(self):
        image_path = self._mock_image_path()
        mock_response = json.dumps({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })

        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock())
                g = graph_from_image(image_path, provider="anthropic")

            assert isinstance(g, DaguaGraph)
            assert g.num_nodes == 2
            assert g.node_labels == ["A", "B"]
        finally:
            os.unlink(image_path)

    def test_graph_dict_from_image(self):
        image_path = self._mock_image_path()
        mock_response = json.dumps({
            "direction": "LR",
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b", "label": "flows to"}],
        })
        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock(), ImageAIConfig(provider="anthropic", model="x", api_key="k"))
                result = graph_dict_from_image(image_path, provider="anthropic")

            assert result["direction"] == "LR"
            assert result["edges"][0]["label"] == "flows to"
        finally:
            os.unlink(image_path)

    def test_graph_code_from_image_returns_best_practice_builder(self):
        image_path = self._mock_image_path()
        mock_response = json.dumps({
            "direction": "TB",
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock(), ImageAIConfig(provider="anthropic", model="x", api_key="k"))
                code = graph_code_from_image(image_path, provider="anthropic")

            assert "def build_graph() -> DaguaGraph:" in code
            assert "g = DaguaGraph(direction='TB')" in code
            assert "g.add_node('a', label='A')" in code
            assert "graph = build_graph()" in code
        finally:
            os.unlink(image_path)

    def test_graph_code_from_image_can_return_ready_to_run_script(self):
        image_path = self._mock_image_path()
        mock_response = json.dumps({
            "direction": "TB",
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock(), ImageAIConfig(provider="anthropic", model="x", api_key="k"))
                code = graph_code_from_image(
                    image_path,
                    provider="anthropic",
                    include_demo_script=True,
                    output_path="magic.png",
                )

            assert "if __name__ == '__main__':" in code
            assert "dagua.draw(graph, config, output='magic.png')" in code
            assert "device='cuda' if __import__('torch').cuda.is_available() else 'cpu'" in code
        finally:
            os.unlink(image_path)

    def test_graph_script_from_dict(self):
        code = graph_script_from_dict(
            {
                "direction": "LR",
                "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                "edges": [{"source": "a", "target": "b"}],
            },
            output_path="demo.png",
        )
        assert "def build_graph() -> DaguaGraph:" in code
        assert "dagua.draw(graph, config, output='demo.png')" in code

    def test_provider_auto_detection_anthropic(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with mock.patch("dagua.io.anthropic", create=True) as mock_anthropic:
                mock_client = mock.MagicMock()
                mock_anthropic.Anthropic.return_value = mock_client
                # Need to mock the import
                import sys
                sys.modules["anthropic"] = mock_anthropic

                try:
                    provider, client, resolved = _get_llm_client()
                    assert provider == "anthropic"
                    assert resolved.api_key == "test-key"
                finally:
                    del sys.modules["anthropic"]

    def test_provider_auto_detection_openai(self):
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=False,
        ):
            # Remove ANTHROPIC_API_KEY if present
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with mock.patch.dict(os.environ, env, clear=True):
                import sys
                mock_openai = mock.MagicMock()
                sys.modules["openai"] = mock_openai

                try:
                    provider, client, resolved = _get_llm_client()
                    assert provider == "openai"
                    assert resolved.api_key == "test-key"
                finally:
                    del sys.modules["openai"]

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No image AI provider configured"):
                _get_llm_client()

    def test_missing_sdk_raises(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=False):
            # Ensure anthropic is not importable
            import sys
            saved = sys.modules.pop("anthropic", None)
            try:
                with mock.patch.dict(sys.modules, {"anthropic": None}):
                    with pytest.raises(ImportError, match="anthropic SDK not installed"):
                        _get_llm_client("anthropic")
            finally:
                if saved is not None:
                    sys.modules["anthropic"] = saved

    def test_explicit_api_key_beats_environment(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            import sys
            mock_openai = mock.MagicMock()
            sys.modules["openai"] = mock_openai
            try:
                _, _, resolved = _get_llm_client("openai", api_key="passed-key")
                assert resolved.api_key == "passed-key"
            finally:
                del sys.modules["openai"]

    def test_api_key_env_override(self):
        with mock.patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-key"}, clear=False):
            import sys
            mock_openai = mock.MagicMock()
            sys.modules["openai"] = mock_openai
            try:
                _, _, resolved = _get_llm_client("openai", api_key_env="MY_CUSTOM_KEY")
                assert resolved.api_key == "custom-key"
            finally:
                del sys.modules["openai"]

    def test_global_image_ai_configuration(self):
        original = get_image_ai_config()
        try:
            configured = configure_image_ai(provider="openai", api_key="abc", model="gpt-test")
            current = get_image_ai_config()
            assert configured.provider == "openai"
            assert current.api_key == "abc"
            assert current.model == "gpt-test"
        finally:
            configure_image_ai(
                provider=original.provider,
                api_key=original.api_key,
                api_key_env=original.api_key_env,
                model=original.model,
                base_url=original.base_url,
            )

    def test_prepare_image_normalizes_tiff_to_png(self):
        Image = pytest.importorskip("PIL.Image")
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            path = f.name
        try:
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(path, format="TIFF")
            image_bytes, media_type = _prepare_image_for_llm(path)
            assert media_type == "image/png"
            assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            os.unlink(path)

    def test_prepare_image_passes_through_png(self):
        image_path = self._mock_image_path()
        try:
            image_bytes, media_type = _prepare_image_for_llm(image_path)
            assert media_type == "image/png"
            assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            os.unlink(image_path)


# ─── TestThemeFromImage (mock-based) ──────────────────────────────────────


class TestThemeFromImage:
    """theme_from_image with mocked LLM calls."""

    def test_returns_theme(self):
        image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_path.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        image_path.close()

        mock_response = json.dumps({
            "node_styles": {
                "default": {"shape": "roundrect", "base_color": "#56B4E9"},
                "input": {"base_color": "#009E73"},
            },
            "edge_styles": {
                "default": {"color": "#666666", "width": 1.5},
            },
            "cluster_style": {"fill": "#F0F0F0"},
            "graph_style": {"background_color": "#FFFFFF"},
        })

        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock())
                result = theme_from_image(image_path.name, provider="anthropic")

            assert isinstance(result, Theme)

            assert isinstance(result.get_node_style("default"), NodeStyle)
            assert result.get_node_style("default").base_color == "#56B4E9"
            assert isinstance(result.get_node_style("input"), NodeStyle)
            assert result.get_node_style("input").base_color == "#009E73"

            assert result.edge_styles["default"].color == "#666666"
            assert result.edge_styles["default"].width == 1.5

            assert isinstance(result.cluster_style, ClusterStyle)
            assert result.cluster_style.fill == "#F0F0F0"

            assert result.graph_style.background_color == "#FFFFFF"
        finally:
            os.unlink(image_path.name)

    def test_theme_dict_from_image(self):
        image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_path.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        image_path.close()
        mock_response = json.dumps({"node_styles": {"default": {"base_color": "#56B4E9"}}})
        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock(), ImageAIConfig(provider="anthropic", model="x", api_key="k"))
                result = theme_dict_from_image(image_path.name, provider="anthropic")
            assert result["node_styles"]["default"]["base_color"] == "#56B4E9"
        finally:
            os.unlink(image_path.name)

    def test_theme_code_from_image(self):
        image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_path.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        image_path.close()
        mock_response = json.dumps({"name": "magical", "node_styles": {"default": {"base_color": "#56B4E9"}}})
        try:
            with mock.patch("dagua.io._get_llm_client") as mock_get, \
                 mock.patch("dagua.io._send_image_to_llm", return_value=mock_response):
                mock_get.return_value = ("anthropic", mock.MagicMock(), ImageAIConfig(provider="anthropic", model="x", api_key="k"))
                code = theme_code_from_image(image_path.name, provider="anthropic")
            assert "theme = Theme(" in code
            assert "'default': NodeStyle(base_color='#56B4E9')" in code
        finally:
            os.unlink(image_path.name)


# ─── TestGraphFromYaml ────────────────────────────────────────────────────


@pytest.mark.smoke
class TestGraphFromYaml:
    """graph_from_yaml: string, file, styles, clusters, theme-by-name."""

    def test_basic_yaml_string(self):
        yaml_str = """
nodes:
  - id: a
    label: "Node A"
  - id: b
    label: "Node B"
edges:
  - source: a
    target: b
"""
        g = graph_from_yaml(yaml_str)
        assert g.num_nodes == 2
        assert g.node_labels == ["Node A", "Node B"]
        assert g.edge_index.shape == (2, 1)

    def test_yaml_file(self):
        data = {
            "nodes": [{"id": "x"}, {"id": "y"}],
            "edges": [{"source": "x", "target": "y"}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            g = graph_from_yaml(path)
            assert g.num_nodes == 2
        finally:
            os.unlink(path)

    def test_annotated_yaml_example_file(self):
        path = "/home/jtaylor/projects/dagua/examples/formats/annotated_graph.yaml"
        g = load(path)
        assert g.num_nodes == 6
        assert "core_system" in g.clusters
        assert g.direction == "TB"

    def test_yaml_with_styles(self):
        yaml_str = """
nodes:
  - id: a
    label: "Styled"
    style:
      shape: ellipse
      base_color: "#D55E00"
edges:
  - source: a
    target: b
    style:
      color: "#FF0000"
      style: dashed
"""
        g = graph_from_yaml(yaml_str)
        assert g.node_styles[0].shape == "ellipse"
        assert g.node_styles[0].base_color == "#D55E00"
        assert g.edge_styles[0].color == "#FF0000"
        assert g.edge_styles[0].style == "dashed"

    def test_yaml_clusters_with_parent(self):
        yaml_str = """
nodes:
  - id: a
  - id: b
  - id: c
  - id: d
edges:
  - source: a
    target: b
  - source: c
    target: d
clusters:
  - name: outer
    members: [a, b, c, d]
    label: "Outer"
  - name: inner
    members: [a, b]
    label: "Inner"
    parent: outer
"""
        g = graph_from_yaml(yaml_str)
        assert "outer" in g.clusters
        assert "inner" in g.clusters
        assert g.cluster_parents["inner"] == "outer"
        assert g.cluster_labels["outer"] == "Outer"

    def test_yaml_theme_by_name(self):
        yaml_str = """
theme: "dark"
nodes:
  - id: a
    type: input
"""
        g = graph_from_yaml(yaml_str)
        assert isinstance(g._theme, Theme)
        assert g._theme.name == "dark"
        # Dark theme has specific graph background
        assert g._theme.graph_style.background_color == DARK_THEME.graph_style.background_color

    def test_yaml_theme_inline(self):
        yaml_str = """
theme:
  node_styles:
    default:
      base_color: "#FF0000"
  graph_style:
    background_color: "#000000"
nodes:
  - id: a
"""
        g = graph_from_yaml(yaml_str)
        assert isinstance(g._theme, Theme)
        assert g._theme.get_node_style("default").base_color == "#FF0000"
        assert g._theme.graph_style.background_color == "#000000"

    def test_yaml_direction(self):
        yaml_str = """
direction: BT
nodes:
  - id: a
"""
        g = graph_from_yaml(yaml_str)
        assert g.direction == "BT"


# ─── TestGraphToYaml ──────────────────────────────────────────────────────


@pytest.mark.smoke
class TestGraphToYaml:
    """graph_to_yaml: roundtrip and serialization."""

    def test_roundtrip_yaml(self):
        yaml_str = """
nodes:
  - id: a
    label: "Node A"
  - id: b
    label: "Node B"
edges:
  - source: a
    target: b
    label: flow
"""
        g = graph_from_yaml(yaml_str)
        output = graph_to_yaml(g)
        g2 = graph_from_yaml(output)
        assert g2.num_nodes == 2
        assert g2.node_labels == ["Node A", "Node B"]
        assert g2.edge_labels == ["flow"]

    def test_to_yaml_string(self):
        g = graph_from_json({"nodes": [{"id": "a"}, {"id": "b"}]})
        result = graph_to_yaml(g)
        assert isinstance(result, str)
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert len(parsed["nodes"]) == 2

    def test_to_yaml_file(self):
        g = graph_from_json({"nodes": [{"id": "a"}]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            graph_to_yaml(g, path)
            with open(path) as f:
                parsed = yaml.safe_load(f)
            assert len(parsed["nodes"]) == 1
        finally:
            os.unlink(path)


# ─── TestUnifiedLoadSave ─────────────────────────────────────────────────


@pytest.mark.smoke
class TestUnifiedLoadSave:
    """Unified load/save API with format auto-detection."""

    def test_load_json_file(self):
        data = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            g = load(path)
            assert g.num_nodes == 2
        finally:
            os.unlink(path)

    def test_annotated_json_example_file(self):
        path = "/home/jtaylor/projects/dagua/examples/formats/annotated_graph.json"
        g = load(path)
        assert g.num_nodes == 6
        assert "core_system" in g.clusters
        assert g.direction == "TB"

    def test_load_yaml_file(self):
        data = {"nodes": [{"id": "a"}, {"id": "b"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            g = load(path)
            assert g.num_nodes == 2
        finally:
            os.unlink(path)

    def test_load_yml_file(self):
        data = {"nodes": [{"id": "a"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            g = load(path)
            assert g.num_nodes == 1
        finally:
            os.unlink(path)

    def test_load_dict(self):
        g = load({"nodes": [{"id": "a"}, {"id": "b"}]})
        assert g.num_nodes == 2

    def test_load_json_string(self):
        g = load('{"nodes": [{"id": "a"}]}')
        assert g.num_nodes == 1

    def test_save_yaml_default(self):
        g = graph_from_json({"nodes": [{"id": "a"}]})
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save(g, path)
            with open(path) as f:
                parsed = yaml.safe_load(f)
            assert len(parsed["nodes"]) == 1
        finally:
            os.unlink(path)

    def test_save_json_explicit(self):
        g = graph_from_json({"nodes": [{"id": "a"}]})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save(g, path)
            with open(path) as f:
                parsed = json.load(f)
            assert len(parsed["nodes"]) == 1
        finally:
            os.unlink(path)

    def test_save_roundtrip(self):
        original = {"nodes": [{"id": "a", "label": "X"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
        g = graph_from_json(original)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save(g, path)
            g2 = load(path)
            assert g2.num_nodes == 2
            assert g2.node_labels[0] == "X"
            assert g2.edge_index.shape == (2, 1)
        finally:
            os.unlink(path)


# ─── TestThemeRegistry ────────────────────────────────────────────────────


@pytest.mark.smoke
class TestThemeRegistry:
    """Theme registry: name → Theme lookup."""

    def test_get_theme_default(self):
        theme = get_theme("default")
        assert isinstance(theme, Theme)
        assert theme.name == "default"

    def test_get_theme_dark(self):
        theme = get_theme("dark")
        assert isinstance(theme, Theme)
        assert theme.name == "dark"
        assert theme.graph_style.background_color == DARK_THEME.graph_style.background_color

    def test_get_theme_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown theme.*Available"):
            get_theme("nonexistent")

    def test_get_theme_returns_copy(self):
        t1 = get_theme("default")
        t2 = get_theme("default")
        t1.name = "modified"
        assert t2.name == "default"  # t2 is a separate copy


# ─── TestBundledGraphs ────────────────────────────────────────────────────


@pytest.mark.smoke
class TestBundledGraphs:
    """Bundled graph library: dagua.graphs.load() and list_graphs()."""

    def test_load_diamond(self):
        import dagua.graphs
        g = dagua.graphs.load("diamond")
        assert isinstance(g, DaguaGraph)
        assert g.num_nodes == 4
        assert g.edge_index.shape[1] == 4

    def test_load_pipeline(self):
        import dagua.graphs
        g = dagua.graphs.load("pipeline")
        assert g.num_nodes == 5
        assert g.direction == "LR"

    def test_load_neural_net(self):
        import dagua.graphs
        g = dagua.graphs.load("neural_net")
        assert g.num_nodes == 8
        assert "feature_extractor" in g.clusters

    def test_load_nested_clusters(self):
        import dagua.graphs
        g = dagua.graphs.load("nested_clusters")
        assert g.cluster_parents["left"] == "outer"
        assert g.cluster_parents["right"] == "outer"

    def test_list_graphs(self):
        import dagua.graphs
        names = dagua.graphs.list_graphs()
        assert isinstance(names, list)
        assert "diamond" in names
        assert "pipeline" in names
        assert "neural_net" in names
        assert "nested_clusters" in names

    def test_unknown_graph_raises(self):
        import dagua.graphs
        with pytest.raises(ValueError, match="Unknown graph.*Available"):
            dagua.graphs.load("nonexistent_graph")

    def test_all_bundled_graphs_load_with_valid_invariants(self):
        """Load every bundled graph and verify basic structural invariants."""
        import dagua.graphs

        names = dagua.graphs.list_graphs()
        assert len(names) >= 35, f"Expected ≥35 bundled graphs, got {len(names)}"

        for name in names:
            g = dagua.graphs.load(name)

            # Basic structure
            assert isinstance(g, DaguaGraph), f"{name}: not a DaguaGraph"
            assert g.num_nodes > 0, f"{name}: zero nodes"

            # edge_index shape: [2, E]
            assert g.edge_index.ndim == 2, f"{name}: edge_index not 2D"
            assert g.edge_index.shape[0] == 2, f"{name}: edge_index first dim != 2"

            # Node labels match count
            assert len(g.node_labels) == g.num_nodes, (
                f"{name}: {len(g.node_labels)} labels != {g.num_nodes} nodes"
            )

            # Edge endpoints within bounds
            num_edges = g.edge_index.shape[1]
            if num_edges > 0:
                assert g.edge_index.max().item() < g.num_nodes, (
                    f"{name}: edge references node >= num_nodes"
                )
                assert g.edge_index.min().item() >= 0, (
                    f"{name}: negative node index in edge_index"
                )

            # Direction is valid
            assert g.direction in ("TB", "BT", "LR", "RL"), (
                f"{name}: invalid direction {g.direction!r}"
            )


# ─── TestClassmethods ─────────────────────────────────────────────────────


@pytest.mark.smoke
class TestGraphClassmethods:
    """DaguaGraph.load/save/from_yaml/to_yaml classmethods."""

    def test_classmethod_load(self):
        g = DaguaGraph.load({"nodes": [{"id": "a"}]})
        assert g.num_nodes == 1

    def test_classmethod_from_yaml(self):
        yaml_str = "nodes:\n  - id: a\n  - id: b\n"
        g = DaguaGraph.from_yaml(yaml_str)
        assert g.num_nodes == 2

    def test_classmethod_to_yaml(self):
        g = graph_from_json({"nodes": [{"id": "a"}]})
        result = g.to_yaml()
        assert isinstance(result, str)
        assert "nodes" in result

    def test_classmethod_save_and_load(self):
        g = graph_from_json({"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]})
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            g.save(path)
            g2 = DaguaGraph.load(path)
            assert g2.num_nodes == 2
        finally:
            os.unlink(path)


# ─── TestToNetworkx ──────────────────────────────────────────────────────


@pytest.mark.smoke
class TestToNetworkx:
    """to_networkx export and roundtrip."""

    def test_basic_export(self):
        nx = pytest.importorskip("networkx")
        from dagua.io import to_networkx

        g = graph_from_json({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        G = to_networkx(g)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert G.nodes["a"]["label"] == "A"

    def test_roundtrip(self):
        pytest.importorskip("networkx")
        from dagua.io import to_networkx

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}],
        })
        G = to_networkx(g)
        g2 = DaguaGraph.from_networkx(G)
        assert g2.num_nodes == 3
        assert g2.edge_index.shape[1] == 2

    def test_cluster_attribute(self):
        pytest.importorskip("networkx")
        from dagua.io import to_networkx

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
            "clusters": [{"name": "grp", "members": ["a", "b"]}],
        })
        G = to_networkx(g)
        assert G.nodes["a"]["cluster"] == "grp"

    def test_classmethod_wrapper(self):
        pytest.importorskip("networkx")

        g = graph_from_json({"nodes": [{"id": "x"}]})
        G = g.to_networkx()
        assert G.number_of_nodes() == 1


# ─── TestToIgraph ────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestToIgraph:
    """to_igraph export and roundtrip."""

    def test_basic_export(self):
        igraph = pytest.importorskip("igraph")
        from dagua.io import to_igraph

        g = graph_from_json({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        ig = to_igraph(g)
        assert isinstance(ig, igraph.Graph)
        assert ig.vcount() == 2
        assert ig.ecount() == 1
        assert ig.vs[0]["label"] == "A"

    def test_roundtrip(self):
        pytest.importorskip("igraph")
        from dagua.io import to_igraph, from_igraph

        g = graph_from_json({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        ig = to_igraph(g)
        g2 = from_igraph(ig)
        assert g2.num_nodes == 2
        assert g2.edge_index.shape[1] == 1
        assert g2.node_labels[0] == "A"

    def test_classmethod_wrappers(self):
        igraph = pytest.importorskip("igraph")

        g = graph_from_json({
            "nodes": [{"id": "x"}, {"id": "y"}],
            "edges": [{"source": "x", "target": "y"}],
        })
        ig = g.to_igraph()
        assert ig.vcount() == 2

        g2 = DaguaGraph.from_igraph(ig)
        assert g2.num_nodes == 2


# ─── TestToScipy ─────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestToScipy:
    """to_scipy export and roundtrip."""

    def test_basic_export(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from dagua.io import to_scipy

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}],
        })
        adj = to_scipy(g)
        assert adj.shape == (3, 3)
        assert adj.nnz == 2

    def test_roundtrip(self):
        pytest.importorskip("scipy.sparse")
        from dagua.io import to_scipy, from_scipy

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}],
        })
        adj = to_scipy(g)
        g2 = from_scipy(adj, labels=["a", "b", "c"])
        assert g2.num_nodes == 3
        assert g2.edge_index.shape[1] == 2

    def test_empty_graph(self):
        pytest.importorskip("scipy.sparse")
        from dagua.io import to_scipy

        g = graph_from_json({"nodes": [{"id": "a"}, {"id": "b"}]})
        adj = to_scipy(g)
        assert adj.shape == (2, 2)
        assert adj.nnz == 0

    def test_classmethod_wrappers(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        adj = g.to_scipy()
        assert adj.shape == (2, 2)

        g2 = DaguaGraph.from_scipy(adj)
        assert g2.num_nodes == 2


# ─── TestToPyg ───────────────────────────────────────────────────────────


class TestToPyg:
    """to_pyg export."""

    def test_basic_export(self):
        torch_geometric = pytest.importorskip("torch_geometric")
        from dagua.io import to_pyg

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        data = to_pyg(g)
        assert data.num_nodes == 2
        assert data.edge_index.shape == (2, 1)

    def test_classmethod_wrapper(self):
        pytest.importorskip("torch_geometric")

        g = graph_from_json({
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        data = g.to_pyg()
        assert data.num_nodes == 2


# ─── TestFromDot ─────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestFromDot:
    """from_dot DOT parsing."""

    def test_basic_parse(self):
        pytest.importorskip("pydot")
        from dagua.io import from_dot

        dot = '''digraph G {
            rankdir=LR;
            a [label="Node A"];
            b [label="Node B"];
            a -> b;
        }'''
        g = from_dot(dot)
        assert g.num_nodes == 2
        assert g.edge_index.shape[1] == 1
        assert g.direction == "LR"

    def test_with_clusters(self):
        pytest.importorskip("pydot")
        from dagua.io import from_dot

        dot = '''digraph G {
            a; b; c;
            a -> b;
            b -> c;
            subgraph cluster_grp {
                label="Group";
                a; b;
            }
        }'''
        g = from_dot(dot)
        assert g.num_nodes == 3
        assert "grp" in g.clusters

    def test_classmethod_wrapper(self):
        pytest.importorskip("pydot")

        dot = 'digraph G { a -> b; }'
        g = DaguaGraph.from_dot(dot)
        assert g.num_nodes == 2

    def test_roundtrip_dot_export_import(self):
        """Export to DOT via graphviz_utils.to_dot, re-import via from_dot."""
        pytest.importorskip("pydot")
        from dagua.io import from_dot
        from dagua.graphviz_utils import to_dot

        g = graph_from_json({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        g.compute_node_sizes()
        dot_str = to_dot(g)
        g2 = from_dot(dot_str)
        # The DOT export uses n0, n1 as node names
        assert g2.num_nodes == 2
        assert g2.edge_index.shape[1] == 1
