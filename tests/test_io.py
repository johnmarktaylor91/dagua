"""Tests for io.py — graph_from_json, graph_to_json, LLM-based construction."""

import json
import os
import tempfile
from unittest import mock

import pytest

from dagua.graph import DaguaGraph
from dagua.io import (
    _dict_to_cluster_style,
    _dict_to_edge_style,
    _dict_to_node_style,
    _extract_json_from_response,
    _get_llm_client,
    graph_from_image,
    graph_from_json,
    graph_to_json,
    theme_from_image,
)
from dagua.styles import ClusterStyle, EdgeStyle, GraphStyle, NodeStyle, Theme


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

    def test_provider_auto_detection_anthropic(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with mock.patch("dagua.io.anthropic", create=True) as mock_anthropic:
                mock_client = mock.MagicMock()
                mock_anthropic.Anthropic.return_value = mock_client
                # Need to mock the import
                import sys
                sys.modules["anthropic"] = mock_anthropic

                try:
                    provider, client = _get_llm_client()
                    assert provider == "anthropic"
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
                    provider, client = _get_llm_client()
                    assert provider == "openai"
                finally:
                    del sys.modules["openai"]

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No LLM API key found"):
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
