"""Tests for dagua.defaults — configure(), context manager, thread safety."""

import threading

import pytest

import dagua
from dagua.defaults import (
    configure,
    defaults,
    get_default_device,
    get_default_theme,
    get_defaults,
    reset,
    set_device,
    set_theme,
)
from dagua.styles import NodeStyle, Theme


@pytest.fixture(autouse=True)
def _reset_defaults():
    """Reset global defaults before and after each test."""
    reset()
    yield
    reset()


class TestConfigure:
    def test_set_theme_string(self):
        configure(theme="dark")
        d = get_defaults()
        assert d["theme"] == "dark"

    def test_set_device(self):
        configure(device="cuda")
        assert get_default_device() == "cuda"

    def test_layout_override(self):
        configure(node_sep=40, rank_sep=80)
        d = get_defaults()
        assert d["node_sep"] == 40
        assert d["rank_sep"] == 80

    def test_node_style_override(self):
        configure(font_size=12.0)
        d = get_defaults()
        assert d["font_size"] == 12.0

    def test_edge_style_override(self):
        configure(edge_color="#FF0000")
        d = get_defaults()
        assert d["edge_color"] == "#FF0000"

    def test_graph_style_override(self):
        configure(background_color="#000000")
        d = get_defaults()
        assert d["background_color"] == "#000000"

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError, match="Unknown configure"):
            configure(nonexistent_option=42)

    def test_did_you_mean(self):
        with pytest.raises(TypeError, match="Did you mean"):
            configure(font_siz=10)  # typo

    def test_multiple_kwargs(self):
        configure(theme="minimal", device="cpu", node_sep=30, font_size=11)
        d = get_defaults()
        assert d["theme"] == "minimal"
        assert d["device"] == "cpu"
        assert d["node_sep"] == 30
        assert d["font_size"] == 11


class TestSetTheme:
    def test_set_theme(self):
        set_theme("dark")
        theme = get_default_theme()
        assert theme.name == "dark"

    def test_set_theme_invalid(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            set_theme("nonexistent")


class TestSetDevice:
    def test_set_device(self):
        set_device("cuda")
        assert get_default_device() == "cuda"


class TestContextManager:
    def test_scoped_override(self):
        configure(node_sep=25)
        assert get_defaults()["node_sep"] == 25

        with defaults(node_sep=60):
            assert get_defaults()["node_sep"] == 60

        # Restored after context exit
        assert get_defaults()["node_sep"] == 25

    def test_nested_contexts(self):
        configure(node_sep=25)
        with defaults(node_sep=40):
            assert get_defaults()["node_sep"] == 40
            with defaults(node_sep=80):
                assert get_defaults()["node_sep"] == 80
            assert get_defaults()["node_sep"] == 40
        assert get_defaults()["node_sep"] == 25

    def test_context_with_theme(self):
        set_theme("default")
        with defaults(theme="dark"):
            assert get_defaults()["theme"] == "dark"
        assert get_defaults()["theme"] == "default"

    def test_context_exception_restores(self):
        configure(node_sep=25)
        try:
            with defaults(node_sep=99):
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        # Should be restored despite exception
        assert get_defaults()["node_sep"] == 25


class TestReset:
    def test_reset(self):
        configure(theme="dark", device="cuda", node_sep=99)
        reset()
        d = get_defaults()
        assert d["theme"] == "default"
        assert d["device"] == "cpu"
        assert "node_sep" not in d


class TestThreadSafety:
    def test_independent_threads(self):
        """Each thread gets its own default stack."""
        results = {}

        def worker(name, sep):
            configure(node_sep=sep)
            import time
            time.sleep(0.01)  # Let threads interleave
            results[name] = get_defaults().get("node_sep")

        t1 = threading.Thread(target=worker, args=("t1", 100))
        t2 = threading.Thread(target=worker, args=("t2", 200))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == 100
        assert results["t2"] == 200


class TestExportConfig:
    def test_export_json(self, tmp_path):
        configure(node_sep=40, theme="dark")
        path = str(tmp_path / "config.json")
        dagua.export_config(path)

        import json
        with open(path) as f:
            data = json.load(f)
        assert data["node_sep"] == 40
        assert data["theme"] == "dark"

    def test_export_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        configure(node_sep=40, theme="dark")
        path = str(tmp_path / "config.yaml")
        dagua.export_config(path)

        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["node_sep"] == 40
        assert data["theme"] == "dark"
