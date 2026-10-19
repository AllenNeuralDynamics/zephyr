import matplotlib

matplotlib.use("Agg")

import pytest

from utils import plots, runs, style


def test_every_entity_has_a_colour() -> None:
    names = [*style.METHODS, *style.VARIANTS, *style.CHANNELS, *style.SETS, "Truth"]
    assert all(style.color(n).startswith("#") for n in names)


def test_every_compared_method_has_its_own_colour() -> None:
    assert set(runs.METHODS) == set(style.METHODS)
    assert set(runs.VARIANTS) == set(style.VARIANTS)
    assert len(set(style.METHODS.values())) == len(style.METHODS)


def test_every_test_set_has_a_colour() -> None:
    assert set(runs.SETS.values()) <= set(style.SETS)


@pytest.mark.parametrize("width", style.COLUMN_IN)
def test_figures_have_the_named_width(width: str) -> None:
    style.use_style()
    fig, _ = style.figure(width, 0.5)
    assert fig.get_size_inches()[0] == style.COLUMN_IN[width]


def test_metric_labels_cover_the_benchmark_metrics() -> None:
    assert set(plots.METRIC_LABELS) == set(runs.METRICS)


def test_every_quality_family_has_a_colour() -> None:
    from zephyr.quality_control.metrics import MEASURES

    assert {family for family, _, _ in MEASURES.values()} == set(style.QUALITY)
