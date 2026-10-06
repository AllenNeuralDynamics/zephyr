import matplotlib

matplotlib.use("Agg")

import pytest

from utils import plots, style


def test_every_entity_has_a_colour() -> None:
    names = [*style.METHODS, *style.CHANNELS, "Truth"]
    assert all(style.color(n).startswith("#") for n in names)


def test_colours_are_unique_within_a_group() -> None:
    for group in (style.METHODS, style.CHANNELS, style.OBJECTIVES, style.VARIANTS):
        assert len(set(group.values())) == len(group)


@pytest.mark.parametrize("width", style.COLUMN_IN)
def test_figures_have_the_named_width(width: str) -> None:
    style.use_style()
    fig, _ = style.figure(width, 0.5)
    assert fig.get_size_inches()[0] == style.COLUMN_IN[width]


def test_ablation_series_use_known_objectives() -> None:
    assert set(plots.OBJECTIVE_MARKERS) == set(style.OBJECTIVES)
