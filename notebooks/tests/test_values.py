import tomllib

from utils import values


def test_numbers_print_as_the_text_does() -> None:
    assert values.num(0.9564) == "0.96"
    assert values.num(-0.0004, 3) == "0.000"
    assert values.num(-16.0, 1) == "−16.0"
    assert values.signed(0.2349) == "+0.2"
    assert values.signed(-15.97) == "−16.0"
    assert values.count(1519650) == "1,519,650"
    assert values.pct(0.7941) == "79"


def test_spans_use_a_dash_unless_a_bound_is_negative() -> None:
    assert values.span(0.94, 0.968) == "0.94–0.97"
    assert values.span(-0.007, 0.038) == "−0.01 to 0.04"
    assert values.span(0.951, 0.949) == "0.95"


def test_written_file_reads_back(tmp_path) -> None:
    path = tmp_path / "values.toml"
    values.write(path, {"b": {"x": "1"}, "a": {"y": 'say "hi" \\'}}, "a test")
    assert tomllib.loads(path.read_text(encoding="utf-8")) == {
        "a": {"y": 'say "hi" \\'},
        "b": {"x": "1"},
    }
