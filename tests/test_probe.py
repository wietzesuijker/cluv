import pytest

from cluv.cli.probe import _parse_time_to_s


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("00:10:00", 600),     # HH:MM:SS
        ("10", 600),           # MM (sbatch: bare integer is minutes)
        ("10:30", 630),        # MM:SS (NOT HH:MM)
        ("1:30:45", 5445),     # HH:MM:SS
        ("2-00", 172800),      # D-HH
        ("1-12:30", 131400),   # D-HH:MM
        ("1-12:30:45", 131445),  # D-HH:MM:SS
        ("0-00:01", 60),       # D-HH:MM with zero day
    ],
)
def test_parse_time_to_s_valid(raw: str, expected: int) -> None:
    assert _parse_time_to_s(raw) == expected


@pytest.mark.parametrize("raw", ["", "garbage", "10:30:45:00", "abc:def", "-5", "1:2:3:4"])
def test_parse_time_to_s_invalid(raw: str) -> None:
    assert _parse_time_to_s(raw) is None
