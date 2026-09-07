import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

GET_SERIES_NO_LINES = 20

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
RESAMPLE_BIN = os.path.abspath("./bin/resample")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [RESAMPLE_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    return np.array([float(line) for line in text.splitlines() if line.strip()])


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


CASES = [
    # label, args, datafile, column, length, exclude, sampletime, order
    ("defaults", [], AR_RUN, 1, None, 0, 0.5, 4),
    ("order_2", ["-p2"], AR_RUN, 1, None, 0, 0.5, 2),
    ("order_6", ["-p6"], AR_RUN, 1, None, 0, 0.5, 6),
    ("order_0", ["-p0"], AR_RUN, 1, None, 0, 0.5, 0),
    ("sampletime_0.25", ["-s0.25"], AR_RUN, 1, None, 0, 0.25, 4),
    ("sampletime_1.0", ["-s1.0"], AR_RUN, 1, None, 0, 1.0, 4),
    ("column_2", ["-c2"], HENON, 2, None, 0, 0.5, 4),
    ("length", ["-l300"], AR_RUN, 1, 300, 0, 0.5, 4),
    ("exclude", ["-x50"], AR_RUN, 1, None, 50, 0.5, 4),
    ("length_and_exclude", ["-l300", "-x50", "-p3"], AR_RUN, 1, 300, 50, 0.5, 3),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, 0, 0.5, 4),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,sampletime,order",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, column, length, exclude, sampletime, order):
    out = run_cli(args + [datafile])
    cli_result = parse_output(out)

    series = load_column(datafile, column, length=length, exclude=exclude)
    py_result = tisean.resample.compute(series, sampletime=sampletime, order=order)

    assert py_result.shape == cli_result.shape
    np.testing.assert_allclose(py_result, cli_result, **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.rs"
    run_cli(["-p3", "-s0.25", "-o" + str(outfile), AR_RUN])

    cli_result = parse_output(outfile.read_text())

    series = load_column(AR_RUN, 1)
    py_result = tisean.resample.compute(series, sampletime=0.25, order=3)

    np.testing.assert_allclose(py_result, cli_result, **CLI_TEXT_TOL)


def test_compute_default_sampletime_and_order_match_cli():
    series = load_column(AR_RUN, 1)

    default = tisean.resample.compute(series)
    explicit = tisean.resample.compute(series, sampletime=0.5, order=4)

    np.testing.assert_array_equal(default, explicit)


def test_compute_empty_result_at_minimum_safe_length():
    # length=2 is exactly the CLI's own crash/no-crash boundary for the
    # default order=4 (horder=5, horder/2=2): the interpolation loop's own
    # condition is false immediately, so both the CLI and the binding
    # produce a valid, empty result rather than any data.
    out = run_cli(["-l2", AR_RUN])
    assert out == ""

    series = load_column(AR_RUN, 1, length=2)
    result = tisean.resample.compute(series)
    assert result.shape == (0,)


def test_compute_rejects_empty_series_like_cli_rejects_zero_lines():
    # -l0 makes get_series() itself read no data and exit(GET_SERIES_NO_LINES)
    # before resample's own math ever runs; passing an empty array straight
    # to the binding hits the same "not enough data" situation from the
    # other side (get_series isn't in the reentrant core at all).
    result = subprocess.run(
        [RESAMPLE_BIN, "-l0", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == GET_SERIES_NO_LINES

    series = np.array([], dtype=float)
    with pytest.raises(ValueError):
        tisean.resample.compute(series)


def test_compute_rejects_series_too_short_for_order():
    # length=1 with the default order=4 is short enough that the CLI's own
    # unchecked arithmetic (length - order/2, both unsigned) wraps around
    # and reads series out of bounds - confirmed to segfault the CLI
    # itself. The binding must reject this instead of crashing the
    # interpreter, so it isn't exercised via subprocess here.
    series = np.array([0.1], dtype=float)
    with pytest.raises(ValueError):
        tisean.resample.compute(series, order=4)
