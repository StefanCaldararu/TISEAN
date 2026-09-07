import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
HISTOGRAM_BIN = os.path.abspath("./bin/histogram")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [HISTOGRAM_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns (min, max, average, stddev, rows) where rows is a (base, 2)
    array of [bin_center, density] - density is box[i]/length, not a raw
    count, matching what the CLI actually prints."""
    lines = text.splitlines()
    interval_line = lines[0]
    lo, hi = interval_line.split("[", 1)[1].rstrip("]").split(":")
    average = float(lines[1].split("=")[1])
    stddev = float(lines[2].split("=")[1])
    rows = [
        [float(x) for x in line.split()]
        for line in lines[3:]
        if line.strip()
    ]
    return float(lo), float(hi), average, stddev, np.array(rows).reshape(-1, 2)


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


def bin_layout(hist, length):
    """Replicates the CLI's own output formulas from a Histogram: bin
    centers mapped back into the original data range, and densities
    (counts / length)."""
    base = hist.base
    if base == 0:
        return np.empty((0, 2))
    size = 1.0 / base
    size2 = size / 2.0
    i = np.arange(base)
    centers = (i * size + size2) * hist.interval + hist.min
    density = hist.box / length
    return np.stack([centers, density], axis=1)


CASES = [
    # label, args, datafile, column, length, exclude, base
    ("defaults", [], AR_RUN, 1, None, 0, 50),
    ("base_20", ["-b20"], AR_RUN, 1, None, 0, 20),
    ("base_5", ["-b5"], AR_RUN, 1, None, 0, 5),
    ("column_2", ["-c2"], HENON, 2, None, 0, 50),
    ("length", ["-l300"], AR_RUN, 1, 300, 0, 50),
    ("exclude", ["-x50"], AR_RUN, 1, None, 50, 50),
    ("length_and_exclude", ["-l300", "-x50", "-b20"], AR_RUN, 1, 300, 50, 20),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, 0, 50),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,base",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, column, length, exclude, base):
    out = run_cli(args + [datafile])
    cli_lo, cli_hi, cli_average, cli_stddev, cli_rows = parse_output(out)

    series = load_column(datafile, column, length=length, exclude=exclude)
    hist = tisean.histogram.compute(series, base=base)

    assert hist.base == base
    np.testing.assert_allclose(hist.min, cli_lo, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.min + hist.interval, cli_hi, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.average, cli_average, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.var, cli_stddev, **CLI_TEXT_TOL)
    assert hist.box.sum() == len(series)

    py_rows = bin_layout(hist, len(series))
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.his"
    run_cli(["-b10", "-o" + str(outfile), AR_RUN])

    cli_lo, cli_hi, cli_average, cli_stddev, cli_rows = parse_output(
        outfile.read_text()
    )

    series = load_column(AR_RUN, 1)
    hist = tisean.histogram.compute(series, base=10)

    np.testing.assert_allclose(hist.min, cli_lo, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.min + hist.interval, cli_hi, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.average, cli_average, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.var, cli_stddev, **CLI_TEXT_TOL)

    py_rows = bin_layout(hist, len(series))
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_default_base_is_fifty():
    series = load_column(AR_RUN, 1)

    default = tisean.histogram.compute(series)
    explicit = tisean.histogram.compute(series, base=50)

    assert default.base == explicit.base == 50
    np.testing.assert_array_equal(default.box, explicit.box)


def test_compute_base_zero_yields_empty_histogram_like_cli():
    out = run_cli(["-b0", AR_RUN])
    cli_lo, cli_hi, cli_average, cli_stddev, cli_rows = parse_output(out)
    assert cli_rows.shape == (0, 2)

    series = load_column(AR_RUN, 1)
    hist = tisean.histogram.compute(series, base=0)

    assert hist.base == 0
    assert hist.box.shape == (0,)
    np.testing.assert_allclose(hist.min, cli_lo, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.min + hist.interval, cli_hi, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.average, cli_average, **CLI_TEXT_TOL)
    np.testing.assert_allclose(hist.var, cli_stddev, **CLI_TEXT_TOL)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [HISTOGRAM_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full(20, 1.5)
    with pytest.raises(ValueError):
        tisean.histogram.compute(series)
