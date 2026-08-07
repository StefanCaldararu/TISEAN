import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23
RESCALE_DATA_ZERO_INTERVAL = 11
FSLE__TOO_LARGE_MINEPS = 55

LORENZ = "tests/refs/lorenz_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
FSLE_BIN = os.path.abspath("./bin/fsle")


def run_cli(args, datafile, tmp_path, name="out.fsl"):
    outfile = tmp_path / name
    subprocess.run(
        [FSLE_BIN] + args + ["-o", str(outfile), datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return outfile.read_text()


def parse_output(text):
    """fsle's output file is '<eps> <lyapunov> <count>' rows, one per
    epsilon bin that saw at least one divergence event - no header/comment
    lines at all."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3:
            rows.append([float(parts[0]), float(parts[1]), int(parts[2])])
    return np.array(rows).reshape(-1, 3)


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


def py_rows(result):
    rows = [
        [result.eps[i], result.lyapunov[i], result.count[i]]
        for i in range(result.n)
    ]
    return np.array(rows).reshape(-1, 3)


CASES = [
    # label, args, datafile, column, length, exclude, dim, delay, mindist, eps0, epsset
    ("defaults", ["-l500"], LORENZ, 1, 500, 0, 2, 1, 0, 1.e-3, False),
    ("dim_3", ["-l500", "-m3"], LORENZ, 1, 500, 0, 3, 1, 0, 1.e-3, False),
    ("delay_2", ["-l500", "-d2"], LORENZ, 1, 500, 0, 2, 2, 0, 1.e-3, False),
    ("mindist_5", ["-l500", "-t5"], LORENZ, 1, 500, 0, 2, 1, 5, 1.e-3, False),
    ("raw_eps", ["-l500", "-r0.5"], LORENZ, 1, 500, 0, 2, 1, 0, 0.5, True),
    ("column_2", ["-l500", "-c2"], LORENZ, 2, 500, 0, 2, 1, 0, 1.e-3, False),
    ("length_300", ["-l300"], LORENZ, 1, 300, 0, 2, 1, 0, 1.e-3, False),
    ("exclude_50", ["-x50", "-l300"], LORENZ, 1, 300, 50, 2, 1, 0, 1.e-3, False),
    ("verbosity_0", ["-l500", "-V0"], LORENZ, 1, 500, 0, 2, 1, 0, 1.e-3, False),
    ("henon", ["-l500"], HENON, 1, 500, 0, 2, 1, 0, 1.e-3, False),
    (
        "combo_dim3_delay2_mindist2",
        ["-l500", "-m3", "-d2", "-t2"],
        LORENZ,
        1,
        500,
        0,
        3,
        2,
        2,
        1.e-3,
        False,
    ),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,dim,delay,mindist,eps0,epsset",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(
    label, args, datafile, column, length, exclude, dim, delay, mindist,
    eps0, epsset, tmp_path,
):
    cli_text = run_cli(args, datafile, tmp_path)
    cli_result = parse_output(cli_text)

    series = load_column(datafile, column, length=length, exclude=exclude)
    result = tisean.fsle.compute(
        series, dim=dim, delay=delay, mindist=mindist, eps0=eps0, epsset=epsset,
    )

    got = py_rows(result)

    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 0], cli_result[:, 0], **CLI_TEXT_TOL)
    np.testing.assert_allclose(got[:, 1], cli_result[:, 1], **CLI_TEXT_TOL)
    np.testing.assert_array_equal(got[:, 2], cli_result[:, 2])


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    cli_text = run_cli(["-l500", "-t5"], LORENZ, tmp_path, name="custom.fsl")
    cli_result = parse_output(cli_text)

    series = load_column(LORENZ, 1, length=500)
    result = tisean.fsle.compute(series, mindist=5)

    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 1], cli_result[:, 1], **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default 2, -d default 1, -t default 0, -r default
    # "(std. dev. of data)/1000" i.e. eps0=1.e-3 used as a fraction of the
    # rescaled series' standard deviation (epsset stays unset/False).
    series = load_column(LORENZ, 1, length=500)

    default = tisean.fsle.compute(series)
    explicit = tisean.fsle.compute(
        series, dim=2, delay=1, mindist=0, eps0=1.e-3, epsset=False,
    )

    assert default.n == explicit.n
    np.testing.assert_array_equal(default.eps, explicit.eps)
    np.testing.assert_array_equal(default.lyapunov, explicit.lyapunov)
    np.testing.assert_array_equal(default.count, explicit.count)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 50) + "\n")
    outfile = tmp_path / "out.fsl"

    result = subprocess.run(
        [FSLE_BIN, "-l50", "-o", str(outfile), str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO
    assert "Variance of the data is zero" in result.stderr
    assert outfile.stat().st_size == 0

    series = np.full(50, 5.0)
    with pytest.raises(ValueError):
        tisean.fsle.compute(series)


def test_compute_rejects_constant_data_after_rounding_like_cli(tmp_path):
    # Constant data whose value isn't exactly representable in binary can
    # round to a tiny nonzero variance in the sequential-sum variance()
    # computation (the average computed by summing l copies of x doesn't
    # necessarily land back on exactly x), while rescale_data()'s own
    # min/max scan is exact comparisons and always finds interval == 0 for
    # truly constant data. This means the *second* degenerate-input exit
    # path (rescale_data's, not variance()'s) is the one that actually
    # fires for values like this - both fsle.c and fsle_api.c must
    # replicate this exact ordering to match.
    datafile = tmp_path / "constant_dot1.txt"
    datafile.write_text("\n".join(["0.1"] * 3) + "\n")
    outfile = tmp_path / "out.fsl"

    result = subprocess.run(
        [FSLE_BIN, "-l3", "-o", str(outfile), str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 1.000000e-01 to 1.000000e-01" in result.stderr
    assert outfile.stat().st_size == 0

    series = np.full(3, 0.1)
    with pytest.raises(ValueError):
        tisean.fsle.compute(series)


def test_compute_rejects_eps_too_large_like_cli():
    result = subprocess.run(
        [FSLE_BIN, "-l500", "-r1e6", LORENZ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == FSLE__TOO_LARGE_MINEPS
    assert "The minimal epsilon is too large" in result.stderr

    series = load_column(LORENZ, 1, length=500)
    with pytest.raises(ValueError):
        tisean.fsle.compute(series, eps0=1e6, epsset=True)


def test_compute_rejects_dim_zero():
    series = load_column(LORENZ, 1, length=500)
    with pytest.raises(ValueError):
        tisean.fsle.compute(series, dim=0)


def test_compute_rejects_series_too_short_for_dim_delay_mindist():
    # The box-building step reads series[i] for i up to
    # length-delay*(dim-1)-1-mindist, so length must be > delay*(dim-1)+mindist;
    # with the defaults (dim=2, delay=1, mindist=0) that means length must
    # be > 1. This is a memory-safety contract (see fsle.h): below this
    # bound the CLI itself reads out of bounds too.
    rng = np.random.default_rng(0)
    too_short = rng.normal(size=1)
    with pytest.raises(ValueError):
        tisean.fsle.compute(too_short, dim=2, delay=1, mindist=0)
