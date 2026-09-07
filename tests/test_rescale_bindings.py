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
RESCALE__WRONG_INTERVAL = 67

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
RESCALE_BIN = os.path.abspath("./bin/rescale")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [RESCALE_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Rows of dim %e-formatted values, one row per time step, shape
    (length, dim)."""
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    return np.array(rows)


def load_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (dim, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


CASES = [
    # label, args, datafile, columns, length, exclude, compute_kwargs
    ("defaults", [], AR_RUN, [1], None, 0, {}),
    ("custom_range", ["-z0.2", "-Z0.8"], AR_RUN, [1], None, 0,
     dict(xmin=0.2, xmax=0.8)),
    ("set_av", ["-a"], AR_RUN, [1], None, 0, dict(set_av=True)),
    ("set_var", ["-v"], AR_RUN, [1], None, 0, dict(set_var=True)),
    ("set_av_and_var", ["-a", "-v"], AR_RUN, [1], None, 0,
     dict(set_av=True, set_var=True)),
    ("column_2", ["-c2"], HENON, [2], None, 0, {}),
    ("multi_dim_2", ["-m2", "-c1,2"], HENON, [1, 2], None, 0, {}),
    ("multi_dim_3", ["-m3", "-c1,2,3"], LORENZ, [1, 2, 3], None, 0, {}),
    ("length", ["-l300"], AR_RUN, [1], 300, 0, {}),
    ("exclude", ["-x50"], AR_RUN, [1], None, 50, {}),
    ("length_and_exclude", ["-l300", "-x50"], AR_RUN, [1], 300, 50, {}),
    ("verbosity_0", ["-V0"], AR_RUN, [1], None, 0, {}),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,compute_kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, columns, length, exclude, compute_kwargs):
    out = run_cli(args + [datafile])
    cli_result = parse_output(out)

    series = load_columns(datafile, columns, length=length, exclude=exclude)
    py_result = tisean.rescale.compute(series, **compute_kwargs)

    assert py_result.shape == series.shape
    np.testing.assert_allclose(py_result.T, cli_result, **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.res"
    run_cli(["-a", "-o" + str(outfile), AR_RUN])

    cli_result = parse_output(outfile.read_text())

    series = load_columns(AR_RUN, [1])
    py_result = tisean.rescale.compute(series, set_av=True)

    np.testing.assert_allclose(py_result.T, cli_result, **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_defaults():
    series = load_columns(AR_RUN, [1])

    default = tisean.rescale.compute(series)
    explicit = tisean.rescale.compute(series, set_av=False, set_var=False, xmin=0.0, xmax=1.0)

    np.testing.assert_array_equal(default, explicit)


def test_compute_default_range_matches_zero_one():
    series = load_columns(AR_RUN, [1])

    result = tisean.rescale.compute(series)

    assert result.min() == pytest.approx(0.0, abs=1e-6)
    assert result.max() == pytest.approx(1.0, abs=1e-6)


def test_compute_set_av_gives_zero_mean():
    series = load_columns(AR_RUN, [1])

    result = tisean.rescale.compute(series, set_av=True)

    assert result.mean() == pytest.approx(0.0, abs=1e-9)


def test_compute_set_var_gives_unit_variance():
    series = load_columns(AR_RUN, [1])

    result = tisean.rescale.compute(series, set_var=True)

    assert result.std() == pytest.approx(1.0, abs=1e-6)


def test_compute_rejects_wrong_interval_like_cli(tmp_path):
    result = subprocess.run(
        [RESCALE_BIN, "-z1", "-Z0", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE__WRONG_INTERVAL

    series = load_columns(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.rescale.compute(series, xmin=1.0, xmax=0.0)


def test_compute_rejects_wrong_interval_even_when_unused_like_cli():
    # xmin/xmax only matter in the default (set_av=set_var=False) mode, but
    # the CLI checks xmin < xmax unconditionally, before it ever looks at
    # -a/-v - so an invalid range is rejected here too, even though it goes
    # unused once set_av is on.
    result = subprocess.run(
        [RESCALE_BIN, "-a", "-z1", "-Z0", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE__WRONG_INTERVAL

    series = load_columns(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.rescale.compute(series, set_av=True, xmin=1.0, xmax=0.0)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [RESCALE_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full((1, 20), 1.5)
    with pytest.raises(ValueError):
        tisean.rescale.compute(series)


def test_compute_rejects_constant_data_even_with_set_av_like_cli(tmp_path):
    # variance() runs unconditionally on every row before the mode-specific
    # rescaling, so -a alone doesn't save a constant row from being
    # rejected - even though the plain "subtract the mean" operation would
    # otherwise be well-defined for constant data.
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [RESCALE_BIN, "-a", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full((1, 20), 1.5)
    with pytest.raises(ValueError):
        tisean.rescale.compute(series, set_av=True)


def test_compute_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        tisean.rescale.compute(np.zeros(10))


def test_compute_rejects_empty_series():
    with pytest.raises(ValueError):
        tisean.rescale.compute(np.zeros((1, 0)))
