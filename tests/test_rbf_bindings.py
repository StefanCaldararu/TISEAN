import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11
SOLVELE_SINGULAR_MATRIX = 19

RBF_BIN = os.path.abspath("./bin/rbf")


def run_cli(args, check=True):
    return subprocess.run(
        [RBF_BIN] + list(args),
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        check=check,
    )


def load_series(path, column=1, length=None, exclude=0):
    """Replicates get_series()'s -x/-l/-c handling: skip `exclude` lines,
    then keep up to `length` of the rest, then pick `column` (1-indexed)."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


def parse_cli_output(text):
    """Parses rbf's stdout/output-file format:
    '#Center points used:' then one '# v1 .. vdim' line per center,
    '#variance= <v>', '#Coefficients:' then one '#<v>' line per
    coefficient (centers+1 of them), '#insample error= <v>', an optional
    '#out of sample error= <v>', and then one forecasted value per
    remaining (non '#'-prefixed) line."""
    section = None
    center_rows = []
    coefs = []
    variance = None
    insample_error = None
    outsample_error = None
    forecast = []

    for line in text.splitlines():
        if line.startswith("#Center points used:"):
            section = "centers"
            continue
        if line.startswith("#variance="):
            variance = float(line.split("=", 1)[1])
            section = None
            continue
        if line.startswith("#Coefficients:"):
            section = "coefs"
            continue
        if line.startswith("#insample error="):
            insample_error = float(line.split("=", 1)[1])
            section = None
            continue
        if line.startswith("#out of sample error="):
            outsample_error = float(line.split("=", 1)[1])
            section = None
            continue
        if section == "centers" and line.startswith("#"):
            center_rows.append([float(x) for x in line[1:].split()])
            continue
        if section == "coefs" and line.startswith("#"):
            coefs.append(float(line[1:]))
            continue
        s = line.strip()
        if s and not line.startswith("#"):
            forecast.append(float(s))

    return (
        np.array(center_rows),
        variance,
        np.array(coefs),
        insample_error,
        outsample_error,
        np.array(forecast),
    )


AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"

CASES = [
    # label, datafile, column, dim, delay, centers, drift, step, insample, cast_length, length, exclude
    ("defaults_all_data", AR_RUN, 1, 2, 1, 10, True, 1, None, 0, None, 0),
    ("dim1_centers5", AR_RUN, 1, 1, 1, 5, True, 1, 300, 20, None, 0),
    # cast_length is 0 here (not e.g. 10) because the CLI forces STEP=1
    # internally whenever -L/cast_length is used (see MAKECAST in
    # source_c/rbf.c), which would make a non-1 step incomparable between
    # the CLI and the (intentionally orthogonal) Python step/cast_length
    # options - see test_makecast_forces_step_to_one_like_cli below.
    ("dim3_delay2_nodrift", AR_RUN, 1, 3, 2, 8, False, 2, 400, 0, None, 0),
    ("step_zero", AR_RUN, 1, 2, 1, 6, True, 0, 200, 0, None, 0),
    ("length_and_exclude", AR_RUN, 1, 2, 1, 10, True, 1, 300, 15, 500, 100),
    ("column2_henon", HENON, 2, 2, 1, 10, True, 1, 300, 5, None, 0),
    ("centers_clamped_to_length", AR_RUN, 1, 2, 1, 20, True, 1, None, 0, 15, 0),
    ("no_outsample_explicit", AR_RUN, 1, 2, 1, 10, True, 1, 1000, 0, None, 0),
    ("cast_only_no_split", AR_RUN, 1, 2, 1, 10, True, 1, None, 100, None, 0),
    ("drift_disabled", AR_RUN, 1, 2, 1, 10, False, 1, 300, 0, None, 0),
]


@pytest.mark.parametrize(
    "label,datafile,column,dim,delay,centers,drift,step,insample,cast_length,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_fit_matches_cli(
    label, datafile, column, dim, delay, centers, drift, step, insample, cast_length,
    length, exclude,
):
    args = [f"-m{dim}", f"-d{delay}", f"-p{centers}", f"-s{step}"]
    if not drift:
        args.append("-X")
    if insample is not None:
        args.append(f"-n{insample}")
    if cast_length:
        args.append(f"-L{cast_length}")
    if column != 1:
        args.append(f"-c{column}")
    if length is not None:
        args.append(f"-l{length}")
    if exclude:
        args.append(f"-x{exclude}")
    args.append(datafile)

    result = run_cli(args)
    cli_center, cli_variance, cli_coefs, cli_insample_err, cli_outsample_err, cli_forecast = (
        parse_cli_output(result.stdout)
    )

    series = load_series(datafile, column=column, length=length, exclude=exclude)

    kwargs = dict(dim=dim, delay=delay, centers=centers, drift=drift, step=step)
    if insample is not None:
        kwargs["insample"] = insample
    if cast_length:
        kwargs["cast_length"] = cast_length

    py_result = tisean.rbf.fit(series, **kwargs)

    expected_centers = min(centers, len(series))
    assert py_result.centers == expected_centers
    assert py_result.dim == dim
    assert py_result.delay == delay
    assert py_result.step == step

    expected_insample = min(insample if insample is not None else len(series), len(series))
    assert py_result.insample == expected_insample
    expected_has_outsample = expected_insample < len(series)
    assert py_result.has_outsample_error == expected_has_outsample

    np.testing.assert_allclose(py_result.center, cli_center, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.variance, cli_variance, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.coefs, cli_coefs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.insample_error, cli_insample_err, **CLI_TEXT_TOL)
    if expected_has_outsample:
        np.testing.assert_allclose(py_result.outsample_error, cli_outsample_err, **CLI_TEXT_TOL)
    else:
        assert py_result.outsample_error == 0.0
    if cast_length:
        np.testing.assert_allclose(py_result.cast, cli_forecast, **CLI_TEXT_TOL)
    else:
        assert py_result.cast.shape == (0,)


def test_defaults_match_cli_defaults():
    # No -m/-d/-p/-X/-s/-n/-L at all: CLI defaults are -m2, -d1, -p10, drift
    # activated, -s1, -n unset (all data used, no out-of-sample error), -L
    # unset (no forecast).
    result = run_cli([AR_RUN])
    cli_center, cli_variance, cli_coefs, cli_insample_err, cli_outsample_err, cli_forecast = (
        parse_cli_output(result.stdout)
    )
    assert cli_outsample_err is None
    assert cli_forecast.size == 0

    series = load_series(AR_RUN)
    py_result = tisean.rbf.fit(series)

    assert py_result.dim == 2
    assert py_result.delay == 1
    assert py_result.centers == 10
    assert py_result.step == 1
    assert py_result.insample == len(series)
    assert py_result.has_outsample_error is False
    assert py_result.cast_length == 0

    np.testing.assert_allclose(py_result.center, cli_center, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.variance, cli_variance, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.coefs, cli_coefs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.insample_error, cli_insample_err, **CLI_TEXT_TOL)


def test_drift_default_is_activated_like_cli():
    # -X (drift=False) must produce a different fit than the default
    # (drift=True), confirming the Python default really does apply drift
    # like the CLI does by default.
    series = load_series(AR_RUN)
    with_drift = tisean.rbf.fit(series, insample=300)
    without_drift = tisean.rbf.fit(series, insample=300, drift=False)
    assert not np.allclose(with_drift.center, without_drift.center)

    cli_x = run_cli(["-n300", "-X", AR_RUN])
    _, _, cli_x_coefs, _, _, _ = parse_cli_output(cli_x.stdout)
    np.testing.assert_allclose(without_drift.coefs, cli_x_coefs, **CLI_TEXT_TOL)


def test_verbosity_does_not_change_output():
    out0 = run_cli(["-m2", "-d1", "-p10", "-n300", "-V0", AR_RUN]).stdout
    out1 = run_cli(["-m2", "-d1", "-p10", "-n300", "-V1", AR_RUN]).stdout
    assert out0 == out1


def test_dash_o_output_file_matches_python(tmp_path):
    # Use insample >= length so no "out of sample error" line is printed:
    # the original CLI has a pre-existing bug where that one line always
    # goes to stdout (it tests `!stdout`, the never-NULL FILE* global,
    # instead of `!stdo`) even when -o redirects everything else to a
    # file. Avoiding that branch keeps this test about the Python
    # bindings, not about reproducing that CLI quirk.
    outfile = tmp_path / "out.rbf"
    run_cli(["-m2", "-d1", "-p10", "-n1000", "-L20", "-o", str(outfile), AR_RUN])
    cli_center, cli_variance, cli_coefs, cli_insample_err, cli_outsample_err, cli_forecast = (
        parse_cli_output(outfile.read_text())
    )
    assert cli_outsample_err is None

    series = load_series(AR_RUN)
    py_result = tisean.rbf.fit(series, insample=1000, cast_length=20)

    np.testing.assert_allclose(py_result.center, cli_center, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.coefs, cli_coefs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_result.cast, cli_forecast, **CLI_TEXT_TOL)


def test_makecast_forces_step_to_one_like_cli():
    # source_c/rbf.c has `if (MAKECAST) STEP=1;`: whenever -L is given, the
    # CLI silently ignores -s and fits with step=1. The Python API keeps
    # step and cast_length orthogonal (a caller can request both a
    # non-default step and a forecast), so this is a real, intentional
    # behavior difference - assert the CLI quirk directly here rather than
    # comparing it against the Python bindings.
    with_s2 = run_cli(["-m2", "-d1", "-p10", "-s2", "-L5", "-n300", AR_RUN]).stdout
    with_s1 = run_cli(["-m2", "-d1", "-p10", "-s1", "-L5", "-n300", AR_RUN]).stdout
    assert with_s2 == with_s1

    series = load_series(AR_RUN)
    py_step2 = tisean.rbf.fit(series, centers=10, insample=300, step=2, cast_length=5)
    py_step1 = tisean.rbf.fit(series, centers=10, insample=300, step=1, cast_length=5)
    assert not np.allclose(py_step2.coefs, py_step1.coefs)

    _, _, cli_coefs, _, _, cli_forecast = parse_cli_output(with_s1)
    np.testing.assert_allclose(py_step1.coefs, cli_coefs, **CLI_TEXT_TOL)
    np.testing.assert_allclose(py_step1.cast, cli_forecast, **CLI_TEXT_TOL)


def test_zero_variance_rejected_like_cli(tmp_path):
    series_path = tmp_path / "const.txt"
    series = np.full(50, 3.0)
    np.savetxt(series_path, series)

    result = run_cli(["-m2", "-d1", "-p10", str(series_path)], check=False)
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    with pytest.raises(ValueError):
        tisean.rbf.fit(series)


def test_singular_matrix_rejected_like_cli(tmp_path):
    # An alternating two-value series makes the CLI's evenly-spaced,
    # un-drifted center placement select several exactly-duplicate
    # centers (they all land on one of only two possible rescaled
    # values). Duplicate centers produce identical kernel-value columns
    # in the RBF fit's normal-equations matrix, which is then exactly
    # singular in a row Gaussian elimination with partial pivoting
    # actually detects (drift=False keeps the duplicate positions exact;
    # with drift enabled, the repulsion step's h/sqr(h)/fabs(h) force
    # divides by zero for exactly-coincident centers and poisons the fit
    # with NaNs instead, which sidesteps the singular-matrix check).
    n = 200
    series = np.empty(n)
    series[0::2] = 0.001
    series[1::2] = -0.001
    series_path = tmp_path / "alt.txt"
    np.savetxt(series_path, series)

    result = run_cli(["-m1", "-d1", "-p5", "-X", str(series_path)], check=False)
    assert result.returncode == SOLVELE_SINGULAR_MATRIX

    with pytest.raises(ValueError):
        tisean.rbf.fit(series, dim=1, delay=1, centers=5, drift=False)


def test_rejects_dim_zero():
    series = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    with pytest.raises(ValueError):
        tisean.rbf.fit(series, dim=0)


def test_rejects_series_too_short_for_dim_delay():
    series = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        tisean.rbf.fit(series, dim=2, delay=5)


def test_rejects_too_few_centers():
    series = load_series(AR_RUN)
    with pytest.raises(ValueError):
        tisean.rbf.fit(series, centers=1)
    with pytest.raises(ValueError):
        tisean.rbf.fit(series, centers=0)


def test_rejects_insample_smaller_than_step():
    series = load_series(AR_RUN)
    with pytest.raises(ValueError):
        tisean.rbf.fit(series, insample=5, step=10)


def test_result_shapes_and_dtype():
    series = load_series(AR_RUN)
    result = tisean.rbf.fit(series, dim=2, delay=1, centers=10, insample=300, cast_length=40)

    assert result.center.shape == (10, 2)
    assert result.coefs.shape == (11,)
    assert result.cast.shape == (40,)
    assert result.center.dtype == np.float64
    assert result.coefs.dtype == np.float64
    assert result.cast.dtype == np.float64
