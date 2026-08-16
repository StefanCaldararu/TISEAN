import subprocess
import sys

import numpy as np
import pytest

import tisean

AR_MODEL_TOO_MANY_POLES = 52
VARIANCE_VAR_EQ_ZERO = 23

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows. rtol=1e-6 is
# looser than 1e-7 (float64's own precision) on purpose - see
# tests/test_ar_model_bindings.py for the fuller explanation.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)


def run_cli(args):
    result = subprocess.run(
        ["./bin/arima-model"] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_data_rows(text):
    """Rows of plain (non-comment) numeric output: residuals, an iterated
    model, or (with -V4/-V5) series+residual pairs."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)


def parse_coeff_rows(text):
    """The '#x_i(n-lag) c1 c2 ... cdim' / '#e_i(n-lag) ...' coefficient
    rows (one per coeff column), shape (size, dim) - the transpose of
    ARIMAModel.coeff. Note arima-model's label format ("#x_"/"#e_", no
    space after '#') differs from ar-model's ("# ", plain space)."""
    rows = []
    for line in text.splitlines():
        if line.startswith("#x_") or line.startswith("#e_"):
            rows.append([float(x) for x in line.split()[1:]])
    return np.array(rows)


def parse_convergence(text):
    """The '#iteration N xdiff_1 ... xdiff_dim diffcoeff' convergence trace
    rows, printed only when the ARMA refinement ran."""
    xdiff_rows = []
    diffcoeff_vals = []
    for line in text.splitlines():
        if line.startswith("#iteration "):
            values = [float(x) for x in line.split()[2:]]
            xdiff_rows.append(values[:-1])
            diffcoeff_vals.append(values[-1])
    return np.array(xdiff_rows), np.array(diffcoeff_vals)


def load_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Unlike ar_model.fit(), arima_model.fit() differences
    and centers its input internally, so - unlike
    tests/test_ar_model_bindings.py's load_columns() - this deliberately
    does NOT center the result."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


# label, datafile, poles, columns, single -c value to pass (or None to omit
# and rely on the CLI's default 1..dim column order). arima-model's -c is
# declared as check_option(...,'u') (a single unsigned int), unlike
# ar-model's -c (a 's' string) - it can't take a comma-separated list, so
# column reordering/subsetting can only be exercised for dim=1.
FIT_CASES = [
    ("dim1_p1", "tests/refs/ar-run_l1000.txt", 1, [1], None),
    ("dim1_p10", "tests/refs/ar-run_l1000.txt", 10, [1], None),
    ("dim1_p2_second_column_of_henon", "tests/refs/henon_l1000.txt", 2, [2], 2),
    ("dim2_p2", "tests/refs/henon_l1000.txt", 2, [1, 2], None),
    ("dim3_p2", "tests/refs/lorenz_l1000.txt", 2, [1, 2, 3], None),
]


@pytest.mark.parametrize(
    "label,datafile,poles,columns,column_flag", FIT_CASES, ids=[c[0] for c in FIT_CASES]
)
def test_initial_ar_fit_matches_cli_across_dim_and_poles(label, datafile, poles, columns, column_flag):
    dim = len(columns)
    args = [f"-m{dim}", f"-p{poles}", "-V0"]
    if column_flag is not None:
        args.append(f"-c{column_flag}")
    args.append(datafile)

    out = run_cli(args)
    cli_coeff = parse_coeff_rows(out)
    cli_residuals = parse_data_rows(out)

    series = load_columns(datafile, columns)
    model = tisean.arima_model.fit(series, poles=poles)

    assert model.dim == dim
    assert model.poles == poles
    assert not model.arimaset
    assert model.order == poles
    assert model.size == dim * poles
    np.testing.assert_allclose(model.coeff.T, cli_coeff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(model.residuals[:, poles:].T, cli_residuals, **CLI_TEXT_TOL)


def test_fit_matches_cli_default_poles_and_dim():
    # No -p/-P/-m/-c at all: defaults are poles=10, arpoles=ipoles=mapoles=0,
    # dim=1, column 1 - matching arima_model.fit()'s own keyword defaults.
    datafile = "tests/refs/ar-run_l1000.txt"

    out = run_cli(["-V0", datafile])
    cli_residuals = parse_data_rows(out)

    series = load_columns(datafile, [1])
    model = tisean.arima_model.fit(series)

    assert model.dim == 1
    assert model.poles == 10
    assert model.arpoles == 0
    assert model.ipoles == 0
    assert model.mapoles == 0
    assert not model.arimaset
    np.testing.assert_allclose(
        model.residuals[:, model.order :].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_fit_matches_cli_with_length_and_exclude():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles, length, exclude = 3, 400, 100

    out = run_cli([f"-p{poles}", f"-l{length}", f"-x{exclude}", "-V0", datafile])
    cli_residuals = parse_data_rows(out)

    series = load_columns(datafile, [1], length=length, exclude=exclude)
    model = tisean.arima_model.fit(series, poles=poles)

    assert model.length == length
    np.testing.assert_allclose(
        model.residuals[:, poles:].T, cli_residuals, **CLI_TEXT_TOL
    )


# label, datafile, poles, columns, arpoles, ipoles, mapoles
ARMA_CASES = [
    ("regression_p10_P2_0_1", "tests/refs/henon_l1000.txt", 10, [1], 2, 0, 1),
    ("P1_1_0_with_differencing", "tests/refs/ar-run_l1000.txt", 5, [1], 1, 1, 0),
    ("P0_2_1", "tests/refs/ar-run_l1000.txt", 5, [1], 0, 2, 1),
]


@pytest.mark.parametrize(
    "label,datafile,poles,columns,arpoles,ipoles,mapoles",
    ARMA_CASES,
    ids=[c[0] for c in ARMA_CASES],
)
def test_arma_refinement_matches_cli(label, datafile, poles, columns, arpoles, ipoles, mapoles):
    dim = len(columns)
    args = [f"-m{dim}", f"-p{poles}", f"-P{arpoles},{ipoles},{mapoles}", "-V0", datafile]

    out = run_cli(args)
    cli_coeff = parse_coeff_rows(out)
    cli_residuals = parse_data_rows(out)
    cli_xdiff, cli_diffcoeff = parse_convergence(out)

    series = load_columns(datafile, columns)
    model = tisean.arima_model.fit(
        series, poles=poles, arpoles=arpoles, ipoles=ipoles, mapoles=mapoles
    )

    assert model.arimaset
    assert model.arpoles == arpoles
    assert model.ipoles == ipoles
    assert model.mapoles == mapoles
    assert model.order == max(arpoles, mapoles)
    assert model.size == (arpoles + mapoles) * dim
    assert model.realiter == len(cli_xdiff)
    np.testing.assert_allclose(model.xdiff, cli_xdiff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(model.diffcoeff, cli_diffcoeff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(model.coeff.T, cli_coeff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(
        model.residuals[:, model.order :].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_arima_regression_case_matches_recorded_reference():
    # Same args as tests/test_arima-model.py's golden regression test -
    # cross-checks the binding against the recorded reference output too,
    # not just a freshly-run CLI subprocess.
    ref = np.loadtxt("tests/refs/arima-model_p10.txt")

    series = load_columns("tests/refs/henon_l1000.txt", [1])
    model = tisean.arima_model.fit(series, poles=10, arpoles=2, ipoles=0, mapoles=1)

    # Unlike tests/test_arima-model.py's own comparison (CLI text vs. CLI
    # text, both lossy through the same %e roundtrip, tight rtol=1e-7 is
    # fine), this compares the binding's full-precision doubles against a
    # reference file that was itself captured through that %e roundtrip -
    # needs the same looser CLI_TEXT_TOL as everything else in this file.
    np.testing.assert_allclose(
        model.residuals[0, model.order :], ref, **CLI_TEXT_TOL
    )


@pytest.mark.parametrize("iterations,convergence", [(5, 1e-3), (100, 1e-6)])
def test_iterations_and_convergence_match_cli(iterations, convergence):
    datafile = "tests/refs/henon_l1000.txt"

    out = run_cli(
        ["-m2", "-p10", "-P2,0,1", f"-I{iterations}", f"-e{convergence}", "-V0", datafile]
    )
    cli_xdiff, cli_diffcoeff = parse_convergence(out)

    series = load_columns(datafile, [1, 2])
    model = tisean.arima_model.fit(
        series,
        poles=10,
        arpoles=2,
        ipoles=0,
        mapoles=1,
        iterations=iterations,
        convergence=convergence,
    )

    assert model.realiter == len(cli_xdiff)
    assert model.realiter <= iterations
    np.testing.assert_allclose(model.xdiff, cli_xdiff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(model.diffcoeff, cli_diffcoeff, **CLI_TEXT_TOL)


def test_verbosity_usr2_matches_series_plus_average():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles = 3

    out = run_cli([f"-p{poles}", "-V5", datafile])
    cli_rows = parse_data_rows(out)

    series = load_columns(datafile, [1])
    model = tisean.arima_model.fit(series, poles=poles)

    expected_series = model.series[0, poles:] + model.average[0]
    expected_residuals = model.residuals[0, poles:]

    np.testing.assert_allclose(cli_rows[:, 0], expected_series, **CLI_TEXT_TOL)
    np.testing.assert_allclose(cli_rows[:, 1], expected_residuals, **CLI_TEXT_TOL)


def test_output_file_matches_stdout(tmp_path):
    datafile = "tests/refs/ar-run_l1000.txt"
    poles = 3

    stdout_out = run_cli([f"-p{poles}", "-V0", datafile])

    outfile = tmp_path / "out.ari"
    subprocess.run(
        ["./bin/arima-model", f"-p{poles}", "-V0", f"-o{outfile}", datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    file_out = outfile.read_text()

    np.testing.assert_allclose(
        parse_data_rows(stdout_out), parse_data_rows(file_out), **CLI_TEXT_TOL
    )
    np.testing.assert_allclose(
        parse_coeff_rows(stdout_out), parse_coeff_rows(file_out), **CLI_TEXT_TOL
    )


@pytest.mark.parametrize("ilength", [1, 20])
def test_iterate_matches_cli_dash_s_plain_ar(ilength):
    datafile = "tests/refs/ar-run_l1000.txt"
    poles = 3

    out = run_cli([f"-p{poles}", f"-s{ilength}", datafile])
    cli_iterated = parse_data_rows(out)
    assert cli_iterated.shape == (ilength, 1)

    # rand.c's rnd_init() only actually (re)seeds once per process (see
    # tests/test_ar_model_bindings.py's test_ar_model_iterate_is_deterministic_given_a_seed),
    # so run the Python side in a fresh subprocess to get a comparably fresh
    # RNG state to the CLI's own subprocess.
    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({datafile!r})\n"
        "series = raw.reshape(1, -1)\n"
        f"model = tisean.arima_model.fit(series, poles={poles})\n"
        f"print(model.iterate({ilength}).tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_iterated = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_iterated, cli_iterated, **CLI_TEXT_TOL)


@pytest.mark.parametrize("ilength", [1, 20])
def test_iterate_matches_cli_dash_s_arma(ilength):
    datafile = "tests/refs/henon_l1000.txt"

    out = run_cli(["-m2", "-p10", "-P2,0,1", f"-s{ilength}", datafile])
    cli_iterated = parse_data_rows(out)
    assert cli_iterated.shape == (ilength, 2)

    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({datafile!r})[:, [0, 1]]\n"
        "series = raw.T.copy()\n"
        "model = tisean.arima_model.fit(series, poles=10, arpoles=2, ipoles=0, mapoles=1)\n"
        f"print(model.iterate({ilength}).tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_iterated = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_iterated, cli_iterated, **CLI_TEXT_TOL)


def test_iterate_default_seed_matches_cli_hardcoded_seed():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles, ilength = 3, 5

    out = run_cli([f"-p{poles}", f"-s{ilength}", datafile])
    cli_iterated = parse_data_rows(out)

    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({datafile!r})\n"
        "series = raw.reshape(1, -1)\n"
        f"model = tisean.arima_model.fit(series, poles={poles})\n"
        f"print(model.iterate({ilength}).tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_iterated = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_iterated, cli_iterated, **CLI_TEXT_TOL)


def test_fit_rejects_too_many_poles_like_cli():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles, length = 5, 5

    result = subprocess.run(
        ["./bin/arima-model", f"-p{poles}", f"-l{length}", datafile],
        capture_output=True,
        text=True,
    )
    assert result.returncode == AR_MODEL_TOO_MANY_POLES

    series = load_columns(datafile, [1], length=length)
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=poles)


def test_fit_rejects_too_many_arma_poles_like_cli():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles, arpoles, length = 2, 5, 5

    result = subprocess.run(
        ["./bin/arima-model", f"-p{poles}", f"-P{arpoles},0,0", f"-l{length}", datafile],
        capture_output=True,
        text=True,
    )
    assert result.returncode == AR_MODEL_TOO_MANY_POLES

    series = load_columns(datafile, [1], length=length)
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=poles, arpoles=arpoles)


def test_fit_rejects_zero_variance_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.0"] * 50) + "\n")

    result = subprocess.run(
        ["./bin/arima-model", "-p3", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.ones((1, 50))
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=3)


def test_iterations_must_be_positive():
    series = load_columns("tests/refs/ar-run_l1000.txt", [1])
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=3, iterations=0)
