import subprocess
import sys

import numpy as np
import pytest

import tisean

AR_MODEL_TOO_MANY_POLES = 52

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows - rtol=1e-7
# (float64's own precision) is tighter than the CLI's own output format can
# ever satisfy.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"


def run_cli(args):
    result = subprocess.run(
        ["./bin/arima-model"] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_data_rows(text):
    """Rows of plain (non-comment) numeric output: residuals or an
    iterated model, depending on whether -s was passed."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)


def parse_coeff_rows(text):
    """The '#x_i(n-k) c1 c2 ... cdim' / '#e_i(n-k) ...' coefficient rows
    (`size` of them), shape (size, dim) - i.e. the transpose of
    ArimaModel.coeff."""
    rows = []
    for line in text.splitlines():
        if line.startswith("#x_") or line.startswith("#e_"):
            rows.append([float(x) for x in line.split()[1:]])
    return np.array(rows)


def parse_convergence(text, dim):
    """The '#iteration k <dim xdiff values> <diffcoeff>' rows, returning
    (xdiff, diffcoeff) with shapes (realiter, dim) and (realiter,)."""
    xdiff, diffcoeff = [], []
    for line in text.splitlines():
        if line.startswith("#iteration"):
            vals = [float(x) for x in line.split()[2:]]
            xdiff.append(vals[:dim])
            diffcoeff.append(vals[dim])
    return np.array(xdiff), np.array(diffcoeff)


def parse_rms_error(text):
    for line in text.splitlines():
        if line.startswith("#individual forecast errors:"):
            return np.array([float(x) for x in line.split()[3:]])
    raise AssertionError("no '#individual forecast errors:' line found")


def variance_mean(row):
    """Mirrors variance.c's mean: a plain sequential sum divided by the
    count. numpy's .mean() (pairwise summation) rounds differently in the
    last bit or two, which is invisible for one pass but can visibly
    diverge after enough iterations of the ARMA refinement loop."""
    total = 0.0
    for v in row:
        total += v
    return total / len(row)


def load_columns(path, columns, length=None, exclude=0, ipoles=0):
    """Replicates get_multi_series()'s -x/-l/-c handling plus the CLI's own
    make_difference() (-P's I-order, applied ipoles times before anything
    else sees the data) and set_averages_to_zero() centering. Returns a
    (dim, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    series = raw.T.copy()

    for _ in range(ipoles):
        series[:, 1:] = series[:, 1:] - series[:, :-1]
    if ipoles:
        series = series[:, ipoles:].copy()

    for row in series:
        row -= variance_mean(row)
    return series


# label, datafile, columns, poles, arpoles, mapoles
#
# arima-model's own -c flag (unlike ar-model's) only accepts a single
# unsigned integer, not a comma list (check_option(argv,argc,'c','u') in
# source_c/arima-model.c - a preexisting CLI quirk, not something this
# extraction changed), so every multi-dim case here relies on -m's default
# column selection (1..dim, in order) instead of passing -c explicitly.
FIT_CASES = [
    ("dim1_p1", AR_RUN, [1], 1, 0, 0),
    ("dim1_p5", AR_RUN, [1], 5, 0, 0),
    ("dim1_p10_ar2", AR_RUN, [1], 10, 2, 0),
    ("dim1_p10_ma1", AR_RUN, [1], 10, 0, 1),
    ("dim1_p10_arma2_1", AR_RUN, [1], 10, 2, 1),
    ("dim2_p1", HENON, [1, 2], 1, 0, 0),
    ("dim2_p10_arma2_1", HENON, [1, 2], 10, 2, 1),
    ("dim3_p2", LORENZ, [1, 2, 3], 2, 0, 0),
    ("dim3_p6_arma2_1", LORENZ, [1, 2, 3], 6, 2, 1),
]


@pytest.mark.parametrize(
    "label,datafile,columns,poles,arpoles,mapoles", FIT_CASES, ids=[c[0] for c in FIT_CASES]
)
def test_fit_matches_cli_across_dim_poles_and_arma_order(
    label, datafile, columns, poles, arpoles, mapoles
):
    dim = len(columns)
    assert columns == list(range(1, dim + 1))
    args = [f"-m{dim}", f"-p{poles}"]
    if arpoles or mapoles:
        args += [f"-P{arpoles},0,{mapoles}"]
    args += ["-V0", datafile]

    out = run_cli(args)
    cli_coeff = parse_coeff_rows(out)
    cli_residuals = parse_data_rows(out)
    cli_rms = parse_rms_error(out)

    series = load_columns(datafile, columns)
    model = tisean.arima_model.fit(series, poles=poles, arpoles=arpoles, mapoles=mapoles)

    assert model.dim == dim
    assert model.poles == poles
    assert model.arpoles == arpoles
    assert model.mapoles == mapoles
    assert model.is_arima == bool(arpoles or mapoles)
    expected_size = (arpoles + mapoles) * dim if (arpoles or mapoles) else dim * poles
    assert model.size == expected_size
    assert model.coeff.shape == (dim, expected_size)

    np.testing.assert_allclose(model.coeff.T, cli_coeff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(model.rms_error, cli_rms, **CLI_TEXT_TOL)
    np.testing.assert_allclose(
        model.residuals[:, model.iter_poles :].T, cli_residuals, **CLI_TEXT_TOL
    )

    if arpoles or mapoles:
        cli_xdiff, cli_diffcoeff = parse_convergence(out, dim)
        assert model.realiter == cli_xdiff.shape[0]
        np.testing.assert_allclose(model.xdiff, cli_xdiff, **CLI_TEXT_TOL)
        np.testing.assert_allclose(model.diffcoeff, cli_diffcoeff, **CLI_TEXT_TOL)
    else:
        assert model.realiter == 0
        assert model.xdiff.shape == (0, dim)
        assert model.diffcoeff.shape == (0,)


def test_fit_matches_cli_with_length_and_exclude():
    poles, length, exclude = 3, 400, 100

    out = run_cli([f"-p{poles}", f"-l{length}", f"-x{exclude}", "-V0", AR_RUN])
    cli_residuals = parse_data_rows(out)

    series = load_columns(AR_RUN, [1], length=length, exclude=exclude)
    model = tisean.arima_model.fit(series, poles=poles)

    assert model.length == length
    np.testing.assert_allclose(
        model.residuals[:, model.iter_poles :].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_fit_matches_cli_with_differencing():
    # Differencing (-P's I-order) is applied by the CLI before anything else
    # sees the data (make_difference(), see source_c/arima-model.c), so it's
    # not part of the reentrant fit() API itself - replicate it in Python
    # via load_columns(..., ipoles=...) instead. Pair it with a real (non-
    # degenerate) ARMA order: arpoles=mapoles=0 alone would make the CLI run
    # its ARMA-refinement loop with a zero-sized coefficient vector, which
    # reads memory this reimplementation deliberately leaves at a
    # deterministic zero instead of the CLI's uninitialized garbage (see
    # ArimaModel.rms_error's docstring in include/arima_model.h) - not a
    # case this test can meaningfully cross-check byte-for-byte.
    poles, ipoles, arpoles, mapoles = 4, 1, 2, 1

    out = run_cli([f"-p{poles}", f"-P{arpoles},{ipoles},{mapoles}", "-V0", AR_RUN])
    cli_residuals = parse_data_rows(out)
    cli_coeff = parse_coeff_rows(out)

    series = load_columns(AR_RUN, [1], ipoles=ipoles)
    model = tisean.arima_model.fit(series, poles=poles, arpoles=arpoles, mapoles=mapoles)

    np.testing.assert_allclose(model.coeff.T, cli_coeff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(
        model.residuals[:, model.iter_poles :].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_fit_matches_cli_default_poles_and_no_arma():
    # No -p/-P at all: CLI defaults are poles=10, arpoles=mapoles=0.
    out = run_cli(["-V0", AR_RUN])
    cli_residuals = parse_data_rows(out)

    series = load_columns(AR_RUN, [1])
    model = tisean.arima_model.fit(series)

    assert model.poles == 10
    assert model.arpoles == 0
    assert model.mapoles == 0
    assert not model.is_arima
    np.testing.assert_allclose(
        model.residuals[:, model.iter_poles :].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_fit_default_convergence_and_max_iter_match_cli():
    # Explicit -I50 -e0.001 (the CLI's own documented defaults) must give
    # the same result as omitting -I/-e altogether, and the same as the
    # binding's own defaults.
    poles, arpoles, mapoles = 8, 2, 1
    args_explicit = [f"-p{poles}", f"-P{arpoles},0,{mapoles}", "-I50", "-e0.001", "-V0", AR_RUN]
    args_default = [f"-p{poles}", f"-P{arpoles},0,{mapoles}", "-V0", AR_RUN]

    assert run_cli(args_explicit) == run_cli(args_default)

    series = load_columns(AR_RUN, [1])
    model_explicit = tisean.arima_model.fit(
        series, poles=poles, arpoles=arpoles, mapoles=mapoles, max_iter=50, convergence=1.0e-3
    )
    model_default = tisean.arima_model.fit(series, poles=poles, arpoles=arpoles, mapoles=mapoles)

    np.testing.assert_array_equal(model_explicit.coeff, model_default.coeff)
    assert model_explicit.realiter == model_default.realiter


@pytest.mark.parametrize("ilength", [1, 25])
def test_iterate_matches_cli_dash_s_with_default_seed(ilength):
    poles, arpoles, mapoles = 5, 2, 1
    columns = [1, 2]

    out = run_cli(
        ["-m2", f"-p{poles}", f"-P{arpoles},0,{mapoles}", f"-s{ilength}", "-V0", HENON]
    )
    cli_iterated = parse_data_rows(out)
    assert cli_iterated.shape == (ilength, 2)

    # rand.c's rnd_init() only actually (re)seeds once per process (see
    # test_ar_model_bindings.py's identically-motivated test), so comparing
    # against a specific seed must run the Python side in its own fresh
    # subprocess instead of reusing this pytest process's RNG state.
    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({HENON!r})[:, {[c - 1 for c in columns]!r}]\n"
        "series = raw.T.copy()\n"
        "for row in series:\n"
        "    total = 0.0\n"
        "    for v in row:\n"
        "        total += v\n"
        "    row -= total / len(row)\n"
        f"model = tisean.arima_model.fit(series, poles={poles}, arpoles={arpoles}, "
        f"mapoles={mapoles})\n"
        f"print(model.iterate({ilength}).tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_iterated = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_iterated, cli_iterated, **CLI_TEXT_TOL)


def test_iterate_matches_cli_dash_s_plain_ar_no_arma():
    poles, ilength = 3, 10

    out = run_cli([f"-p{poles}", f"-s{ilength}", "-V0", AR_RUN])
    cli_iterated = parse_data_rows(out)

    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({AR_RUN!r})\n"
        "series = raw.reshape(1, -1)\n"
        "total = 0.0\n"
        "for v in series[0]:\n"
        "    total += v\n"
        "series[0] -= total / series.shape[1]\n"
        f"model = tisean.arima_model.fit(series, poles={poles})\n"
        f"print(model.iterate({ilength}).tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_iterated = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_iterated, cli_iterated, **CLI_TEXT_TOL)


def test_iterate_is_deterministic_given_a_seed():
    # Same one-shot-seed caveat as ar-model (see rand.c's rnd_init()):
    # check determinism across fresh processes, not within one.
    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({AR_RUN!r})[:1000]\n"
        "series = (raw - raw.mean()).reshape(1, -1)\n"
        "model = tisean.arima_model.fit(series, poles=3, arpoles=2, mapoles=1)\n"
        "print(model.iterate(5, seed=0x44325).tolist())\n"
    )
    a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert a.stdout == b.stdout


def test_residuals_before_iter_poles_are_zeroed_not_garbage():
    raw = np.loadtxt(AR_RUN)[:1000]
    series = (raw - raw.mean()).reshape(1, -1)

    model = tisean.arima_model.fit(series, poles=5)

    assert model.iter_poles == 5
    np.testing.assert_array_equal(model.residuals[0, :5], np.zeros(5))


def test_fit_rejects_too_many_poles_like_cli():
    poles, length = 5, 5

    result = subprocess.run(
        ["./bin/arima-model", f"-p{poles}", f"-l{length}", "-V0", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == AR_MODEL_TOO_MANY_POLES

    series = load_columns(AR_RUN, [1], length=length)
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=poles)


def test_fit_rejects_too_many_arma_poles_like_cli():
    poles, arpoles, length = 3, 10, 10

    result = subprocess.run(
        ["./bin/arima-model", f"-p{poles}", f"-P{arpoles},0,0", f"-l{length}", "-V0", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == AR_MODEL_TOO_MANY_POLES

    series = load_columns(AR_RUN, [1], length=length)
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=poles, arpoles=arpoles)


def test_fit_rejects_zero_poles():
    series = load_columns(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.arima_model.fit(series, poles=0)
