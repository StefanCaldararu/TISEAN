import subprocess

import numpy as np
import pytest

import tisean

AR_MODEL_TOO_MANY_POLES = 52

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows - confirmed by
# comparing against a full-precision (%.17g) build of the same fit/iterate
# call, which matches the Python bindings exactly. rtol=1e-7 (float64's own
# precision) is tighter than the CLI's own output format can ever satisfy.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)


def run_cli(args):
    result = subprocess.run(
        ["./bin/ar-model"] + args,
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
    """The '# c1 c2 ... cdim' coefficient rows (dim*poles of them),
    shape (dim*poles, dim) - i.e. the transpose of ARModel.coeff."""
    rows = []
    for line in text.splitlines():
        if line.startswith("# "):
            rows.append([float(x) for x in line.split()[1:]])
    return np.array(rows)


def variance_mean(row):
    """Mirrors variance.c's mean: a plain sequential sum divided by the
    count. numpy's .mean() (pairwise summation) and Python's built-in
    sum() (compensated since 3.12) are both more accurate than this and so
    round differently in the last bit or two - which is enough for
    ar-model's iterate() to visibly diverge from the CLI after enough
    steps, since it feeds the fitted coefficients back through themselves
    on every step."""
    total = 0.0
    for v in row:
        total += v
    return total / len(row)


def load_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order), and finally center each row like the CLI's
    set_averages_to_zero() does. Returns a (dim, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    series = raw.T.copy()
    for row in series:
        row -= variance_mean(row)
    return series


FIT_CASES = [
    # label, datafile, poles, columns (1-indexed, defines dim too)
    ("dim1_p1", "tests/refs/ar-run_l1000.txt", 1, [1]),
    ("dim1_p2", "tests/refs/ar-run_l1000.txt", 2, [1]),
    ("dim1_p5", "tests/refs/ar-run_l1000.txt", 5, [1]),
    ("dim2_p1", "tests/refs/henon_l1000.txt", 1, [1, 2]),
    ("dim2_p2", "tests/refs/henon_l1000.txt", 2, [1, 2]),
    ("dim2_p2_reordered_columns", "tests/refs/henon_l1000.txt", 2, [2, 1]),
    ("dim3_p2", "tests/refs/lorenz_l1000.txt", 2, [1, 2, 3]),
    ("dim2_p1_column_subset", "tests/refs/lorenz_l1000.txt", 1, [3, 1]),
]


@pytest.mark.parametrize(
    "label,datafile,poles,columns", FIT_CASES, ids=[c[0] for c in FIT_CASES]
)
def test_fit_matches_cli_across_dim_poles_and_columns(label, datafile, poles, columns):
    dim = len(columns)
    args = [f"-m{dim}", f"-p{poles}", "-c" + ",".join(str(c) for c in columns), datafile]

    out = run_cli(args)
    cli_coeff = parse_coeff_rows(out)
    cli_residuals = parse_data_rows(out)

    series = load_columns(datafile, columns)
    model = tisean.ar_model.fit(series, poles=poles)

    assert model.dim == dim
    assert model.poles == poles
    np.testing.assert_allclose(model.coeff.T, cli_coeff, **CLI_TEXT_TOL)
    np.testing.assert_allclose(
        model.residuals[:, poles:].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_fit_matches_cli_with_length_and_exclude():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles, length, exclude = 3, 400, 100

    out = run_cli([f"-p{poles}", f"-l{length}", f"-x{exclude}", datafile])
    cli_residuals = parse_data_rows(out)

    series = load_columns(datafile, [1], length=length, exclude=exclude)
    model = tisean.ar_model.fit(series, poles=poles)

    assert model.length == length
    np.testing.assert_allclose(
        model.residuals[:, poles:].T, cli_residuals, **CLI_TEXT_TOL
    )


def test_fit_matches_cli_default_dim_and_poles():
    # No -m/-p/-c at all: defaults are dim=1, poles=1, column 1.
    datafile = "tests/refs/ar-run_l1000.txt"

    out = run_cli([datafile])
    cli_residuals = parse_data_rows(out)

    series = load_columns(datafile, [1])
    model = tisean.ar_model.fit(series)

    assert model.dim == 1
    assert model.poles == 1
    np.testing.assert_allclose(
        model.residuals[:, model.poles :].T, cli_residuals, **CLI_TEXT_TOL
    )


@pytest.mark.parametrize("ilength", [1, 25])
def test_iterate_matches_cli_dash_s_with_default_seed(ilength):
    datafile = "tests/refs/henon_l1000.txt"
    poles = 2
    columns = [1, 2]

    out = run_cli(["-m2", f"-p{poles}", f"-s{ilength}", datafile])
    cli_iterated = parse_data_rows(out)
    assert cli_iterated.shape == (ilength, 2)

    # rand.c's rnd_init() only actually (re)seeds once per process (see
    # test_ar_model_iterate_is_deterministic_given_a_seed below), so calling
    # iterate() here in the same pytest process - after other tests/other
    # parametrizations of this same test have already called it - would
    # compare against stale RNG state instead of a fresh seed like the CLI
    # subprocess gets. Do the fit+iterate in its own fresh subprocess too.
    # Centering must match variance_mean()'s plain sequential sum (see its
    # docstring), not a faster/more-accurate mean, or the fitted
    # coefficients pick up last-bit differences that iterate() visibly
    # amplifies over enough steps.
    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({datafile!r})[:, {[c - 1 for c in columns]!r}]\n"
        "series = raw.T.copy()\n"
        "for row in series:\n"
        "    total = 0.0\n"
        "    for v in row:\n"
        "        total += v\n"
        "    row -= total / len(row)\n"
        f"model = tisean.ar_model.fit(series, poles={poles})\n"
        f"print(model.iterate({ilength}).tolist())\n"
    )
    import sys

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_iterated = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_iterated, cli_iterated, **CLI_TEXT_TOL)


def test_ar_model_iterate_is_deterministic_given_a_seed():
    # rand.c's rnd_init() is a process-wide, one-shot seed (guarded by a
    # static was_set flag - see source_c/routines/rand.c), so a second
    # iterate() call in the same process reuses whatever RNG state the
    # first call left behind rather than reseeding. That's inherited
    # behavior from the underlying C library, not something this binding
    # changes, so we check determinism across fresh processes instead of
    # within one.
    script = (
        "import numpy as np, tisean\n"
        "raw = np.loadtxt('tests/refs/ar-run_l1000.txt')[:1000]\n"
        "series = (raw - raw.mean()).reshape(1, -1)\n"
        "model = tisean.ar_model.fit(series, poles=2)\n"
        "print(model.iterate(5, seed=0x44325).tolist())\n"
    )
    import sys

    a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert a.stdout == b.stdout


def test_residuals_before_poles_are_zeroed_not_garbage():
    raw = np.loadtxt("tests/refs/ar-run_l1000.txt")[:1000]
    series = (raw - raw.mean()).reshape(1, -1)

    model = tisean.ar_model.fit(series, poles=3)

    np.testing.assert_array_equal(model.residuals[0, :3], np.zeros(3))


def test_fit_rejects_too_many_poles_like_cli():
    datafile = "tests/refs/ar-run_l1000.txt"
    poles, length = 5, 5

    result = subprocess.run(
        ["./bin/ar-model", f"-p{poles}", f"-l{length}", datafile],
        capture_output=True,
        text=True,
    )
    assert result.returncode == AR_MODEL_TOO_MANY_POLES

    series = load_columns(datafile, [1], length=length)
    with pytest.raises(ValueError):
        tisean.ar_model.fit(series, poles=poles)
