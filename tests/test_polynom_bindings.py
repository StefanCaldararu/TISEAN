import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# output file can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23

POLYNOM_BIN = os.path.abspath("./bin/polynom")
DATAFILE = "tests/refs/ar-run_l1000.txt"
HENON_FILE = "tests/refs/henon_l1000.txt"


def run_cli(flags, datafile, outfile, check=True):
    # polynom always writes to a file (never stdout), so every invocation
    # gets an explicit -o pointing at a path we can read back.
    return subprocess.run(
        [POLYNOM_BIN] + list(flags) + ["-o", str(outfile), datafile],
        capture_output=True,
        text=True,
        check=check,
    )


def load_column(path, column=1, length=None, exclude=0):
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


def parse_output(text, dim):
    """Parses polynom's output-file format: a '#number of free
    parameters=' header, a '#used norm for the fit=' line, one
    '#e1 .. edim coeff' line per term, an '#average insample error=' line,
    an optional '#average out of sample error=' line, and then one
    forecasted value per remaining (non-#) line."""
    plength = norm = error_insample = error_outsample = None
    exponent = []
    coeff = []
    forecast = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#number of free parameters="):
            plength = int(line.split("=")[1])
        elif line.startswith("#used norm for the fit="):
            norm = float(line.split("=")[1])
        elif line.startswith("#average insample error="):
            error_insample = float(line.split("=")[1])
        elif line.startswith("#average out of sample error="):
            error_outsample = float(line.split("=")[1])
        elif line.startswith("#"):
            parts = line[1:].split()
            exponent.append([int(x) for x in parts[:dim]])
            coeff.append(float(parts[dim]))
        else:
            forecast.append(float(line.strip()))
    return {
        "plength": plength,
        "norm": norm,
        "exponent": np.array(exponent, dtype=int),
        "coeff": np.array(coeff),
        "error_insample": error_insample,
        "error_outsample": error_outsample,
        "forecast": np.array(forecast),
    }


CASES = [
    # label, datafile, column, dim, delay, order, insample, step, length, exclude
    ("defaults_all_data", DATAFILE, 1, 2, 1, 2, None, 0, None, 0),
    ("small_dim1_order2", DATAFILE, 1, 1, 1, 2, 300, 50, None, 0),
    ("dim3_order2_delay2", DATAFILE, 1, 3, 2, 2, 400, 30, None, 0),
    ("order0_constant_only", DATAFILE, 1, 2, 1, 0, 200, 20, None, 0),
    ("insample_equal_length_no_outsample", DATAFILE, 1, 2, 1, 2, 1000, 20, None, 0),
    ("length_and_exclude", DATAFILE, 1, 2, 1, 2, 300, 40, 500, 100),
    ("column2_henon", HENON_FILE, 2, 2, 1, 2, 300, 25, None, 0),
    ("step_zero_no_cast", DATAFILE, 1, 2, 1, 2, 300, 0, None, 0),
]


@pytest.mark.parametrize(
    "label,datafile,column,dim,delay,order,insample,step,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_fit_matches_cli(
    tmp_path, label, datafile, column, dim, delay, order, insample, step, length, exclude
):
    outfile = tmp_path / "out.pol"
    args = [f"-m{dim}", f"-d{delay}", f"-p{order}", f"-c{column}"]
    if insample is not None:
        args.append(f"-n{insample}")
    if length is not None:
        args.append(f"-l{length}")
    if exclude:
        args.append(f"-x{exclude}")
    if step:
        args.append(f"-L{step}")

    run_cli(args, datafile, outfile)
    parsed = parse_output(outfile.read_text(), dim)

    series = load_column(datafile, column=column, length=length, exclude=exclude)

    kwargs = dict(dim=dim, delay=delay, order=order, step=step)
    if insample is not None:
        kwargs["insample"] = insample
    result = tisean.polynom.fit(series, **kwargs)

    assert result.dim == dim
    assert result.delay == delay
    assert result.order == order
    assert result.plength == parsed["plength"]
    expected_has_outsample = insample is not None and insample < len(series)
    assert result.has_outsample == expected_has_outsample

    np.testing.assert_array_equal(result.exponent, parsed["exponent"])
    np.testing.assert_allclose(result.coeff, parsed["coeff"], **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.norm, parsed["norm"], **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.error_insample, parsed["error_insample"], **CLI_TEXT_TOL)
    if expected_has_outsample:
        np.testing.assert_allclose(
            result.error_outsample, parsed["error_outsample"], **CLI_TEXT_TOL
        )

    if step:
        np.testing.assert_allclose(result.forecast, parsed["forecast"], **CLI_TEXT_TOL)
    else:
        assert result.forecast is None


def test_defaults_match_cli_defaults(tmp_path):
    # No -m/-d/-p/-n/-L at all: CLI defaults are -m2, -d1, -p2, -n unset
    # (all data used, no out-of-sample error), no cast.
    outfile = tmp_path / "defaults.pol"
    run_cli([], DATAFILE, outfile)
    parsed = parse_output(outfile.read_text(), dim=2)

    series = load_column(DATAFILE)
    result = tisean.polynom.fit(series)

    assert result.dim == 2
    assert result.delay == 1
    assert result.order == 2
    assert result.has_outsample is False
    assert result.forecast is None
    np.testing.assert_array_equal(result.exponent, parsed["exponent"])
    np.testing.assert_allclose(result.coeff, parsed["coeff"], **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.error_insample, parsed["error_insample"], **CLI_TEXT_TOL)


def test_verbosity_does_not_change_output(tmp_path):
    out0 = tmp_path / "out0.pol"
    out1 = tmp_path / "out1.pol"
    run_cli(["-m2", "-d1", "-p2", "-n300", "-L50", "-V0"], DATAFILE, out0)
    run_cli(["-m2", "-d1", "-p2", "-n300", "-L50", "-V1"], DATAFILE, out1)
    assert out0.read_text() == out1.read_text()


def test_default_outfile_name(tmp_path):
    # No -o: the CLI defaults to '<datafile>.pol'.
    local_datafile = tmp_path / "data.txt"
    local_datafile.write_bytes(open(DATAFILE, "rb").read())

    subprocess.run(
        [POLYNOM_BIN, "-m2", "-d1", "-p2", str(local_datafile)],
        capture_output=True,
        text=True,
        check=True,
    )

    default_outfile = tmp_path / "data.txt.pol"
    assert default_outfile.exists()
    parsed = parse_output(default_outfile.read_text(), dim=2)

    series = load_column(str(local_datafile))
    result = tisean.polynom.fit(series, dim=2, delay=1, order=2)
    np.testing.assert_allclose(result.coeff, parsed["coeff"], **CLI_TEXT_TOL)


def test_zero_variance_rejected_like_cli(tmp_path):
    series = np.full(50, 3.0)
    series_path = tmp_path / "const.txt"
    np.savetxt(series_path, series)
    outfile = tmp_path / "const.pol"

    result = run_cli(["-m2", "-d1", "-p2"], str(series_path), outfile, check=False)
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    with pytest.raises(ValueError):
        tisean.polynom.fit(series, dim=2, delay=1, order=2)


def test_result_shapes_and_dtype():
    series = load_column(DATAFILE)
    result = tisean.polynom.fit(series, dim=2, delay=1, order=3, insample=300, step=40)

    assert result.coeff.shape == (result.plength,)
    assert result.exponent.shape == (result.plength, result.dim)
    assert result.forecast.shape == (40,)
    assert result.coeff.dtype == np.float64
    assert np.issubdtype(result.exponent.dtype, np.integer)
