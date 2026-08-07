import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# figures), so comparisons against numbers parsed from its stdout can never
# be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LZO_TEST_BIN = os.path.abspath("./bin/lzo-test")


def run_cli(args, datafile, tmp_path=None, name="out.zer"):
    if tmp_path is not None:
        outfile = tmp_path / name
        subprocess.run(
            [LZO_TEST_BIN] + args + ["-o", str(outfile), datafile],
            capture_output=True,
            text=True,
            check=True,
        )
        return outfile.read_text()
    result = subprocess.run(
        [LZO_TEST_BIN] + args + [datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_output(text, dim):
    """Splits lzo-test's stdout into (error_rows, diffs_rows).

    error_rows is (step, dim): rows are "<horizon> <err_1> ... <err_dim>",
    optionally prefixed with "#" (verbosity & VER_USR1, i.e. -V2/-V3).
    diffs_rows is (n_ref, dim): only present under -V2/-V3, plain
    "<v_1> ... <v_dim>" rows with no leading index/hash. The two blocks are
    told apart by token count (dim+1 for an error row vs dim for a diffs
    row), which is unambiguous for any dim.
    """
    error_rows, diffs_rows = [], []
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "#":
            parts = parts[1:]
        if len(parts) == dim + 1:
            error_rows.append([float(p) for p in parts[1:]])
        elif len(parts) == dim:
            diffs_rows.append([float(p) for p in parts])
    return np.array(error_rows).reshape(-1, dim), np.array(diffs_rows).reshape(-1, dim)


def load_multi_series(path, columns, length=None, exclude=0):
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


# label, cli_args, datafile, columns, length, exclude, kwargs (embed/delay/
# minn/step/refstep/causal/n_ref/eps0/epsset/epsf - only overrides from
# tisean.lzo_test.compute's defaults need to be listed)
CASES = [
    ("defaults", ["-m1,2", "-d1", "-n50", "-k10"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=50)),
    ("embed_3", ["-m1,3", "-d1", "-n50", "-k10"], AR_RUN, [1], None, 0,
     dict(embed=3, minn=10, n_ref=50)),
    ("delay_2", ["-m1,2", "-d2", "-n50", "-k10"], AR_RUN, [1], None, 0,
     dict(delay=2, minn=10, n_ref=50)),
    ("minn_5", ["-m1,2", "-d1", "-n50", "-k5"], AR_RUN, [1], None, 0,
     dict(minn=5, n_ref=50)),
    ("step_2", ["-m1,2", "-d1", "-n80", "-k10", "-s2"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=80, step=2)),
    ("refstep_2", ["-m1,2", "-d1", "-n40", "-k10", "-S2"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=40, refstep=2)),
    ("causal_5", ["-m1,2", "-d1", "-n50", "-k10", "-C5"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=50, causal=5)),
    ("eps0_raw", ["-m1,2", "-d1", "-n50", "-k10", "-r0.01"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=50, eps0=0.01, epsset=True)),
    ("epsf_1_5", ["-m1,2", "-d1", "-n50", "-k10", "-f1.5"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=50, epsf=1.5)),
    ("multivariate_henon", ["-m2,2", "-d1", "-n50", "-k10", "-c1,2"], HENON, [1, 2], None, 0,
     dict(minn=10, n_ref=50)),
    ("length_and_exclude", ["-m1,2", "-d1", "-n50", "-k10", "-l500", "-x100"], AR_RUN, [1], 500, 100,
     dict(minn=10, n_ref=50)),
    ("verbosity_0", ["-m1,2", "-d1", "-n50", "-k10", "-V0"], AR_RUN, [1], None, 0,
     dict(minn=10, n_ref=50)),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, columns, length, exclude, kwargs):
    cli_text = run_cli(args, datafile)
    dim = len(columns)
    cli_error, _ = parse_output(cli_text, dim)

    series = load_multi_series(datafile, columns, length=length, exclude=exclude)
    result = tisean.lzo_test.compute(series, **kwargs)

    assert result.dim == dim
    py_error = result.error.reshape(result.step, dim)
    assert py_error.shape == cli_error.shape
    np.testing.assert_allclose(py_error, cli_error, **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    cli_text = run_cli(["-m1,2", "-d1", "-n50", "-k10"], AR_RUN, tmp_path=tmp_path)
    cli_error, _ = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lzo_test.compute(series, minn=10, n_ref=50)

    py_error = result.error.reshape(result.step, 1)
    np.testing.assert_allclose(py_error, cli_error, **CLI_TEXT_TOL)


def test_compute_matches_cli_diffs_under_verbosity_2():
    args = ["-m1,2", "-d1", "-n50", "-k10", "-s2", "-V2"]
    cli_text = run_cli(args, AR_RUN)
    cli_error, cli_diffs = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lzo_test.compute(series, minn=10, n_ref=50, step=2)

    py_error = result.error.reshape(result.step, 1)
    np.testing.assert_allclose(py_error, cli_error, **CLI_TEXT_TOL)

    assert result.n_ref == len(cli_diffs)
    py_diffs = result.diffs.reshape(result.n_ref, 1)
    np.testing.assert_allclose(py_diffs, cli_diffs, **CLI_TEXT_TOL)


def test_compute_causal_defaults_to_step():
    args = ["-m1,2", "-d1", "-n50", "-k10", "-s2"]
    cli_text = run_cli(args, AR_RUN)
    cli_error, _ = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lzo_test.compute(series, minn=10, n_ref=50, step=2)

    py_error = result.error.reshape(result.step, 1)
    np.testing.assert_allclose(py_error, cli_error, **CLI_TEXT_TOL)


def test_compute_n_ref_defaults_to_whole_series():
    # No -n given: CLENGTH defaults to LENGTH (the CLI's "default: length").
    args = ["-m1,2", "-d1", "-k10", "-l300"]
    cli_text = run_cli(args, AR_RUN)
    cli_error, _ = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1], length=300)
    result = tisean.lzo_test.compute(series, minn=10)

    py_error = result.error.reshape(result.step, 1)
    np.testing.assert_allclose(py_error, cli_error, **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default embed=2 (dim comes from series.shape[0]),
    # -d default 1, -n default "length", -S default 1, -k default 30,
    # -s default 1, -C default "steps", -r/-f default 1.e-3/1.2.
    series = load_multi_series(AR_RUN, [1], length=300)

    default = tisean.lzo_test.compute(series)
    explicit = tisean.lzo_test.compute(
        series, embed=2, delay=1, minn=30, step=1, refstep=1, causal=None,
        n_ref=None, eps0=1.0e-3, epsset=False, epsf=1.2,
    )

    assert default.dim == explicit.dim
    assert default.step == explicit.step
    assert default.n_ref == explicit.n_ref
    np.testing.assert_array_equal(default.error, explicit.error)
    np.testing.assert_array_equal(default.diffs, explicit.diffs)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 60) + "\n")

    result = subprocess.run(
        [LZO_TEST_BIN, "-m1,2", "-d1", "-n50", "-k10", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 5.000000e+00 to 5.000000e+00" in result.stderr

    series = np.full((1, 60), 5.0)
    with pytest.raises(ValueError):
        tisean.lzo_test.compute(series, minn=10, n_ref=50)


def test_compute_rejects_2d_shape_mismatch():
    series = load_multi_series(AR_RUN, [1], length=500).reshape(-1)
    with pytest.raises(ValueError):
        tisean.lzo_test.compute(series)


def test_compute_rejects_embed_zero():
    series = load_multi_series(AR_RUN, [1], length=500)
    with pytest.raises(ValueError):
        tisean.lzo_test.compute(series, embed=0)


def test_compute_rejects_refstep_zero():
    series = load_multi_series(AR_RUN, [1], length=500)
    with pytest.raises(ValueError):
        tisean.lzo_test.compute(series, refstep=0)


def test_compute_rejects_step_at_or_beyond_length():
    series = load_multi_series(AR_RUN, [1], length=20)
    with pytest.raises(ValueError):
        tisean.lzo_test.compute(series, step=20)
