import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# figures), so comparisons against numbers parsed from its stdout can never
# be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
LZO_GM_BIN = os.path.abspath("./bin/lzo-gm")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [LZO_GM_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text, dim):
    """Rows of plain numeric output (5 + dim - 1 = dim + 4 columns:
    epsilon, avg_error, error[0..dim-1], fraction, avneighbors)."""
    ncols = dim + 4
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != ncols:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return np.array(rows)


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


# label, args, datafile, columns, dim, embed, delay, step, iterations
CASES = [
    ("defaults_dim1", ["-m1,2", "-i50"], AR_RUN, [1], 1, 2, 1, 1, 50),
    ("embed3", ["-m1,3", "-i50"], AR_RUN, [1], 1, 3, 1, 1, 50),
    ("delay2", ["-m1,2", "-d2", "-i50"], AR_RUN, [1], 1, 2, 2, 1, 50),
    ("step2", ["-m1,2", "-i50", "-s2"], AR_RUN, [1], 1, 2, 1, 2, 50),
    ("dim2_henon", ["-m2,3", "-i100"], HENON, [1, 2], 2, 3, 1, 1, 100),
    ("dim3_lorenz", ["-m3,2", "-i100"], LORENZ, [1, 2, 3], 3, 2, 1, 1, 100),
    ("iterations_default", ["-m1,2"], AR_RUN, [1], 1, 2, 1, 1, None),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,dim,embed,delay,step,iterations",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli_across_dim_embed_delay_step(
    label, args, datafile, columns, dim, embed, delay, step, iterations
):
    out = run_cli(args + [datafile])
    cli_rows = parse_output(out, dim)

    series = load_multi_series(datafile, columns)
    result = tisean.lzo_gm.compute(
        series, embed=embed, delay=delay, step=step, iterations=iterations
    )

    assert result.dim == dim
    assert result.n_rows == len(cli_rows)
    py_rows = np.column_stack(
        [
            result.epsilon,
            result.avg_error,
            result.error.reshape(result.n_rows, dim),
            result.fraction,
            result.avneighbors,
        ]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_matches_cli_with_causal():
    datafile = AR_RUN
    out = run_cli(["-m1,2", "-i50", "-s2", "-C3", datafile])
    cli_rows = parse_output(out, 1)

    series = load_multi_series(datafile, [1])
    result = tisean.lzo_gm.compute(series, embed=2, delay=1, step=2, causal=3, iterations=50)

    py_rows = np.column_stack(
        [result.epsilon, result.avg_error, result.error.reshape(-1, 1), result.fraction, result.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_causal_defaults_to_step():
    datafile = AR_RUN
    out = run_cli(["-m1,2", "-i50", "-s2", datafile])
    cli_rows = parse_output(out, 1)

    series = load_multi_series(datafile, [1])
    result = tisean.lzo_gm.compute(series, embed=2, delay=1, step=2, iterations=50)

    py_rows = np.column_stack(
        [result.epsilon, result.avg_error, result.error.reshape(-1, 1), result.fraction, result.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_matches_cli_with_raw_eps_range():
    datafile = AR_RUN
    out = run_cli(["-m1,2", "-i50", "-r0.01", "-R0.5", datafile])
    cli_rows = parse_output(out, 1)

    series = load_multi_series(datafile, [1])
    result = tisean.lzo_gm.compute(
        series, embed=2, delay=1, iterations=50,
        eps0=0.01, eps0_raw=True, eps1=0.5, eps1_raw=True,
    )

    py_rows = np.column_stack(
        [result.epsilon, result.avg_error, result.error.reshape(-1, 1), result.fraction, result.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_matches_cli_with_epsf():
    datafile = AR_RUN
    out = run_cli(["-m1,2", "-i50", "-f1.5", datafile])
    cli_rows = parse_output(out, 1)

    series = load_multi_series(datafile, [1])
    result = tisean.lzo_gm.compute(series, embed=2, delay=1, iterations=50, epsf=1.5)

    py_rows = np.column_stack(
        [result.epsilon, result.avg_error, result.error.reshape(-1, 1), result.fraction, result.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_matches_cli_with_length_and_exclude():
    datafile = AR_RUN
    length, exclude = 500, 100
    out = run_cli(["-m1,2", "-i50", f"-l{length}", f"-x{exclude}", datafile])
    cli_rows = parse_output(out, 1)

    series = load_multi_series(datafile, [1], length=length, exclude=exclude)
    result = tisean.lzo_gm.compute(series, embed=2, delay=1, iterations=50)

    py_rows = np.column_stack(
        [result.epsilon, result.avg_error, result.error.reshape(-1, 1), result.fraction, result.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.lm"
    run_cli(["-m1,2", "-i50", "-o" + str(outfile), AR_RUN])

    cli_rows = parse_output(outfile.read_text(), 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lzo_gm.compute(series, embed=2, delay=1, iterations=50)

    py_rows = np.column_stack(
        [result.epsilon, result.avg_error, result.error.reshape(-1, 1), result.fraction, result.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_defaults():
    # No -m/-d/-s/-C/-r/-R/-f at all: dim=1 (via column selection), embed=2,
    # delay=1, step=1, causal=step, eps0=1e-3, eps1=1.0, epsf=1.2.
    datafile = AR_RUN
    out = run_cli(["-i50", datafile])
    cli_rows = parse_output(out, 1)

    series = load_multi_series(datafile, [1])
    default = tisean.lzo_gm.compute(series, iterations=50)
    explicit = tisean.lzo_gm.compute(
        series, embed=2, delay=1, step=1, causal=1, iterations=50,
        eps0=1.0e-3, eps0_raw=False, eps1=1.0, eps1_raw=False, epsf=1.2,
    )

    np.testing.assert_array_equal(default.epsilon, explicit.epsilon)
    py_rows = np.column_stack(
        [default.epsilon, default.avg_error, default.error.reshape(-1, 1),
         default.fraction, default.avneighbors]
    )
    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_compute_rejects_constant_dimension_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 30) + "\n")

    result = subprocess.run(
        [LZO_GM_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series = np.full((1, 30), 1.5)
    with pytest.raises(ValueError):
        tisean.lzo_gm.compute(series)


def test_compute_rejects_step_at_or_beyond_length():
    series = load_multi_series(AR_RUN, [1], length=20)
    with pytest.raises(ValueError):
        tisean.lzo_gm.compute(series, step=20)


def test_compute_rejects_embed_zero():
    series = load_multi_series(AR_RUN, [1], length=100)
    with pytest.raises(ValueError):
        tisean.lzo_gm.compute(series, embed=0)


def test_compute_rejects_iterations_less_than_step():
    series = load_multi_series(AR_RUN, [1], length=100)
    with pytest.raises(ValueError):
        tisean.lzo_gm.compute(series, step=5, iterations=2)
