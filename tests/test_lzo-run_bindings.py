import os
import subprocess
import sys

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
LZO_RUN_BIN = os.path.abspath("./bin/lzo-run")


def run_cli(args, datafile):
    result = subprocess.run(
        [LZO_RUN_BIN] + args + [datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_output(text, dim):
    """lzo-run's stdout is pure data: one row of `dim` space-separated %e
    values per iterated forecast point."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != dim:
            continue
        rows.append([float(p) for p in parts])
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


# label, cli_args, datafile, columns, kwargs (only overrides from
# tisean.lzo_run.forecast's defaults need to be listed)
CASES = [
    ("defaults", ["-m1,2", "-d1", "-L50", "-k10", "-I1"], AR_RUN, [1],
     dict(minn=10, flength=50, seed=1)),
    ("embed_3", ["-m1,3", "-d1", "-L50", "-k10", "-I1"], AR_RUN, [1],
     dict(embed=3, minn=10, flength=50, seed=1)),
    ("delay_2", ["-m1,2", "-d2", "-L50", "-k10", "-I1"], AR_RUN, [1],
     dict(delay=2, minn=10, flength=50, seed=1)),
    ("minn_5", ["-m1,2", "-d1", "-L50", "-k5", "-I1"], AR_RUN, [1],
     dict(minn=5, flength=50, seed=1)),
    ("flength_20", ["-m1,2", "-d1", "-L20", "-k10", "-I1"], AR_RUN, [1],
     dict(minn=10, flength=20, seed=1)),
    ("fix_neighbors_dash_K", ["-m1,2", "-d1", "-L50", "-k10", "-K", "-I1"], AR_RUN, [1],
     dict(minn=10, flength=50, seed=1)),
    ("eps0_raw", ["-m1,2", "-d1", "-L20", "-k10", "-I1", "-r0.01"], AR_RUN, [1],
     dict(minn=10, flength=20, seed=1, eps0=0.01, epsset=True)),
    ("epsf_1_5", ["-m1,2", "-d1", "-L20", "-k10", "-I1", "-f1.5"], AR_RUN, [1],
     dict(minn=10, flength=20, seed=1, epsf=1.5)),
    ("multivariate_henon", ["-m2,2", "-d1", "-c1,2", "-L20", "-k10", "-I1"], HENON, [1, 2],
     dict(minn=10, flength=20, seed=1)),
    ("length_and_exclude", ["-m1,2", "-d1", "-l500", "-x100", "-L20", "-k10", "-I1"], AR_RUN, [1],
     dict(minn=10, flength=20, seed=1)),
    ("verbosity_0", ["-m1,2", "-d1", "-L20", "-k10", "-I1", "-V0"], AR_RUN, [1],
     dict(minn=10, flength=20, seed=1)),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,kwargs", CASES, ids=[c[0] for c in CASES]
)
def test_forecast_matches_cli(label, args, datafile, columns, kwargs):
    exclude = 100 if label == "length_and_exclude" else 0
    length_kw = 500 if label == "length_and_exclude" else None

    cli_text = run_cli(args, datafile)
    dim = len(columns)
    cli_rows = parse_output(cli_text, dim)

    series = load_multi_series(datafile, columns, length=length_kw, exclude=exclude)
    result = tisean.lzo_run.forecast(series, **kwargs)

    assert result.dim == dim
    assert result.length == kwargs["flength"]
    np.testing.assert_allclose(result.series, cli_rows, **CLI_TEXT_TOL)


def test_forecast_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.lzr"
    subprocess.run(
        [LZO_RUN_BIN, "-m1,2", "-d1", "-L20", "-k10", "-I1", "-o", str(outfile), AR_RUN],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_rows = parse_output(outfile.read_text(), 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lzo_run.forecast(series, minn=10, flength=20, seed=1)

    np.testing.assert_allclose(result.series, cli_rows, **CLI_TEXT_TOL)


def test_forecast_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default embed=2 (dim comes from series.shape[0]),
    # -d default 1, -L default 1000, -k default 50, -r/-f default
    # 1.e-3/1.2, -I default 0x9074325 (fixed).
    #
    # -K's own documented default is "no", but the CLI's setsort global
    # actually defaults to 1 and scan_options() never resets it to 0 (see
    # lzo-run.h), so the shipped CLI always behaves as if -K were given -
    # fix_neighbors=True is this binding's default for the same reason, and
    # is what the no-flags CLI run below actually corresponds to.
    series = load_multi_series(AR_RUN, [1])

    default = tisean.lzo_run.forecast(series)
    explicit = tisean.lzo_run.forecast(
        series, embed=2, delay=1, minn=50, fix_neighbors=True, flength=1000,
        eps0=1.0e-3, epsset=False, epsf=1.2, noise_pct=None, seed=0x9074325,
    )

    assert default.dim == explicit.dim
    assert default.length == explicit.length
    np.testing.assert_array_equal(default.series, explicit.series)


def test_forecast_default_kwargs_match_cli_with_no_flags():
    cli_text = run_cli([], AR_RUN)
    cli_rows = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lzo_run.forecast(series)

    assert result.length == 1000
    np.testing.assert_allclose(result.series, cli_rows, **CLI_TEXT_TOL)


@pytest.mark.parametrize("flength", [1, 10])
def test_forecast_matches_cli_with_noise_and_default_seed(flength):
    # rand.c's rnd_init() only actually (re)seeds once per process (see
    # test_ar_model_bindings.py's analogous test), so comparing against a
    # specific seed must run the Python side in its own fresh subprocess,
    # matching what the CLI subprocess gets.
    args = ["-m1,2", "-d1", f"-L{flength}", "-k10", "-I1", "-%5"]
    cli_text = run_cli(args, AR_RUN)
    cli_rows = parse_output(cli_text, 1)
    assert cli_rows.shape == (flength, 1)

    script = (
        "import numpy as np, tisean\n"
        f"series = np.loadtxt({AR_RUN!r}).reshape(1, -1)\n"
        f"result = tisean.lzo_run.forecast(series, minn=10, flength={flength}, "
        "seed=1, noise_pct=5.0)\n"
        "print(result.series.tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_rows = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_rows, cli_rows, **CLI_TEXT_TOL)


def test_forecast_noise_pct_none_and_zero_both_disable_noise():
    # scan_options() only enables noise when -% is given AND the resulting
    # value is > 0.0 - "not given" (None) and "given as 0" must therefore
    # produce identical, noise-free output.
    series = load_multi_series(AR_RUN, [1])

    without_flag = tisean.lzo_run.forecast(series, minn=10, flength=10, seed=1)
    with_zero = tisean.lzo_run.forecast(
        series, minn=10, flength=10, seed=1, noise_pct=0.0
    )

    np.testing.assert_array_equal(without_flag.series, with_zero.series)


def test_forecast_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 60) + "\n")

    result = subprocess.run(
        [LZO_RUN_BIN, "-m1,2", "-L5", "-k10", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 5.000000e+00 to 5.000000e+00" in result.stderr

    series = np.full((1, 60), 5.0)
    with pytest.raises(ValueError):
        tisean.lzo_run.forecast(series, minn=10, flength=5)


def test_forecast_rejects_2d_shape_mismatch():
    series = load_multi_series(AR_RUN, [1]).reshape(-1)
    with pytest.raises(ValueError):
        tisean.lzo_run.forecast(series)


def test_forecast_rejects_embed_zero():
    series = load_multi_series(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.lzo_run.forecast(series, embed=0)


def test_forecast_rejects_series_too_short_for_embed_delay():
    series = load_multi_series(AR_RUN, [1], length=3)
    with pytest.raises(ValueError):
        tisean.lzo_run.forecast(series, embed=5, delay=1, minn=1, flength=1)
