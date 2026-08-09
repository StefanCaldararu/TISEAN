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
LORENZ = "tests/refs/lorenz_l1000.txt"
LFO_RUN_BIN = os.path.abspath("./bin/lfo-run")


def run_cli(args, datafile):
    result = subprocess.run(
        [LFO_RUN_BIN] + args + [datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_output(text, dim):
    """lfo-run's stdout is pure data: one row of `dim` space-separated %e
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
# tisean.lfo_run.forecast's defaults need to be listed)
CASES = [
    ("defaults", ["-m1,2", "-d1", "-L50", "-k10"], AR_RUN, [1],
     dict(minn=10, flength=50)),
    ("embed_3", ["-m1,3", "-d1", "-L50", "-k10"], AR_RUN, [1],
     dict(embed=3, minn=10, flength=50)),
    ("delay_2", ["-m1,2", "-d2", "-L50", "-k10"], AR_RUN, [1],
     dict(delay=2, minn=10, flength=50)),
    ("minn_5", ["-m1,2", "-d1", "-L50", "-k5"], AR_RUN, [1],
     dict(minn=5, flength=50)),
    ("flength_20", ["-m1,2", "-d1", "-L20", "-k10"], AR_RUN, [1],
     dict(minn=10, flength=20)),
    ("zeroth_order_dash_0", ["-m1,2", "-d1", "-L20", "-k10", "-0"], AR_RUN, [1],
     dict(minn=10, flength=20, zeroth_order=True)),
    ("eps0_raw", ["-m1,2", "-d1", "-L20", "-k10", "-r0.01"], AR_RUN, [1],
     dict(minn=10, flength=20, eps0=0.01, epsset=True)),
    ("epsf_1_5", ["-m1,2", "-d1", "-L20", "-k10", "-f1.5"], AR_RUN, [1],
     dict(minn=10, flength=20, epsf=1.5)),
    # Not HENON: its 2D points lie on a thin fractal attractor, so local
    # neighborhoods for a 2-component/embed-2 local-linear fit are prone to
    # near-collinear design matrices. On x86_64 this deterministically
    # tips into an exactly-singular matrix (lfo-run exits 19,
    # SOLVELE_SINGULAR_MATRIX) while ARM64's differing FP codegen doesn't
    # hit it - passed locally, failed every time in CI (same root cause as
    # lfo-ar's "dim2_henon" case). Verified: LORENZ's first two columns
    # don't trigger it, confirmed with 10/10 passing runs under x86_64.
    ("multivariate_lorenz_2col", ["-m2,2", "-d1", "-c1,2", "-L20", "-k10"], LORENZ, [1, 2],
     dict(minn=10, flength=20)),
    ("length_and_exclude", ["-m1,2", "-d1", "-l500", "-x100", "-L20", "-k10"], AR_RUN, [1],
     dict(minn=10, flength=20)),
    ("verbosity_0", ["-m1,2", "-d1", "-L20", "-k10", "-V0"], AR_RUN, [1],
     dict(minn=10, flength=20)),
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
    result = tisean.lfo_run.forecast(series, **kwargs)

    assert result.dim == dim
    assert result.length == kwargs["flength"]
    np.testing.assert_allclose(result.series, cli_rows, **CLI_TEXT_TOL)


def test_forecast_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.cast"
    subprocess.run(
        [LFO_RUN_BIN, "-m1,2", "-d1", "-L20", "-k10", "-o", str(outfile), AR_RUN],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_rows = parse_output(outfile.read_text(), 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lfo_run.forecast(series, minn=10, flength=20)

    np.testing.assert_allclose(result.series, cli_rows, **CLI_TEXT_TOL)


def test_forecast_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default embed=2 (dim comes from series.shape[0]),
    # -d default 1, -L default 1000, -k default 30, -r/-f default
    # 1.e-3/1.2, -0 default not set (zeroth_order=False).
    series = load_multi_series(AR_RUN, [1])

    default = tisean.lfo_run.forecast(series)
    explicit = tisean.lfo_run.forecast(
        series, embed=2, delay=1, minn=30, zeroth_order=False, flength=1000,
        eps0=1.0e-3, epsset=False, epsf=1.2,
    )

    assert default.dim == explicit.dim
    assert default.length == explicit.length
    np.testing.assert_array_equal(default.series, explicit.series)


def test_forecast_default_kwargs_match_cli_with_no_flags():
    cli_text = run_cli([], AR_RUN)
    cli_rows = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lfo_run.forecast(series)

    assert result.length == 1000
    np.testing.assert_allclose(result.series, cli_rows, **CLI_TEXT_TOL)


def test_forecast_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 60) + "\n")

    result = subprocess.run(
        [LFO_RUN_BIN, "-m1,2", "-L5", "-k10", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 5.000000e+00 to 5.000000e+00" in result.stderr

    series = np.full((1, 60), 5.0)
    with pytest.raises(ValueError):
        tisean.lfo_run.forecast(series, minn=10, flength=5)


def test_forecast_rejects_2d_shape_mismatch():
    series = load_multi_series(AR_RUN, [1]).reshape(-1)
    with pytest.raises(ValueError):
        tisean.lfo_run.forecast(series)


def test_forecast_rejects_embed_zero():
    series = load_multi_series(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.lfo_run.forecast(series, embed=0)


def test_forecast_rejects_series_too_short_for_embed_delay():
    series = load_multi_series(AR_RUN, [1], length=3)
    with pytest.raises(ValueError):
        tisean.lfo_run.forecast(series, embed=5, delay=1, minn=1, flength=1)
