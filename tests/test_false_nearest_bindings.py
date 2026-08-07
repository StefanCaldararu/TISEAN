import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11
VARIANCE_VAR_EQ_ZERO = 23
FALSE_NEAREST_NOT_ENOUGH_POINTS = 54

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
FALSE_NEAREST_BIN = os.path.abspath("./bin/false_nearest")


def run_cli(args, datafile, tmp_path=None, name="out.fnn"):
    if tmp_path is not None:
        outfile = tmp_path / name
        subprocess.run(
            [FALSE_NEAREST_BIN] + args + ["-o", str(outfile), datafile],
            capture_output=True,
            text=True,
            check=True,
        )
        return outfile.read_text()
    result = subprocess.run(
        [FALSE_NEAREST_BIN] + args + [datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_output(text):
    """false_nearest's output is '<dim> <fraction> <avg_eps> <sigma_eps>'
    rows, one per embedding dimension - no header/comment lines."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            rows.append([float(p) for p in parts])
    return np.array(rows).reshape(-1, 4)


def load_multi(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (comp, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


def py_rows(result):
    rows = [
        [result.dimension[i], result.fraction[i], result.avg_eps[i], result.sigma_eps[i]]
        for i in range(result.n)
    ]
    return np.array(rows).reshape(-1, 4)


# label, cli_args, datafile, columns, length, exclude, kwargs (minemb/maxemb/
# delay/theiler/rt/eps0 - only overrides from tisean.false_nearest.compute's
# defaults need to be listed)
CASES = [
    ("defaults", ["-l500"], AR_RUN, [1], 500, 0, {}),
    ("minemb_2", ["-m2", "-M1,5", "-l500"], AR_RUN, [1], 500, 0, dict(minemb=2)),
    ("maxemb_3", ["-M1,3", "-l500"], AR_RUN, [1], 500, 0, dict(maxemb=3)),
    ("delay_2_multivariate", ["-M2,3", "-d2", "-l500"], HENON, [1, 2], 500, 0,
     dict(maxemb=3, delay=2)),
    ("theiler_10", ["-M1,5", "-t10", "-l500"], AR_RUN, [1], 500, 0, dict(theiler=10)),
    ("escape_factor_3_5", ["-M1,5", "-f3.5", "-l500"], AR_RUN, [1], 500, 0, dict(rt=3.5)),
    ("exclude_50", ["-M1,5", "-x50", "-l300"], AR_RUN, [1], 300, 50, {}),
    ("column_2", ["-M1,5", "-c2", "-l500"], HENON, [2], 500, 0, {}),
    ("verbosity_0", ["-M1,5", "-V0", "-l500"], AR_RUN, [1], 500, 0, {}),
    ("multivariate_comp2_maxemb3", ["-m1", "-M2,3", "-l500"], HENON, [1, 2], 500, 0,
     dict(maxemb=3)),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, columns, length, exclude, kwargs):
    cli_text = run_cli(args, datafile)
    cli_result = parse_output(cli_text)

    series = load_multi(datafile, columns, length=length, exclude=exclude)
    result = tisean.false_nearest.compute(series, **kwargs)

    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_array_equal(got[:, 0], cli_result[:, 0])
    np.testing.assert_allclose(got[:, 1:], cli_result[:, 1:], **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    cli_text = run_cli(["-M1,5", "-t5", "-l500"], AR_RUN, tmp_path=tmp_path, name="custom.fnn")
    cli_result = parse_output(cli_text)

    series = load_multi(AR_RUN, [1], length=500)
    result = tisean.false_nearest.compute(series, theiler=5)

    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 1:], cli_result[:, 1:], **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default 1, -M default "1,5" (comp=1, maxemb=5),
    # -d default 1, -f default 2.00, -t default 0.
    series = load_multi(AR_RUN, [1], length=500)

    default = tisean.false_nearest.compute(series)
    explicit = tisean.false_nearest.compute(
        series, minemb=1, maxemb=5, delay=1, theiler=0, rt=2.0, eps0=1.0e-5,
    )

    assert default.n == explicit.n
    np.testing.assert_array_equal(default.dimension, explicit.dimension)
    np.testing.assert_array_equal(default.fraction, explicit.fraction)
    np.testing.assert_array_equal(default.avg_eps, explicit.avg_eps)
    np.testing.assert_array_equal(default.sigma_eps, explicit.sigma_eps)


def test_compute_minemb_greater_than_maxemb_returns_empty_like_cli():
    # The CLI's emb loop just doesn't execute when minemb > maxemb - no
    # error, an empty (zero-row) result.
    result = subprocess.run(
        [FALSE_NEAREST_BIN, "-m10", "-M1,5", "-l500", AR_RUN],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""

    series = load_multi(AR_RUN, [1], length=500)
    py_result = tisean.false_nearest.compute(series, minemb=10, maxemb=5)
    assert py_result.n == 0


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 50) + "\n")

    result = subprocess.run(
        [FALSE_NEAREST_BIN, "-M1,5", "-l50", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 5.000000e+00 to 5.000000e+00" in result.stderr

    series = np.full((1, 50), 5.0)
    with pytest.raises(ValueError):
        tisean.false_nearest.compute(series)


def test_compute_rejects_not_enough_points_like_cli():
    # A huge escape factor makes 2*varianz/rt smaller than the starting
    # epsilon, so the search loop never runs and no neighbor is ever found -
    # deterministic without needing degenerate data.
    result = subprocess.run(
        [FALSE_NEAREST_BIN, "-M1,3", "-f1e10", "-l500", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == FALSE_NEAREST_NOT_ENOUGH_POINTS
    assert "Not enough points found!" in result.stderr

    series = load_multi(AR_RUN, [1], length=500)
    with pytest.raises(ValueError):
        tisean.false_nearest.compute(series, maxemb=3, rt=1e10)


def test_compute_rejects_2d_shape_mismatch():
    series = load_multi(AR_RUN, [1], length=500).reshape(-1)
    with pytest.raises(ValueError):
        tisean.false_nearest.compute(series)


def test_compute_rejects_minemb_zero():
    series = load_multi(AR_RUN, [1], length=500)
    with pytest.raises(ValueError):
        tisean.false_nearest.compute(series, minemb=0)


def test_compute_rejects_series_too_short_for_maxemb_delay():
    # The box-building step reads series[c][i] for i up to
    # length-(maxemb+1)*delay-1, so length must be > (maxemb+1)*delay; with
    # the defaults (maxemb=5, delay=1) that means length must be > 6.
    rng = np.random.default_rng(0)
    too_short = rng.normal(size=(1, 5))
    with pytest.raises(ValueError):
        tisean.false_nearest.compute(too_short, maxemb=5, delay=1)
