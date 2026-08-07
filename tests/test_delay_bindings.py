import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

DELAY_WRONG_FORMAT_F = 73
DELAY_DIM_NOT_EQUAL_F_M = 74
DELAY_DIM_NOT_EQUAL_F_m = 75
DELAY_WRONG_FORMAT_D = 76
DELAY_WRONG_NUM_D = 77
DELAY_INCONS_d_D = 78
DELAY_SMALL_ZERO = 79
DELAY_INCONS_m_M = 80

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
DELAY_BIN = os.path.abspath("./bin/delay")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [DELAY_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
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


CASES = [
    # label, cli_args, datafile, columns, length, exclude, embed_kwargs
    ("defaults_m3d1", ["-m3", "-d1"], AR_RUN, [1], 200, 0, dict(embdim=3, delay=1)),
    ("m2_default_delay", ["-m2"], AR_RUN, [1], 300, 0, dict(embdim=2)),
    ("m4_d3", ["-m4", "-d3"], AR_RUN, [1], 300, 0, dict(embdim=4, delay=3)),
    (
        "explicit_F_single_column",
        ["-F3"],
        AR_RUN,
        [1],
        200,
        0,
        dict(format=np.array([3], dtype=np.uint32)),
    ),
    (
        "two_columns_M2_m4",
        ["-M2", "-m4"],
        HENON,
        [1, 2],
        200,
        0,
        dict(embdim=4),
    ),
    (
        "two_columns_explicit_F",
        ["-M2", "-F2,1"],
        HENON,
        [1, 2],
        200,
        0,
        dict(format=np.array([2, 1], dtype=np.uint32)),
    ),
    (
        "two_columns_c_reordered",
        ["-M2", "-c2,1", "-m4"],
        HENON,
        [2, 1],
        200,
        0,
        dict(embdim=4),
    ),
    (
        "multidelay_single_column",
        ["-m3", "-D2,5"],
        AR_RUN,
        [1],
        200,
        0,
        dict(embdim=3, multidelay=np.array([2, 5], dtype=np.uint32)),
    ),
    (
        "multidelay_two_columns",
        ["-M2", "-F2,2", "-D3,1"],
        HENON,
        [1, 2],
        200,
        0,
        dict(
            format=np.array([2, 2], dtype=np.uint32),
            multidelay=np.array([3, 1], dtype=np.uint32),
        ),
    ),
    ("length_and_exclude", ["-m3", "-l200", "-x50"], AR_RUN, [1], 200, 50, dict(embdim=3)),
    ("verbosity_0", ["-m3", "-V0"], AR_RUN, [1], 200, 0, dict(embdim=3)),
]


@pytest.mark.parametrize(
    "label,cli_args,datafile,columns,length,exclude,embed_kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_embed_matches_cli(label, cli_args, datafile, columns, length, exclude, embed_kwargs):
    col_arg = ["-c" + ",".join(str(c) for c in columns)] if len(columns) > 1 else []
    args = cli_args + col_arg + [f"-l{length}", f"-x{exclude}", datafile]

    out = run_cli(args)
    cli_vectors = parse_output(out)

    series = load_multi_series(datafile, columns, length=length, exclude=exclude)
    result = tisean.delay.embed(series, **embed_kwargs)

    assert result.n_vectors == cli_vectors.shape[0]
    assert result.alldim == cli_vectors.shape[1]
    np.testing.assert_allclose(result.vectors, cli_vectors, **CLI_TEXT_TOL)


def test_embed_default_kwargs_match_cli_defaults():
    # No -m/-d/-F/-D/-M at all: defaults are M=1, m=2, d=1.
    length = 200
    out = run_cli([f"-l{length}", AR_RUN])
    cli_vectors = parse_output(out)

    series = load_multi_series(AR_RUN, [1], length=length)
    result = tisean.delay.embed(series)

    assert result.alldim == 2
    np.testing.assert_allclose(result.vectors, cli_vectors, **CLI_TEXT_TOL)


def test_embed_short_series_yields_empty_result_like_cli():
    # length <= maxdelay: the CLI prints nothing rather than erroring.
    out = run_cli(["-m3", "-d5", "-l10", AR_RUN])
    assert out == ""

    series = load_multi_series(AR_RUN, [1], length=10)
    result = tisean.delay.embed(series, embdim=3, delay=5)

    assert result.n_vectors == 0
    assert result.vectors.shape == (0, 3)


def test_embed_rejects_delay_below_one_like_cli():
    result = subprocess.run(
        [DELAY_BIN, "-d0", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == DELAY_SMALL_ZERO

    series = load_multi_series(AR_RUN, [1], length=200)
    with pytest.raises(ValueError):
        tisean.delay.embed(series, delay=0)


def test_embed_rejects_inconsistent_m_and_M_like_cli():
    result = subprocess.run(
        [DELAY_BIN, "-M2", "-m5", "-c1,2", HENON],
        capture_output=True,
        text=True,
    )
    assert result.returncode == DELAY_INCONS_m_M

    series = load_multi_series(HENON, [1, 2], length=200)
    with pytest.raises(ValueError):
        tisean.delay.embed(series, embdim=5)


def test_embed_rejects_wrong_multidelay_length_like_cli():
    result = subprocess.run(
        [DELAY_BIN, "-m3", "-D2", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == DELAY_WRONG_NUM_D

    series = load_multi_series(AR_RUN, [1], length=200)
    with pytest.raises(ValueError):
        tisean.delay.embed(series, embdim=3, multidelay=np.array([2], dtype=np.uint32))


def test_embed_rejects_format_length_mismatch():
    series = load_multi_series(HENON, [1, 2], length=200)
    with pytest.raises(ValueError):
        tisean.delay.embed(series, format=np.array([2], dtype=np.uint32))
