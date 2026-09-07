import os
import subprocess
import sys

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows. recurr's own pair output is all
# integers (no rounding involved there), but the constant-data error
# message it reproduces is still "%e"-formatted.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
RECURR_BIN = os.path.abspath("./bin/recurr")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [RECURR_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """recurr prints one '<point> <neighbor>' pair per line, 1-based
    indices, no header/comment lines at all."""
    rows = [
        [int(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def load_multi_series(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (dim, length) array - recurr_find()
    does its own rescaling internally, so unlike ar-model's loader this
    does no centering."""
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
    # label, args, datafile, columns, length, exclude, embed, delay, eps, eps_is_raw
    ("defaults", [], AR_RUN, [1], None, 0, 2, 1, 1.e-3, False),
    ("explicit_default_m", ["-m1,2"], AR_RUN, [1], None, 0, 2, 1, 1.e-3, False),
    ("embed_3", ["-m1,3"], AR_RUN, [1], None, 0, 3, 1, 1.e-3, False),
    ("delay_2", ["-r0.1", "-d2"], AR_RUN, [1], None, 0, 2, 2, 0.1, True),
    ("raw_eps", ["-r0.1"], AR_RUN, [1], None, 0, 2, 1, 0.1, True),
    ("length_100", ["-r0.1", "-l100"], AR_RUN, [1], 100, 0, 2, 1, 0.1, True),
    ("exclude_50", ["-r0.1", "-x50", "-l100"], AR_RUN, [1], 100, 50, 2, 1, 0.1, True),
    ("column_2", ["-c2", "-r0.1"], HENON, [2], None, 0, 2, 1, 0.1, True),
    (
        "dim2_henon",
        ["-m2,2", "-r0.1", "-l200"],
        HENON,
        [1, 2],
        200,
        0,
        2,
        1,
        0.1,
        True,
    ),
    (
        "dim2_henon_reordered_columns",
        ["-m2,2", "-c2,1", "-r0.1", "-l200"],
        HENON,
        [2, 1],
        200,
        0,
        2,
        1,
        0.1,
        True,
    ),
    (
        "dim3_lorenz",
        ["-m3,2", "-r0.1", "-l200"],
        LORENZ,
        [1, 2, 3],
        200,
        0,
        2,
        1,
        0.1,
        True,
    ),
    ("verbosity_0", ["-V0", "-r0.1", "-l100"], AR_RUN, [1], 100, 0, 2, 1, 0.1, True),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,embed,delay,eps,eps_is_raw",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_find_matches_cli(
    label, args, datafile, columns, length, exclude, embed, delay, eps, eps_is_raw
):
    out = run_cli(args + [datafile])
    cli_pairs = parse_output(out)

    series = load_multi_series(datafile, columns, length=length, exclude=exclude)
    result = tisean.recurr.find(
        series, embed=embed, delay=delay, eps=eps, eps_is_raw=eps_is_raw
    )

    assert result.count == len(cli_pairs)
    py_pairs = np.stack([result.point, result.neighbor], axis=1)
    np.testing.assert_array_equal(py_pairs, cli_pairs)


def test_find_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.rec"
    run_cli(["-m1,2", "-d1", "-r0.1", "-l200", "-o" + str(outfile), AR_RUN])

    cli_pairs = parse_output(outfile.read_text())

    series = load_multi_series(AR_RUN, [1], length=200)
    result = tisean.recurr.find(series, embed=2, delay=1, eps=0.1, eps_is_raw=True)

    py_pairs = np.stack([result.point, result.neighbor], axis=1)
    np.testing.assert_array_equal(py_pairs, cli_pairs)


def test_find_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default "1,2" (dim,embed), -d default 1, -r
    # default "(data interval)/1000" i.e. eps=1.e-3 used directly in the
    # already-rescaled [0,1) space (epsset stays unset), -% default 100.0
    # i.e. fraction=1.0.
    series = load_multi_series(AR_RUN, [1])

    default = tisean.recurr.find(series)
    explicit = tisean.recurr.find(
        series, embed=2, delay=1, eps=1.e-3, eps_is_raw=False, fraction=1.0
    )

    assert default.count == explicit.count
    np.testing.assert_array_equal(default.point, explicit.point)
    np.testing.assert_array_equal(default.neighbor, explicit.neighbor)


def test_find_rejects_constant_dimension_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [RECURR_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 1.500000e+00 to 1.500000e+00" in result.stderr

    series = np.full((1, 20), 1.5)
    with pytest.raises(ValueError, match=r"1\.500000"):
        tisean.recurr.find(series)


def test_find_matches_cli_with_fraction_below_one_in_fresh_process():
    # rand.c's rnd_init() only actually (re)seeds once per process (see
    # source_c/routines/rand.c's static rnd_init_was_set guard), and
    # recurr_find() always seeds with the CLI's own fixed value
    # (0x9834725L). Any other test in this same pytest process that has
    # already exercised the RNG (directly, or indirectly through another
    # tisean binding) would leave stale state behind, so a fraction < 1.0
    # comparison against a fresh CLI subprocess must run the Python side in
    # its own fresh subprocess too.
    args = ["-m1,2", "-d1", "-r0.1", "-%50", "-l200"]
    cli_pairs = parse_output(run_cli(args + [AR_RUN]))

    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({AR_RUN!r})[:200]\n"
        "series = raw.reshape(1, -1)\n"
        "result = tisean.recurr.find(series, embed=2, delay=1, eps=0.1,\n"
        "                             eps_is_raw=True, fraction=0.5)\n"
        "for p, n in zip(result.point, result.neighbor):\n"
        "    print(p, n)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_pairs = parse_output(result.stdout)

    np.testing.assert_array_equal(py_pairs, cli_pairs)


def test_find_fraction_one_is_deterministic_regardless_of_prior_rng_use():
    # Unlike the fraction < 1.0 case above, fraction=1.0 always keeps every
    # candidate found (rnd69069()/ULONG_MAX <= 1.0 is always true), so the
    # result must be identical no matter what RNG state earlier calls left
    # behind.
    series = load_multi_series(AR_RUN, [1], length=200)

    tisean.recurr.find(series, embed=2, delay=1, eps=0.1, eps_is_raw=True, fraction=0.3)
    result = tisean.recurr.find(
        series, embed=2, delay=1, eps=0.1, eps_is_raw=True, fraction=1.0
    )

    cli_pairs = parse_output(
        run_cli(["-m1,2", "-d1", "-r0.1", "-l200", AR_RUN])
    )
    py_pairs = np.stack([result.point, result.neighbor], axis=1)
    np.testing.assert_array_equal(py_pairs, cli_pairs)
