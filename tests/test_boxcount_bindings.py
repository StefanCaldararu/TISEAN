import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# figures), so comparisons against numbers parsed from its stdout/output
# file can never be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
BOXCOUNT_BIN = os.path.abspath("./bin/boxcount")


def run_cli(args, outfile):
    subprocess.run(
        [BOXCOUNT_BIN] + args + ["-o", str(outfile)],
        capture_output=True,
        text=True,
        check=True,
    )
    return outfile.read_text()


def parse_output(text):
    """Returns {(component, embedding): rows}, 1-indexed to match the CLI's
    own header fields, where rows is a (epscount, 3) array of
    [eps, entropy, entropy_diff]."""
    blocks = {}
    comp = emb = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#component"):
            if comp is not None:
                blocks[(comp, emb)] = np.array(rows).reshape(-1, 3)
            tokens = line.replace("=", " ").split()
            comp, emb = int(tokens[1]), int(tokens[3])
            rows = []
        elif line:
            rows.append([float(x) for x in line.split()])
    if comp is not None:
        blocks[(comp, emb)] = np.array(rows).reshape(-1, 3)
    return blocks


def load_series(path, dimension, columns=None, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed;
    defaults to 1..dimension, matching the CLI's own default when -c is not
    given). Returns a (dimension, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    if columns is None:
        columns = list(range(1, dimension + 1))
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


def assert_matches_cli(bc, cli_blocks, dimension, maxembed):
    for slot in range(dimension * maxembed):
        component = int(bc.which_component[slot])
        embed = int(bc.which_embed[slot])
        cli_rows = cli_blocks[(component + 1, embed + 1)]

        np.testing.assert_allclose(bc.eps, cli_rows[:, 0], **CLI_TEXT_TOL)
        np.testing.assert_allclose(
            bc.entropy[:, slot], cli_rows[:, 1], **CLI_TEXT_TOL
        )

        if slot == 0:
            diff = bc.entropy[:, slot]
        else:
            diff = bc.entropy[:, slot] - bc.entropy[:, slot - 1]
        np.testing.assert_allclose(diff, cli_rows[:, 2], **CLI_TEXT_TOL)


CASES = [
    # label, args, datafile, dimension, columns, kwargs, length, exclude
    ("defaults", [], AR_RUN, 1, None, {}, None, 0),
    ("maxembed_5", ["-M1,5"], AR_RUN, 1, None, dict(maxembed=5), None, 0),
    ("dimension_2", ["-M2,4"], HENON, 2, None, dict(maxembed=4), None, 0),
    ("columns_reordered", ["-c2,1"], HENON, 2, [2, 1], {}, None, 0),
    ("delay_2", ["-M1,4", "-d2"], AR_RUN, 1, None, dict(maxembed=4, delay=2), None, 0),
    ("Q_shannon", ["-Q1.0"], AR_RUN, 1, None, dict(q=1.0), None, 0),
    ("Q_1_5", ["-Q1.5"], AR_RUN, 1, None, dict(q=1.5), None, 0),
    ("epscount_10", ["-#10"], AR_RUN, 1, None, dict(epscount=10), None, 0),
    ("epscount_0", ["-#0"], AR_RUN, 1, None, dict(epscount=0), None, 0),
    (
        "epsmin_absolute",
        ["-r0.05"],
        AR_RUN,
        1,
        None,
        dict(epsmin=0.05, epsmin_absolute=True),
        None,
        0,
    ),
    (
        "epsmax_absolute",
        ["-R0.5"],
        AR_RUN,
        1,
        None,
        dict(epsmax=0.5, epsmax_absolute=True),
        None,
        0,
    ),
    ("length", [], AR_RUN, 1, None, {}, 300, 0),
    ("exclude", [], AR_RUN, 1, None, {}, None, 50),
    ("length_and_exclude", [], AR_RUN, 1, None, {}, 300, 50),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, {}, None, 0),
]


@pytest.mark.parametrize(
    "label,args,datafile,dimension,columns,kwargs,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(
    label, args, datafile, dimension, columns, kwargs, length, exclude, tmp_path
):
    outfile = tmp_path / "out.box"
    cli_args = list(args) + [datafile]
    if length is not None:
        cli_args = [f"-l{length}"] + cli_args
    if exclude:
        cli_args = [f"-x{exclude}"] + cli_args
    out = run_cli(cli_args, outfile)
    cli_blocks = parse_output(out)

    series = load_series(
        datafile, dimension, columns=columns, length=length, exclude=exclude
    )
    bc = tisean.boxcount.compute(series, **kwargs)

    maxembed = kwargs.get("maxembed", 10)
    assert bc.dimension == dimension
    assert bc.maxembed == maxembed
    assert bc.epscount == kwargs.get("epscount", 20)
    assert_matches_cli(bc, cli_blocks, dimension, maxembed)


def test_compute_default_kwargs_match_cli():
    series = load_series(AR_RUN, 1)

    default = tisean.boxcount.compute(series)
    explicit = tisean.boxcount.compute(
        series,
        maxembed=10,
        delay=1,
        q=2.0,
        epsmin=1.0e-3,
        epsmin_absolute=False,
        epsmax=1.0,
        epsmax_absolute=False,
        epscount=20,
    )

    assert default.maxembed == explicit.maxembed == 10
    assert default.epscount == explicit.epscount == 20
    np.testing.assert_array_equal(default.eps, explicit.eps)
    np.testing.assert_array_equal(default.entropy, explicit.entropy)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")
    outfile = tmp_path / "out.box"

    result = subprocess.run(
        [BOXCOUNT_BIN, str(datafile), "-o", str(outfile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series = np.full((1, 20), 1.5)
    with pytest.raises(ValueError):
        tisean.boxcount.compute(series)


def test_compute_rejects_series_too_short_for_maxembed_delay():
    # Not a CLI-checked path (the original code would silently misbehave via
    # unsigned-integer underflow rather than erroring cleanly), but the
    # Python binding validates it explicitly to avoid a crash - see
    # boxcount_compute_binding in python/src/bindings.cpp.
    series = np.random.default_rng(0).normal(size=(1, 5))
    with pytest.raises(ValueError):
        tisean.boxcount.compute(series, maxembed=10, delay=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(maxembed=0),
        dict(delay=0),
    ],
)
def test_compute_rejects_invalid_scalar_options(kwargs):
    series = load_series(AR_RUN, 1)
    with pytest.raises(ValueError):
        tisean.boxcount.compute(series, **kwargs)


def test_compute_rejects_non_2d_series():
    with pytest.raises(ValueError):
        tisean.boxcount.compute(np.zeros(20))
