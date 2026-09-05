import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# figures), so comparisons against numbers parsed from its stdout/output
# files can never be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

GHKSS_TOO_MANY_NEIGHBORS = 56
RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
GHKSS_BIN = os.path.abspath("./bin/ghkss")


def run_cli(args, datafile, iterations=1):
    """ghkss always writes '<datafile>.opt.<iter>' for every iteration as a
    side effect, in addition to echoing the last iteration to stdout when
    -o is omitted (see tests/test_ghkss.py). Clean those artifacts up so
    the test doesn't litter tests/refs/."""
    artifacts = [f"{datafile}.opt.{i}" for i in range(1, iterations + 1)]
    try:
        result = subprocess.run(
            [GHKSS_BIN] + args + [datafile],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    finally:
        for path in artifacts:
            if os.path.exists(path):
                os.remove(path)


def parse_series_text(text, comp):
    """Rows of exactly `comp` floats (the CLI's "%e "*comp per line format);
    non-matching lines (banners, blank lines) are skipped."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != comp:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return np.array(rows)


def load_multi_series(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (comp, length) array. Unlike ar-model,
    ghkss does not center the data before use."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


# label, cli_args, datafile, columns, length, exclude, kwargs (only
# overrides from tisean.ghkss.reduce's defaults need to be listed)
CASES = [
    ("embed_3", ["-m1,3", "-k20", "-l300"], AR_RUN, [1], 300, 0,
     dict(embed=3, minn=20)),
    ("comp2_embed2", ["-m2,2", "-k15", "-l400"], HENON, [1, 2], 400, 0,
     dict(embed=2, minn=15)),
    ("delay_2", ["-m1,3", "-d2", "-k20", "-l300"], AR_RUN, [1], 300, 0,
     dict(embed=3, delay=2, minn=20)),
    ("qdim_1", ["-m1,3", "-q1", "-k20", "-l300"], AR_RUN, [1], 300, 0,
     dict(embed=3, qdim=1, minn=20)),
    ("minn_10", ["-m1,3", "-k10", "-l300"], AR_RUN, [1], 300, 0,
     dict(embed=3, minn=10)),
    ("mineps_explicit", ["-m1,3", "-k20", "-r0.01", "-l300"], AR_RUN, [1], 300, 0,
     dict(embed=3, minn=20, mineps=0.01)),
    ("euclidean", ["-m1,3", "-k20", "-2", "-l300"], AR_RUN, [1], 300, 0,
     dict(embed=3, minn=20, euclidean=True)),
    ("verbosity_0", ["-m1,3", "-k20", "-l300", "-V0"], AR_RUN, [1], 300, 0,
     dict(embed=3, minn=20)),
    ("length_and_exclude", ["-m1,3", "-k20", "-l300", "-x50"], AR_RUN, [1], 300, 50,
     dict(embed=3, minn=20)),
    ("column_subset_reordered", ["-m2,2", "-k15", "-l400", "-c2,1"], HENON, [2, 1], 400, 0,
     dict(embed=2, minn=15)),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_reduce_matches_cli_across_options(label, args, datafile, columns, length, exclude, kwargs):
    comp = len(columns)
    out = run_cli(args, datafile)
    cli_series = parse_series_text(out, comp)

    series = load_multi_series(datafile, columns, length=length, exclude=exclude)
    result = tisean.ghkss.reduce(series, **kwargs)

    assert result.comp == comp
    assert result.n_iterations == 1
    py_series = result.series[-1].T  # (length, comp)

    assert py_series.shape == cli_series.shape
    np.testing.assert_allclose(py_series, cli_series, **CLI_TEXT_TOL)


def test_reduce_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default 1,5; -d default 1; -q default 2; -k
    # default 50; -r default (interval of data)/1000 (i.e. mineps=None);
    # -i default 1; -2 default off (euclidean=False).
    series = load_multi_series(AR_RUN, [1])

    default = tisean.ghkss.reduce(series)
    explicit = tisean.ghkss.reduce(
        series, embed=5, delay=1, qdim=2, minn=50, mineps=None, iterations=1,
        euclidean=False,
    )

    assert default.comp == explicit.comp == 1
    assert default.n_iterations == explicit.n_iterations == 1
    np.testing.assert_array_equal(default.series, explicit.series)
    np.testing.assert_array_equal(default.shift, explicit.shift)
    np.testing.assert_array_equal(default.rms, explicit.rms)


def test_reduce_matches_cli_across_multiple_iterations():
    args = ["-m1,3", "-k20", "-l300", "-i2"]
    datafile = AR_RUN
    artifacts = [f"{datafile}.opt.1", f"{datafile}.opt.2"]
    try:
        subprocess.run(
            [GHKSS_BIN] + args + [datafile],
            capture_output=True,
            text=True,
            check=True,
        )
        cli_iter1 = np.loadtxt(artifacts[0]).reshape(-1, 1)
        cli_iter2 = np.loadtxt(artifacts[1]).reshape(-1, 1)
    finally:
        for path in artifacts:
            if os.path.exists(path):
                os.remove(path)

    series = load_multi_series(datafile, [1], length=300)
    result = tisean.ghkss.reduce(series, embed=3, minn=20, iterations=2)

    assert result.n_iterations == 2
    np.testing.assert_allclose(result.series[0, 0, :], cli_iter1[:, 0], **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.series[1, 0, :], cli_iter2[:, 0], **CLI_TEXT_TOL)


def test_reduce_rejects_length_below_minn_like_cli():
    result = subprocess.run(
        [GHKSS_BIN, "-m1,3", "-k20", "-l10", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == GHKSS_TOO_MANY_NEIGHBORS
    assert "you will never find" in result.stderr

    series = load_multi_series(AR_RUN, [1], length=10)
    with pytest.raises(ValueError):
        tisean.ghkss.reduce(series, embed=3, minn=20)


def test_reduce_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["2.5"] * 100) + "\n")

    result = subprocess.run(
        [GHKSS_BIN, "-m1,3", "-k20", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 2.500000e+00 to 2.500000e+00" in result.stderr

    series = np.full((1, 100), 2.5)
    with pytest.raises(ValueError):
        tisean.ghkss.reduce(series, embed=3, minn=20)


def test_reduce_rejects_1d_series():
    series = load_multi_series(AR_RUN, [1]).reshape(-1)
    with pytest.raises(ValueError):
        tisean.ghkss.reduce(series)
