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
VARIANCE_VAR_EQ_ZERO = 23
NSTAT_Z_TOO_MANY_PIECES = 61

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
NSTAT_Z_BIN = os.path.abspath("./bin/nstat_z")


def run_cli(args, datafile, tmp_path=None, name="out.nsz"):
    if tmp_path is not None:
        outfile = tmp_path / name
        subprocess.run(
            [NSTAT_Z_BIN] + args + ["-o", str(outfile), datafile],
            capture_output=True,
            text=True,
            check=True,
        )
        return outfile.read_text()
    result = subprocess.run(
        [NSTAT_Z_BIN] + args + [datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_output(text):
    """Rows are "<first> <second> <value>" (1-indexed first/second),
    separated by a blank line after each group of same `first`. Returns a
    (n_pairs, 3) array; headers/blank lines are skipped by keeping only
    3-token numeric rows."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return np.array(rows).reshape(-1, 3)


def load_series(path, column=1, length=None, exclude=0):
    """Replicates get_series()'s -x/-l/-c handling: skip `exclude` lines,
    keep up to `length` of the rest, then pick 1-indexed `column`."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


# label, cli_args, datafile, column, length, exclude, kwargs (only overrides
# from tisean.nstat_z.compute's defaults need to be listed; "pieces" is
# always required)
CASES = [
    ("defaults", ["-#4", "-m2", "-n20", "-k10"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20)),
    ("dim_3", ["-#4", "-m3", "-n20", "-k10"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=3, minn=10, center=20)),
    ("delay_2", ["-#4", "-m2", "-d2", "-n20", "-k10"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, delay=2, minn=10, center=20)),
    ("minn_5", ["-#4", "-m2", "-n20", "-k5"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=5, center=20)),
    ("step_2", ["-#4", "-m2", "-n15", "-k10", "-s2"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=15, step=2)),
    ("causal_3", ["-#4", "-m2", "-n20", "-k10", "-C3"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, causal=3)),
    ("eps0_raw", ["-#4", "-m2", "-n20", "-k10", "-r0.01"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, eps0=0.01, epsset=True)),
    ("epsf_1_5", ["-#4", "-m2", "-n20", "-k10", "-f1.5"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, epsf=1.5)),
    ("pieces_6", ["-#6", "-m2", "-n15", "-k8"], AR_RUN, 1, None, 0,
     dict(pieces=6, dim=2, minn=8, center=15)),
    ("length_and_exclude", ["-#4", "-m2", "-n15", "-k8", "-l500", "-x100"], AR_RUN, 1, 500, 100,
     dict(pieces=4, dim=2, minn=8, center=15)),
    ("column_2", ["-#4", "-m2", "-n20", "-k10", "-c2"], HENON, 2, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20)),
    ("verbosity_0", ["-#4", "-m2", "-n20", "-k10", "-V0"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20)),
    ("first_window_range", ["-#4", "-m2", "-n20", "-k10", "-1", "1-2"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, first_window=[1, 1, 0, 0])),
    ("second_window_list", ["-#4", "-m2", "-n20", "-k10", "-2", "2,4"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, second_window=[0, 1, 0, 1])),
    ("first_offset_1", ["-#4", "-m2", "-n20", "-k10", "-1", "+1"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, first_offset=1)),
    ("second_offset_0", ["-#4", "-m2", "-n20", "-k10", "-2", "+0"], AR_RUN, 1, None, 0,
     dict(pieces=4, dim=2, minn=10, center=20, second_offset=0)),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(label, args, datafile, column, length, exclude, kwargs):
    cli_text = run_cli(args, datafile)
    cli_rows = parse_output(cli_text)

    series = load_series(datafile, column=column, length=length, exclude=exclude)
    result = tisean.nstat_z.compute(series, **kwargs)

    assert result.n_pairs == len(cli_rows)
    py_rows = np.column_stack(
        [result.first + 1, result.second + 1, result.value]
    )
    # Both are already emitted/computed in (first, second) ascending order.
    np.testing.assert_allclose(py_rows[:, :2], cli_rows[:, :2])
    np.testing.assert_allclose(py_rows[:, 2], cli_rows[:, 2], **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    args = ["-#4", "-m2", "-n20", "-k10"]
    cli_text = run_cli(args, AR_RUN, tmp_path=tmp_path)
    cli_rows = parse_output(cli_text)

    series = load_series(AR_RUN)
    result = tisean.nstat_z.compute(series, pieces=4, dim=2, minn=10, center=20)

    py_rows = np.column_stack([result.first + 1, result.second + 1, result.value])
    np.testing.assert_allclose(py_rows[:, :2], cli_rows[:, :2])
    np.testing.assert_allclose(py_rows[:, 2], cli_rows[:, 2], **CLI_TEXT_TOL)


def test_compute_causal_defaults_to_step():
    args = ["-#4", "-m2", "-n20", "-k10", "-s2"]
    cli_text = run_cli(args, AR_RUN)
    cli_rows = parse_output(cli_text)

    series = load_series(AR_RUN)
    result = tisean.nstat_z.compute(series, pieces=4, dim=2, minn=10, center=20, step=2)

    py_rows = np.column_stack([result.first + 1, result.second + 1, result.value])
    np.testing.assert_allclose(py_rows[:, :2], cli_rows[:, :2])
    np.testing.assert_allclose(py_rows[:, 2], cli_rows[:, 2], **CLI_TEXT_TOL)


def test_compute_center_defaults_to_whole_piece():
    # No -n given: the CLI's "default: all" - every point of a piece used
    # as a reference point (clamped to fit).
    args = ["-#2", "-m1", "-k5", "-l200"]
    cli_text = run_cli(args, AR_RUN)
    cli_rows = parse_output(cli_text)

    series = load_series(AR_RUN, length=200)
    result = tisean.nstat_z.compute(series, pieces=2, dim=1, minn=5)

    py_rows = np.column_stack([result.first + 1, result.second + 1, result.value])
    assert py_rows.shape == cli_rows.shape
    np.testing.assert_allclose(py_rows[:, :2], cli_rows[:, :2])
    np.testing.assert_allclose(py_rows[:, 2], cli_rows[:, 2], **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default 3, -d default 1, -n default "all", -k
    # default 30, -s default 1, -C default "steps", -r/-f default
    # 1.e-3/1.2, -1/-2 default "all pieces" (no offset).
    series = load_series(AR_RUN, length=300)

    default = tisean.nstat_z.compute(series, pieces=3)
    explicit = tisean.nstat_z.compute(
        series, pieces=3, dim=3, delay=1, minn=30, step=1, causal=None,
        center=None, first_window=None, second_window=None, first_offset=None,
        second_offset=None, eps0=1.0e-3, epsset=False, epsf=1.2,
    )

    assert default.pieces == explicit.pieces
    assert default.n_pairs == explicit.n_pairs
    np.testing.assert_array_equal(default.first, explicit.first)
    np.testing.assert_array_equal(default.second, explicit.second)
    np.testing.assert_array_equal(default.value, explicit.value)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 60) + "\n")

    result = subprocess.run(
        [NSTAT_Z_BIN, "-#4", "-m2", "-n5", "-k5", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 5.000000e+00 to 5.000000e+00" in result.stderr

    series = np.full(60, 5.0)
    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=4, dim=2, minn=5, center=5)


def test_compute_rejects_zero_variance_piece_like_cli(tmp_path):
    rng = np.random.default_rng(0)
    part1 = rng.normal(size=30)
    part2 = np.full(30, 3.0)
    series = np.concatenate([part1, part2])
    datafile = tmp_path / "zerovar.txt"
    np.savetxt(datafile, series)

    result = subprocess.run(
        [NSTAT_Z_BIN, "-#2", "-m1", "-k5", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO
    assert "Variance of the data is zero" in result.stderr

    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=2, dim=1, minn=5)


def test_compute_rejects_too_many_pieces_like_cli():
    result = subprocess.run(
        [NSTAT_Z_BIN, "-#100", "-m2", "-n5", "-k10", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == NSTAT_Z_TOO_MANY_PIECES
    assert "too many pieces" in result.stderr

    series = load_series(AR_RUN)
    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=100, dim=2, minn=10, center=5)


def test_compute_rejects_2d_series():
    series = load_series(AR_RUN).reshape(-1, 1)
    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=4)


def test_compute_rejects_pieces_zero():
    series = load_series(AR_RUN)
    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=0)


def test_compute_rejects_dim_zero():
    series = load_series(AR_RUN)
    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=4, dim=0)


def test_compute_rejects_wrong_length_window():
    series = load_series(AR_RUN)
    with pytest.raises(ValueError):
        tisean.nstat_z.compute(series, pieces=4, first_window=[1, 1, 0])
