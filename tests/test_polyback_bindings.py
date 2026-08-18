import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import tisean

VARIANCE_VAR_EQ_ZERO = 23
SOLVELE_SINGULAR_MATRIX = 19

PARFILE = "tests/refs/parameter_m2p3.pol"

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)


def run_cli(extra_args, datafile, parfile=PARFILE):
    # polyback writes intermediate "<parameter file>.N" files next to the
    # parameter file as it eliminates terms, so use a scratch copy of the
    # parameter file to avoid littering tests/refs/ with those artifacts.
    with tempfile.TemporaryDirectory() as tmpdir:
        parfile_copy = os.path.join(tmpdir, os.path.basename(parfile))
        shutil.copyfile(parfile, parfile_copy)

        result = subprocess.run(
            ["./bin/polyback"] + extra_args + ["-p", parfile_copy, datafile],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rows.append([float(p) for p in parts])
    return rows


def load_order(path=PARFILE):
    order = np.loadtxt(path)
    return order.astype(np.uint32)


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


def check_against_cli(rows, order, model, dim):
    n_terms = int(rows[0][0])
    assert model.n_terms == n_terms
    np.testing.assert_allclose(model.error_in, rows[0][1], **CLI_TEXT_TOL)
    np.testing.assert_allclose(model.error_out, rows[0][2], **CLI_TEXT_TOL)

    level_rows = rows[1:]
    assert model.n_levels == len(level_rows)
    for i, row in enumerate(level_rows):
        assert model.level_n_terms[i] == int(row[0])
        np.testing.assert_allclose(model.level_error_in[i], row[1], **CLI_TEXT_TOL)
        np.testing.assert_allclose(model.level_error_out[i], row[2], **CLI_TEXT_TOL)
        removed = order[model.removed_index[i]]
        np.testing.assert_array_equal(removed, [int(x) for x in row[3 : 3 + dim]])


FIT_CASES = [
    # label, cli_extra_args, python_kwargs, datafile, column
    ("defaults", [], {}, "tests/refs/ar-run_l1000.txt", 1),
    ("length", ["-l400"], {"length": 400}, "tests/refs/ar-run_l1000.txt", 1),
    ("exclude", ["-x100"], {"exclude": 100}, "tests/refs/ar-run_l1000.txt", 1),
    ("column", ["-c2"], {"column": 2}, "tests/refs/henon_l1000.txt", 2),
    ("delay", ["-d2"], {"delay": 2}, "tests/refs/ar-run_l1000.txt", 1),
    ("insample", ["-n300"], {"insample": 300}, "tests/refs/ar-run_l1000.txt", 1),
    ("step", ["-s3"], {"step": 3}, "tests/refs/ar-run_l1000.txt", 1),
    ("down_to", ["-#4"], {"down_to": 4}, "tests/refs/ar-run_l1000.txt", 1),
    ("down_to_all", ["-#10"], {"down_to": 10}, "tests/refs/ar-run_l1000.txt", 1),
    ("down_to_clamped", ["-#0"], {"down_to": 0}, "tests/refs/ar-run_l1000.txt", 1),
]


@pytest.mark.parametrize(
    "label,cli_extra_args,python_kwargs,datafile,column",
    FIT_CASES,
    ids=[c[0] for c in FIT_CASES],
)
def test_fit_matches_cli_across_flags(label, cli_extra_args, python_kwargs, datafile, column):
    order = load_order()
    dim = order.shape[1]

    out = run_cli(cli_extra_args, datafile)
    rows = parse_rows(out)

    series = load_series(
        datafile,
        column=column,
        length=python_kwargs.get("length"),
        exclude=python_kwargs.get("exclude", 0),
    )
    fit_kwargs = {
        k: v for k, v in python_kwargs.items() if k not in ("length", "exclude", "column")
    }
    model = tisean.polyback.fit(series, order, **fit_kwargs)

    check_against_cli(rows, order, model, dim)


def test_fit_default_kwargs_match_documented_cli_defaults():
    # Documented CLI defaults (show_options() in source_c/polyback.c):
    # -d delay=1, -s step=1, -# down_to=1, -n insample unset (whole series).
    order = load_order()
    series = load_series("tests/refs/ar-run_l1000.txt", column=1)

    default_model = tisean.polyback.fit(series, order)
    explicit_model = tisean.polyback.fit(
        series, order, delay=1, insample=len(series), step=1, down_to=1
    )

    assert default_model.n_terms == explicit_model.n_terms
    assert default_model.has_outsample == explicit_model.has_outsample == False
    np.testing.assert_array_equal(default_model.error_in, explicit_model.error_in)
    np.testing.assert_array_equal(default_model.error_out, explicit_model.error_out)
    np.testing.assert_array_equal(
        default_model.level_error_in, explicit_model.level_error_in
    )


def test_fit_has_outsample_reflects_insample_vs_length():
    order = load_order()
    series = load_series("tests/refs/ar-run_l1000.txt", column=1)

    whole = tisean.polyback.fit(series, order)
    assert whole.has_outsample is False
    assert whole.error_out == 0.0
    np.testing.assert_array_equal(whole.level_error_out, np.zeros(whole.n_levels))

    partial = tisean.polyback.fit(series, order, insample=300)
    assert partial.has_outsample is True


def test_fit_rejects_zero_variance_series_like_cli():
    order = load_order()

    with tempfile.TemporaryDirectory() as tmpdir:
        datafile = os.path.join(tmpdir, "constant.dat")
        with open(datafile, "w") as f:
            for _ in range(20):
                f.write("1.0\n")

        result = run_cli_allow_failure([], datafile)
        assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.ones(20)
    with pytest.raises(ValueError):
        tisean.polyback.fit(series, order)


def run_cli_allow_failure(extra_args, datafile, parfile=PARFILE):
    with tempfile.TemporaryDirectory() as tmpdir:
        parfile_copy = os.path.join(tmpdir, os.path.basename(parfile))
        shutil.copyfile(parfile, parfile_copy)
        return subprocess.run(
            ["./bin/polyback"] + extra_args + ["-p", parfile_copy, datafile],
            capture_output=True,
            text=True,
        )


def test_fit_rejects_singular_matrix_like_cli():
    # Two identical terms make the normal-equations matrix rank-deficient.
    series = load_series("tests/refs/ar-run_l1000.txt", column=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        parfile = os.path.join(tmpdir, "singular.pol")
        with open(parfile, "w") as f:
            f.write("0 0\n1 0\n1 0\n")

        result = run_cli_allow_failure([], "tests/refs/ar-run_l1000.txt", parfile=parfile)
        assert result.returncode == SOLVELE_SINGULAR_MATRIX

    order = np.array([[0, 0], [1, 0], [1, 0]], dtype=np.uint32)
    with pytest.raises(ValueError):
        tisean.polyback.fit(series, order)


def test_fit_rejects_too_short_series():
    order = load_order()
    series = np.arange(1.0, 3.0)  # length 2, too short for dim=2/delay=1/step=1

    with pytest.raises(ValueError):
        tisean.polyback.fit(series, order)


def test_fit_rejects_1d_order():
    order = load_order()
    series = load_series("tests/refs/ar-run_l1000.txt", column=1)

    with pytest.raises(ValueError):
        tisean.polyback.fit(series, order[0])


def test_fit_rejects_2d_series():
    order = load_order()
    series = load_series("tests/refs/ar-run_l1000.txt", column=1)
    series_2d = np.stack([series, series])

    with pytest.raises(ValueError):
        tisean.polyback.fit(series_2d, order)
