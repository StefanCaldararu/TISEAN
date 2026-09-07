import os
import subprocess
import sys

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
MAKENOISE_BIN = os.path.abspath("./bin/makenoise")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [MAKENOISE_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Rows of noisy data, shape (length, dim)."""
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    return np.array(rows)


def load_columns(path, columns, length=None, exclude=0):
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


def run_py_add(datafile, columns, length=None, exclude=0, add_kwargs=None):
    """Runs tisean.makenoise.add() in a fresh subprocess.

    rand.c's rnd_init() only actually (re)seeds once per process (a static
    was_set flag guards it - see source_c/routines/rand.c), so comparing
    against a specific-seed CLI run must happen in its own fresh process,
    or a second in-process add() call would silently continue from
    whatever RNG state the first call left behind instead of reseeding.
    """
    add_kwargs = dict(add_kwargs or {})
    kwargs_src = ", ".join(f"{k}={v!r}" for k, v in add_kwargs.items())
    script = (
        "import numpy as np, tisean\n"
        f"raw = np.loadtxt({datafile!r})\n"
        "if raw.ndim == 1:\n"
        "    raw = raw.reshape(-1, 1)\n"
        f"raw = raw[{exclude}:]\n"
        f"if {length!r} is not None:\n"
        f"    raw = raw[:{length!r}]\n"
        f"series = raw[:, {[c - 1 for c in columns]!r}].T.copy()\n"
        f"result = tisean.makenoise.add(series, {kwargs_src})\n"
        "print(result.series.tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return np.array(eval(result.stdout))


CASES = [
    # label, args, datafile, columns, length, exclude, add_kwargs
    ("defaults", [], AR_RUN, [1], None, 0, {}),
    ("noiselevel_pct", ["-%10", "-I5"], AR_RUN, [1], None, 0,
     dict(noiselevel=0.10, seed=5)),
    ("absolute", ["-r0.1", "-I5"], AR_RUN, [1], None, 0,
     dict(noiselevel=0.1, absolute=True, seed=5)),
    ("absolute_gaussian", ["-r0.2", "-g", "-I5"], AR_RUN, [1], None, 0,
     dict(noiselevel=0.2, absolute=True, gaussian=True, seed=5)),
    ("gaussian_relative", ["-g", "-I5"], AR_RUN, [1], None, 0,
     dict(gaussian=True, seed=5)),
    ("seed", ["-I12345"], AR_RUN, [1], None, 0, dict(seed=12345)),
    ("column_2", ["-c2", "-I5"], HENON, [2], None, 0, dict(seed=5)),
    ("multi_dim", ["-m2", "-c1,2", "-I5"], HENON, [1, 2], None, 0, dict(seed=5)),
    ("length", ["-l200", "-I5"], AR_RUN, [1], 200, 0, dict(seed=5)),
    ("exclude", ["-x50", "-I5"], AR_RUN, [1], None, 50, dict(seed=5)),
    ("length_and_exclude", ["-l100", "-x50", "-I5"], AR_RUN, [1], 100, 50,
     dict(seed=5)),
    ("verbosity_0", ["-V0", "-I5"], AR_RUN, [1], None, 0, dict(seed=5)),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,add_kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_add_matches_cli(label, args, datafile, columns, length, exclude, add_kwargs):
    out = run_cli(args + [datafile])
    cli_result = parse_output(out)

    py_result = run_py_add(datafile, columns, length=length, exclude=exclude,
                            add_kwargs=add_kwargs)

    assert py_result.T.shape == cli_result.shape
    np.testing.assert_allclose(py_result.T, cli_result, **CLI_TEXT_TOL)


def test_add_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.noi"
    run_cli(["-I5", "-o" + str(outfile), AR_RUN])
    cli_result = parse_output(outfile.read_text())

    py_result = run_py_add(AR_RUN, [1], add_kwargs=dict(seed=5))

    np.testing.assert_allclose(py_result.T, cli_result, **CLI_TEXT_TOL)


def test_add_matches_cli_dash_0_just_create():
    # -0 requires -r and -l and generates the noise itself, rather than
    # reading a datafile - equivalent, on the Python side, to adding
    # absolute noise to an all-zero series of the requested length.
    out = run_cli(["-I", "1", "-l", "1000", "-r", "1", "-0"])
    cli_result = parse_output(out)

    ref = np.loadtxt("tests/refs/makenoise_I1l1000r1.txt")
    np.testing.assert_allclose(cli_result.flatten(), ref, rtol=1e-7, atol=1e-7)

    script = (
        "import numpy as np, tisean\n"
        "series = np.zeros((1, 1000))\n"
        "result = tisean.makenoise.add(series, noiselevel=1, absolute=True, seed=1)\n"
        "print(result.series.tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    py_result = np.array(eval(result.stdout))

    np.testing.assert_allclose(py_result[0], cli_result.flatten(), **CLI_TEXT_TOL)


def test_add_default_kwargs_match_documented_cli_defaults():
    # CLI defaults: -% 5 (noiselevel=0.05), no -r (absolute=False), no -g
    # (gaussian=False), fixed default seed 3441341. Each run_py_add() call
    # is its own fresh process, so two such calls with identical
    # (logical) settings must reproduce bit-identical output.
    default_out = run_py_add(AR_RUN, [1])
    explicit_out = run_py_add(
        AR_RUN, [1],
        add_kwargs=dict(noiselevel=0.05, absolute=False, gaussian=False, seed=3441341),
    )
    np.testing.assert_array_equal(default_out, explicit_out)


def test_add_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [MAKENOISE_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full((1, 20), 1.5)
    with pytest.raises(ValueError):
        tisean.makenoise.add(series)


def test_add_absolute_allows_constant_data():
    # With absolute=True, per-column variance is never computed, so
    # constant (zero-variance) input is not an error - matching the CLI,
    # which only calls variance() when -r is not given.
    series = np.full((1, 20), 1.5)
    result = tisean.makenoise.add(series, noiselevel=0.1, absolute=True)
    assert result.series.shape == (1, 20)


def test_add_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        tisean.makenoise.add(np.zeros(10))


def test_add_rejects_empty_series():
    with pytest.raises(ValueError):
        tisean.makenoise.add(np.zeros((1, 0)))
