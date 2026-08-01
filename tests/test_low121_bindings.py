import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LOW121_BIN = os.path.abspath("./bin/low121")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [LOW121_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    return np.array([float(line) for line in text.splitlines() if line.strip()])


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


CASES = [
    # label, args, datafile, column, length, exclude, iterations
    ("defaults", [], AR_RUN, 1, None, 0, 1),
    ("iterations_2", ["-i2"], AR_RUN, 1, None, 0, 2),
    ("iterations_5", ["-i5"], AR_RUN, 1, None, 0, 5),
    ("column_2", ["-c2"], HENON, 2, None, 0, 1),
    ("length", ["-l200"], AR_RUN, 1, 200, 0, 1),
    ("exclude", ["-x50"], AR_RUN, 1, None, 50, 1),
    ("length_and_exclude", ["-l100", "-x50"], AR_RUN, 1, 100, 50, 1),
    ("verbosity_0", ["-V0"], AR_RUN, 1, None, 0, 1),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,iterations",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_filter_matches_cli(label, args, datafile, column, length, exclude, iterations):
    out = run_cli(args + [datafile])
    cli_result = parse_output(out)

    series = load_column(datafile, column, length=length, exclude=exclude)
    py_result = tisean.low121.filter(series, iterations=iterations)

    assert py_result.shape == series.shape
    np.testing.assert_allclose(py_result, cli_result, **CLI_TEXT_TOL)


def test_filter_matches_cli_with_per_iteration_output_files(tmp_path):
    # -V2 (VER_USR1) makes the CLI additionally write each iteration to a
    # file named after the (here, copied-into-tmp_path) datafile, but the
    # final stdout output (once stdo stays true, i.e. no -o given) must
    # still match the plain (-V1) case - verbosity only controls side
    # output, not the computed values.
    datafile_copy = tmp_path / "ar-run_l1000.txt"
    datafile_copy.write_text(open(AR_RUN).read())

    out = run_cli(["-V2", "-i3", str(datafile_copy)], cwd=tmp_path)
    cli_result = parse_output(out)

    series = load_column(AR_RUN, 1)
    py_result = tisean.low121.filter(series, iterations=3)

    np.testing.assert_allclose(py_result, cli_result, **CLI_TEXT_TOL)


def test_filter_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.low"
    run_cli(["-i2", "-o" + str(outfile), AR_RUN])

    written = outfile.with_name(outfile.name + ".2")
    cli_result = parse_output(written.read_text())

    series = load_column(AR_RUN, 1)
    py_result = tisean.low121.filter(series, iterations=2)

    np.testing.assert_allclose(py_result, cli_result, **CLI_TEXT_TOL)


def test_filter_default_iterations_is_one():
    series = load_column(AR_RUN, 1)

    default = tisean.low121.filter(series)
    explicit = tisean.low121.filter(series, iterations=1)

    np.testing.assert_array_equal(default, explicit)
