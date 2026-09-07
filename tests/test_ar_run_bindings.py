import subprocess
import sys

import numpy as np
import pytest

import tisean

AR_RUN_BIN = "./bin/ar-run"

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)


def run_cli(args, coeff_file):
    result = subprocess.run(
        [AR_RUN_BIN] + args + [str(coeff_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_series(text):
    return np.array([float(x) for x in text.split()])


def write_coeff_file(path, var, coeff):
    path.write_text("\n".join([str(var)] + [str(c) for c in coeff]) + "\n")


def run_py_generate(coeff, var, length, ntrans, seed):
    # rand.c's rnd_init() only actually (re)seeds once per process (a
    # static was_set flag - see source_c/routines/rand.c), so calling
    # generate() here in the same pytest process - after other
    # tests/parametrizations have already called it - would compare
    # against stale RNG state instead of a fresh seed like the CLI
    # subprocess gets. Do the call in its own fresh subprocess too.
    script = (
        "import tisean\n"
        f"series = tisean.ar_run.generate({coeff!r}, {var!r}, length={length}, "
        f"ntrans={ntrans}, seed={seed})\n"
        "print(series.tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return np.array(eval(result.stdout))


CASES = [
    # label, coeff, var, length, ntrans, seed
    ("p1", [0.5], 0.1, 200, 0, 1),
    ("p1_transients", [0.7], 0.2, 300, 100, 7),
    ("p3", [0.4, -0.2, 0.1], 0.05, 500, 50, 42),
    ("p2_default_ntrans", [0.6, -0.1], 0.15, 250, 10000, 123),
]


@pytest.mark.parametrize(
    "label,coeff,var,length,ntrans,seed", CASES, ids=[c[0] for c in CASES]
)
def test_generate_matches_cli(label, coeff, var, length, ntrans, seed, tmp_path):
    coeff_file = tmp_path / "coeffs.txt"
    write_coeff_file(coeff_file, var, coeff)

    cli_text = run_cli(
        [f"-l{length}", f"-p{len(coeff)}", f"-x{ntrans}", f"-I{seed}"], coeff_file
    )
    cli_series = parse_series(cli_text)
    assert cli_series.shape == (length,)

    py_series = run_py_generate(coeff, var, length, ntrans, seed)
    np.testing.assert_allclose(py_series, cli_series, **CLI_TEXT_TOL)


def test_generate_matches_cli_default_ntrans_and_poles(tmp_path):
    # No -x/-p at all: defaults are ntrans=10000 (from source_c/ar-run.c's
    # own global) and poles auto-detected from the number of coefficient
    # lines in the file.
    coeff, var, length, seed = [0.3, 0.2], 0.1, 100, 5
    coeff_file = tmp_path / "coeffs.txt"
    write_coeff_file(coeff_file, var, coeff)

    cli_text = run_cli([f"-l{length}", f"-I{seed}"], coeff_file)
    cli_series = parse_series(cli_text)
    assert cli_series.shape == (length,)

    py_series = run_py_generate(coeff, var, length, ntrans=10000, seed=seed)
    np.testing.assert_allclose(py_series, cli_series, **CLI_TEXT_TOL)


def test_generate_is_deterministic_given_a_seed():
    script = (
        "import tisean\n"
        "print(tisean.ar_run.generate([0.5, -0.2], 0.1, length=50, ntrans=20, "
        "seed=99).tolist())\n"
    )
    a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert a.stdout == b.stdout


def test_generate_rejects_empty_coeff():
    with pytest.raises(ValueError):
        tisean.ar_run.generate([], 0.1, length=10, ntrans=0, seed=1)
