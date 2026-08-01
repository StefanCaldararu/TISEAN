import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints integers with "%u ", so this doesn't strictly need the
# tolerance the other bindings' tests need for "%e" output - but we keep
# the same constant name/shape for consistency across the test suite.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

POLYPAR_BIN = os.path.abspath("./bin/polypar")


def run_cli(args, cwd):
    # polypar always writes its parameter combinations to a file (never
    # stdout) - so every invocation needs an explicit -o target to read
    # the result back from.
    outfile = os.path.join(str(cwd), "out.pol")
    subprocess.run(
        [POLYPAR_BIN] + list(args) + ["-o", outfile],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    with open(outfile) as f:
        return f.read()


def parse_output(text):
    rows = [
        [int(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    return np.array(rows)


CASES = [
    # label, dim, order, extra_args
    ("defaults", 2, 3, []),
    ("dim1_order0", 1, 0, ["-m1", "-p0"]),
    ("dim1_order5", 1, 5, ["-m1", "-p5"]),
    ("dim3_order2", 3, 2, ["-m3", "-p2"]),
    ("dim2_order0", 2, 0, ["-m2", "-p0"]),
    ("verbosity_0", 2, 3, ["-V0"]),
]


@pytest.mark.parametrize(
    "label,dim,order,extra_args", CASES, ids=[c[0] for c in CASES]
)
def test_generate_matches_cli(tmp_path, label, dim, order, extra_args):
    out = run_cli(extra_args, cwd=tmp_path)
    cli_rows = parse_output(out)

    result = tisean.polypar.generate(dim=dim, order=order)

    assert result.dim == dim
    assert result.order == order
    assert result.count == cli_rows.shape[0]
    np.testing.assert_allclose(result.params, cli_rows, **CLI_TEXT_TOL)


def test_generate_default_dim_and_order_match_cli_defaults():
    default = tisean.polypar.generate()
    explicit = tisean.polypar.generate(dim=2, order=3)

    assert default.dim == explicit.dim == 2
    assert default.order == explicit.order == 3
    np.testing.assert_array_equal(default.params, explicit.params)


def test_generate_dash_o_output_file_matches_python(tmp_path):
    outfile = tmp_path / "out.pol"
    subprocess.run(
        [POLYPAR_BIN, "-m2", "-p3", "-o", str(outfile)],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_rows = parse_output(outfile.read_text())

    result = tisean.polypar.generate(dim=2, order=3)

    np.testing.assert_allclose(result.params, cli_rows, **CLI_TEXT_TOL)


def test_generate_rejects_dim_zero():
    # Unlike histogram's variance()/rescale_data() exit paths, the CLI has
    # no defined/clean behaviour for -m0: source_c/polypar.c's original
    # make_parameter() recursion is passed dim - 1 as an unsigned depth
    # counter, which underflows for dim == 0 and recurses until the stack
    # overflows. That's not something worth reproducing (or invoking) in a
    # test - only the Python binding's own guard against this is checked
    # here, so a bad `dim` can never crash the interpreter.
    with pytest.raises(ValueError):
        tisean.polypar.generate(dim=0, order=3)


def test_params_shape_and_dtype():
    result = tisean.polypar.generate(dim=2, order=3)

    assert result.params.shape == (result.count, result.dim)
    assert result.params.sum(axis=1).max() <= result.order
