import struct
import subprocess

import numpy as np
import pytest

import tisean

HENON_BIN = "./bin/henon"

# henon_api.c's iteration loop already rounds each output point to single
# precision before storing it (out[2*n] = (double)(float)xn, mirroring the
# original Fortran's `write(iunit,*) real(xn), real(yn)`), and the CLI just
# prints that already-rounded double via "%.9e" -- so only ~7 significant
# figures are ever meaningful. 1e-7 is too tight; this matches the floor
# used for the same reason in test_ar_run_bindings.py.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

# Matches henon.c's own defaults (see its show_options()).
CLI_DEFAULTS = dict(a=1.4, b=0.3, x0=0.68587, y0=0.65876, ntrans=10000)
BASE_LENGTH = 20


def f32(v):
    """Round a Python float through IEEE-754 single precision."""
    return struct.unpack("f", struct.pack("f", v))[0]


def run_cli(args):
    result = subprocess.run(
        [HENON_BIN] + args, capture_output=True, text=True, check=True
    )
    return result.stdout


def parse_series(text):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append([float(parts[0]), float(parts[1])])
    return np.array(rows)


def run_py_generate(length, a, b, x0, y0, ntrans):
    # henon.c passes a/b/x0/y0 through the equivalent of fcan() -- an
    # unconditional double(float(v)) round trip -- before iterating, even
    # for its own defaults (source_c/henon.c, right before the length!=0
    # branch). Henon is chaotic (Lyapunov exponent ~0.42/step), so with the
    # default ntrans=10000 transient, a raw, unrounded parameter that's
    # ~1e-7 off from what the CLI actually used would diverge onto an
    # unrelated point on the attractor long before the first output point
    # is reached. Apply the same rounding here so both sides iterate the
    # exact same doubles.
    series = tisean.henon.generate(
        length=length, a=f32(a), b=f32(b), x0=f32(x0), y0=f32(y0), ntrans=ntrans
    )
    assert series.shape == (length, 2)
    # The binding returns full double precision; round to float32 to match
    # what the CLI printed (see CLI_TEXT_TOL above) rather than loosening
    # the tolerance further.
    return series.astype(np.float32).astype(np.float64)


FLAG_CASES = [
    ("l", ["-l7"], dict(length=7)),
    ("A", ["-l20", "-A1.2"], dict(a=1.2)),
    ("B", ["-l20", "-B0.5"], dict(b=0.5)),
    ("X", ["-l20", "-X0.1"], dict(x0=0.1)),
    ("Y", ["-l20", "-Y0.2"], dict(y0=0.2)),
    ("x", ["-l20", "-x500"], dict(ntrans=500)),
]


@pytest.mark.parametrize(
    "label,cli_args,overrides", FLAG_CASES, ids=[c[0] for c in FLAG_CASES]
)
def test_generate_matches_cli(label, cli_args, overrides):
    cli_series = parse_series(run_cli(cli_args))

    params = dict(length=BASE_LENGTH, **CLI_DEFAULTS)
    params.update(overrides)

    py_series = run_py_generate(**params)
    assert py_series.shape == cli_series.shape
    np.testing.assert_allclose(py_series, cli_series, **CLI_TEXT_TOL)


def test_generate_defaults_match_documented_cli_defaults():
    implicit = tisean.henon.generate(length=BASE_LENGTH)
    explicit = tisean.henon.generate(length=BASE_LENGTH, **CLI_DEFAULTS)
    np.testing.assert_array_equal(implicit, explicit)


def test_generate_rejects_zero_length():
    with pytest.raises(ValueError):
        tisean.henon.generate(length=0)
