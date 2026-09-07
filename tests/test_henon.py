import struct
import subprocess
import numpy as np


def f32(v):
    """Round a Python float through IEEE-754 single precision."""
    return struct.unpack('f', struct.pack('f', v))[0]


def run_henon():
    result = subprocess.run(
        ["./bin/henon", "-l", "1000"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def run_henon_raw(args):
    return subprocess.run(
        ["./bin/henon"] + args,
        capture_output=True,
        text=True,
        check=True
    )


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_henon_regression():
    out = run_henon()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/henon_l1000.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)


def henon_recurrence(n, ntrans=10000, a=1.4, b=0.3, x0=.68587, y0=.65876):
    # Parameters and the initial condition pass through fcan() in the
    # Fortran, which returns a single precision real that is then widened
    # to double -- so the values actually used are double(float32(v)), not
    # the literal decimals.
    a = f32(a)
    b = f32(b)
    x = f32(x0)
    y = f32(y0)

    rows = []
    nn = -ntrans
    while True:
        nn += 1
        # xn=1.-a*xo**2+b*yo : xo**2 binds tighter than the multiply, so
        # this is 1 - a*(x*x) + b*y, not 1 - (a*x)*x.
        xn = 1. - a * (x * x) + b * y
        yn = x
        x, y = xn, yn
        if nn < 1:
            continue
        # write(iunit,*) real(xn), real(yn) -- output is single precision
        rows.append((f32(x), f32(y)))
        if n != 0 and nn >= n:
            break
    return np.array(rows)


def test_henon_matches_recurrence():
    out = run_henon_raw(["-l1000"]).stdout
    data = parse_output(out)

    ref = henon_recurrence(1000)

    np.testing.assert_allclose(data, ref, rtol=1e-6, atol=1e-6)


def test_henon_flags(short_outdir):
    baseline = parse_output(run_henon_raw(["-l5", "-x0"]).stdout)

    # -A, -B, -X, -Y each change the output
    for flag in ["-A1.2", "-B0.5", "-X0.1", "-Y0.2"]:
        out = parse_output(run_henon_raw(["-l5", "-x0", flag]).stdout)
        assert not np.allclose(out, baseline), flag

    # -l controls the row count
    assert len(parse_output(run_henon_raw(["-l5", "-x0"]).stdout)) == 5
    assert len(parse_output(run_henon_raw(["-l12", "-x0"]).stdout)) == 12

    # -x shifts the sequence: skipping N transients up front just drops
    # the first N rows of the longer, unshifted run.
    long_run = parse_output(run_henon_raw(["-l50", "-x0"]).stdout)
    shifted = parse_output(run_henon_raw(["-l45", "-x5"]).stdout)
    np.testing.assert_allclose(shifted, long_run[5:], rtol=1e-6, atol=1e-6)

    # -o <file> writes the same numbers to that file that stdout carries
    # without it
    stdout_data = parse_output(run_henon_raw(["-l10", "-x0"]).stdout)
    outfile = short_outdir / "henon.out"
    run_henon_raw(["-l10", "-x0", "-o", str(outfile)])
    file_data = parse_output(outfile.read_text())
    np.testing.assert_allclose(file_data, stdout_data, rtol=1e-6, atol=1e-6)

    # -V0 silences stderr but leaves stdout byte-identical
    verbose = run_henon_raw(["-l5", "-x0"])
    quiet = run_henon_raw(["-l5", "-x0", "-V0"])
    assert verbose.stdout == quiet.stdout
    assert verbose.stderr != ""
    assert quiet.stderr == ""