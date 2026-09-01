import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def outdir():
    # randomize.f stores filenames in a Fortran character*72; pytest's
    # tmp_path is an absolute path well over 72 characters and gets
    # silently truncated, so surrogates land nowhere the test expects.
    # A short path relative to the repo root (the test runner's cwd)
    # stays under the limit regardless of how deep /tmp nests.
    d = tempfile.mkdtemp(prefix="t", dir="tests")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# Flags shared by every run: -a/-S/-s/-z make the exponential cooling
# schedule terminate almost immediately on a 200-point input (default
# flags never terminate, see cool/exp.f), -V0 silences progress chatter.
COOL_ARGS = ["-a0.5", "-S200", "-s50", "-z40", "-V0"]

BINARIES = {
    "randomize_auto_exp_random": dict(
        input="tests/refs/ar-run_l1000.txt",
        args=["-D5", "-l200"],
        invariant="values",
    ),
    "randomize_autop_exp_random": dict(
        input="tests/refs/ar-run_l1000.txt",
        args=["-D5", "-l200"],
        invariant="values",
    ),
    "randomize_spikeauto_exp_random": dict(
        input="tests/refs/spike_times.txt",
        args=["-d1", "-D30", "-l200"],
        invariant="diffs",
    ),
    "randomize_spikespec_exp_event": dict(
        input="tests/refs/spike_times.txt",
        args=["-l200"],
        invariant="diffs",
    ),
    "randomize_uneven_exp_random": dict(
        # perm/random.f's exch() only swaps within the first nx elements
        # of the shared common block, i.e. column 1. Duplicating column 1
        # as both "value" and "time" (-c1,1) keeps the fixture a plain
        # single-column file while still satisfying uneven's 2-column
        # (value, time) input shape.
        input="tests/refs/ar-run_l1000.txt",
        args=["-c1,1", "-d0.05", "-D2", "-l200"],
        invariant="values",
        value_col=0,
    ),
}

BINARY_NAMES = list(BINARIES.keys())


def run_randomize(binary, outfile, seed=1, nsur=1):
    entry = BINARIES[binary]
    cmd = (
        ["./bin/" + binary]
        + entry["args"]
        + COOL_ARGS
        + ["-I" + str(seed), "-n" + str(nsur), "-o", str(outfile)]
        + [entry["input"]]
    )
    return subprocess.run(cmd, capture_output=True, timeout=60)


def load_output(entry, path):
    data = np.loadtxt(path)
    if data.ndim == 2:
        return data[:, entry.get("value_col", 0)]
    return data


@pytest.mark.parametrize("binary", BINARY_NAMES)
def test_exits_cleanly(binary, outdir):
    result = run_randomize(binary, outdir / "out")
    assert result.returncode == 0, result.stderr.decode(errors="replace")


@pytest.mark.parametrize("binary", BINARY_NAMES)
def test_deterministic_under_fixed_seed(binary, outdir):
    out1 = outdir / "out1"
    out2 = outdir / "out2"
    r1 = run_randomize(binary, out1, seed=1)
    r2 = run_randomize(binary, out2, seed=1)
    assert r1.returncode == 0, r1.stderr.decode(errors="replace")
    assert r2.returncode == 0, r2.stderr.decode(errors="replace")
    assert out1.read_bytes() == out2.read_bytes()


@pytest.mark.parametrize("binary", BINARY_NAMES)
def test_seed_reseeds(binary, outdir):
    out1 = outdir / "out1"
    out2 = outdir / "out2"
    r1 = run_randomize(binary, out1, seed=1)
    r2 = run_randomize(binary, out2, seed=2)
    assert r1.returncode == 0, r1.stderr.decode(errors="replace")
    assert r2.returncode == 0, r2.stderr.decode(errors="replace")
    assert out1.read_bytes() != out2.read_bytes()


@pytest.mark.parametrize("binary", BINARY_NAMES)
def test_surrogate_is_permutation(binary, outdir):
    entry = BINARIES[binary]
    outfile = outdir / "out"
    result = run_randomize(binary, outfile, seed=1)
    assert result.returncode == 0, result.stderr.decode(errors="replace")

    output = load_output(entry, outfile)
    input_data = np.loadtxt(entry["input"], max_rows=200)
    if input_data.ndim == 2:
        input_data = input_data[:, entry.get("value_col", 0)]

    if entry["invariant"] == "values":
        np.testing.assert_allclose(np.sort(output), np.sort(input_data),
                                    rtol=1e-6)
    else:
        # The "event" permutation scheme (and the times->intervals->times
        # round trip used by default for spike trains) swaps adjacent
        # inter-spike intervals rather than raw point values, so the
        # invariant holds over sorted successive differences, not over
        # the sorted values themselves.
        input_sorted = np.sort(input_data)
        np.testing.assert_allclose(
            np.sort(np.diff(output)), np.sort(np.diff(input_sorted)),
            rtol=1e-6)


@pytest.mark.parametrize("binary", BINARY_NAMES)
def test_n_controls_surrogate_count(binary, outdir):
    outfile = outdir / "out"
    result = run_randomize(binary, outfile, seed=1, nsur=2)
    assert result.returncode == 0, result.stderr.decode(errors="replace")

    produced = sorted(outdir.glob("out*"))
    assert len(produced) == 2
    for f in produced:
        assert f.stat().st_size > 0
