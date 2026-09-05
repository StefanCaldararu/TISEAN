import subprocess
import numpy as np


def run_lorenz():
    result = subprocess.run(
        ["./bin/lorenz", "-l", "1000"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def run_lorenz_raw(args):
    return subprocess.run(
        ["./bin/lorenz"] + args,
        capture_output=True,
        text=True,
        check=True
    )


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers)
        if len(parts) == 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    return np.array(data)


def test_lorenz_regression():
    out = run_lorenz()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lorenz_l1000.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)


def test_lorenz_flags(tmp_path):
    # -l controls the row count (one extra row for the post-transient
    # starting point written before the sampled loop begins)
    small = parse_output(run_lorenz_raw(["-l", "5"]).stdout)
    large = parse_output(run_lorenz_raw(["-l", "15"]).stdout)
    assert small.shape == (6, 3)
    assert large.shape == (16, 3)

    baseline_out = run_lorenz_raw(["-l", "20"]).stdout
    baseline = parse_output(baseline_out)

    # -R, -S, -B change the trajectory
    for flag in [["-R", "20"], ["-S", "12"], ["-B", "2.0"]]:
        out = parse_output(run_lorenz_raw(["-l", "20"] + flag).stdout)
        assert not np.allclose(out, baseline), flag

    # -f (points sampled per time unit) changes the trajectory
    out = parse_output(run_lorenz_raw(["-l", "20", "-f", "50"]).stdout)
    assert not np.allclose(out, baseline)

    # -x (transient length) changes the trajectory
    out = parse_output(run_lorenz_raw(["-l", "20", "-x", "10"]).stdout)
    assert not np.allclose(out, baseline)

    # -r (noise level) changes the trajectory
    out = parse_output(run_lorenz_raw(["-l", "20", "-r", "0.5"]).stdout)
    assert not np.allclose(out, baseline)

    # two runs with identical flags are byte-identical
    repeat_out = run_lorenz_raw(["-l", "20"]).stdout
    assert repeat_out == baseline_out

    # -o <file> writes the same numbers to that file that stdout carries
    # without it
    outfile = tmp_path / "lorenz.out"
    run_lorenz_raw(["-l", "20", "-o", str(outfile)])
    file_data = parse_output(outfile.read_text())
    np.testing.assert_allclose(file_data, baseline, rtol=1e-7, atol=1e-7)

    # -V0 silences stderr but leaves stdout byte-identical
    verbose = run_lorenz_raw(["-l", "5"])
    quiet = run_lorenz_raw(["-l", "5", "-V0"])
    assert verbose.stdout == quiet.stdout
    assert verbose.stderr != ""
    assert quiet.stderr == ""