import subprocess
import numpy as np


def run_spikeauto():
    result = subprocess.run(
        ["./bin/spikeauto", "-d1", "-D30", "tests/refs/spike_times.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


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


def test_spikeauto_regression():
    out = run_spikeauto()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/spikeauto_d1D30.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
