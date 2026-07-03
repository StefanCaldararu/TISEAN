import subprocess
import numpy as np


def run_lyap_spec():
    result = subprocess.run(
        ["./bin/lyap_spec", "-m1,3", "-k20", "-n50", "./tests/refs/lorenz_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/footers)
        if len(parts) == 4:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_lyap_spec_regression():
    out = run_lyap_spec()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lyap_spec_m1_3k20n50.txt")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
