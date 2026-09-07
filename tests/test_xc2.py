import subprocess
import numpy as np


def run_xc2():
    result = subprocess.run(
        ["./bin/xc2", "-M1,2", "-d1", "-n50", "-N100", "-r0.01",
         "-o", "./tests/refs/xc2_run_out.txt",
         "./tests/refs/ar-run_l1000.txt", "./tests/refs/henon_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers like "#m= 2")
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_xc2_regression():
    run_xc2()
    with open("./tests/refs/xc2_run_out.txt") as f:
        out = f.read()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/xc2_M12d1n50N100r001.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
