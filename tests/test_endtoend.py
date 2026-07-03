import re
import subprocess
import numpy as np


def run_endtoend():
    result = subprocess.run(
        ["./bin/endtoend", "-l300", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    # endtoend prints one block per offset, each block reporting
    # length, offset, lost%, jump%, slip%, weighted% as labelled text
    # rather than a plain numeric table. Extract the numbers in order.
    data = []
    for block in text.strip().split("\n\n"):
        nums = []
        for line in block.splitlines():
            nums.extend(re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line))
        if len(nums) == 6:
            try:
                data.append([float(n) for n in nums])
            except ValueError:
                continue

    return np.array(data)


def test_endtoend_regression():
    out = run_endtoend()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/endtoend_l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
