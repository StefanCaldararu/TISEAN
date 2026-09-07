import subprocess
import numpy as np

INPUT = "./tests/refs/ar-run_l1000.txt"
N = 300


def run_addnoise(*args):
    result = subprocess.run(
        ["./bin/addnoise", *args, "-l%d" % N, INPUT],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def run_makenoise(*args):
    result = subprocess.run(
        ["./bin/makenoise", *args, "-l%d" % N, INPUT],
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
        if len(parts) == 1:
            try:
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def load_input():
    return np.loadtxt(INPUT)[:N]


def test_addnoise_gaussian_absolute_amplitude():
    eps = 0.01
    data = parse_output(run_addnoise("-r%g" % eps))
    noise = data - load_input()

    assert len(noise) == N
    assert abs(noise.std() - eps) < 0.1 * eps


def test_addnoise_gaussian_relative_amplitude():
    frac = 0.1
    inp = load_input()
    rms = inp.std()
    expected = frac * rms

    data = parse_output(run_addnoise("-v%g" % frac))
    noise = data - inp

    assert abs(noise.std() - expected) < 0.1 * expected


def test_addnoise_uniform():
    eps = 0.05
    data = parse_output(run_addnoise("-r%g" % eps, "-u"))
    noise = data - load_input()

    assert len(noise) == N
    # every difference lies in [0, eps], allowing tiny float slop
    assert noise.min() >= -1e-6
    assert noise.max() <= eps + 1e-6
    assert abs(noise.mean() - eps / 2) < 0.1 * eps


def test_addnoise_preserves_length_and_columns():
    out = run_addnoise("-r0.01")
    lines = [l for l in out.splitlines() if l.strip()]

    assert len(lines) == N
    for line in lines:
        assert len(line.strip().split()) == 1


def test_addnoise_agrees_with_makenoise():
    eps = 0.01
    inp = load_input()

    noise_addnoise = parse_output(run_addnoise("-r%g" % eps)) - inp
    noise_makenoise = parse_output(run_makenoise("-r%g" % eps, "-g")) - inp

    std_addnoise = noise_addnoise.std()
    std_makenoise = noise_makenoise.std()

    assert abs(std_addnoise - std_makenoise) / std_addnoise < 0.1
