import subprocess
import numpy as np

INPUT = "./tests/refs/ar-run_l1000.txt"
N = 300


def run_surrogates(*args):
    result = subprocess.run(
        ["./bin/surrogates", *args, "-l%d" % N, INPUT],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_blocks(text):
    """Split stdout into one array per surrogate (blocks are separated by
    the blank lines xwritecfile emits between successive -n surrogates)."""
    blocks = []
    current = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(np.array(current))
                current = []
            continue
        parts = line.strip().split()
        try:
            current.append(float(parts[0]))
        except (ValueError, IndexError):
            continue
    if current:
        blocks.append(np.array(current))
    return blocks


def load_input():
    return np.loadtxt(INPUT)[:N]


def periodogram(x):
    x = np.asarray(x, dtype=float)
    return np.abs(np.fft.rfft(x - x.mean())) ** 2


def spectral_distance(x, y):
    return np.sum((periodogram(x) - periodogram(y)) ** 2)


def test_surrogate_is_a_permutation_of_the_input():
    out = run_surrogates("-n1", "-I1")
    surrogate = parse_blocks(out)[0]
    inp = load_input()

    np.testing.assert_allclose(
        np.sort(surrogate), np.sort(inp), atol=1e-6
    )


def test_surrogate_length_matches_nless():
    out = run_surrogates("-n1", "-I1")
    surrogate = parse_blocks(out)[0]

    # nless(300) == 300: the largest 5-smooth number at or below 300 is 300
    assert len(surrogate) == N


def test_spectrum_exact_flag_matches_input_spectrum_more_closely():
    inp = load_input()

    default_surrogate = parse_blocks(run_surrogates("-n1", "-I1"))[0]
    spec_exact_surrogate = parse_blocks(run_surrogates("-n1", "-I1", "-S"))[0]

    d_default = spectral_distance(default_surrogate, inp)
    d_spec_exact = spectral_distance(spec_exact_surrogate, inp)

    assert d_spec_exact < d_default


def test_two_surrogates_in_one_run_differ():
    blocks = parse_blocks(run_surrogates("-n2", "-I1"))

    assert len(blocks) == 2
    assert not np.allclose(blocks[0], blocks[1])


def test_seed_controls_surrogate_identity():
    surrogate_i1_a = parse_blocks(run_surrogates("-n1", "-I1"))[0]
    surrogate_i1_b = parse_blocks(run_surrogates("-n1", "-I1"))[0]
    surrogate_i2 = parse_blocks(run_surrogates("-n1", "-I2"))[0]

    np.testing.assert_array_equal(surrogate_i1_a, surrogate_i1_b)
    assert not np.allclose(surrogate_i1_a, surrogate_i2)
