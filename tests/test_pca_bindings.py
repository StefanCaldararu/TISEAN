import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VARIANCE_VAR_EQ_ZERO = 23

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
PCA_BIN = os.path.abspath("./bin/pca")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [PCA_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_values(text):
    """-W0 output: 'i eigenvalue' rows, in the order the CLI printed them
    (already sorted descending)."""
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals.append(float(line.split()[1]))
    return np.array(vals)


def parse_vectors(text):
    """-W1 output: '#i eigenvalue' comment rows followed by dimemb rows of
    dimemb floats each (the eigenvector matrix, columns already sorted like
    the eigenvalues)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)


def variance_mean(row):
    """Mirrors variance.c's mean: a plain sequential sum divided by the
    count, not numpy's (more accurate) pairwise-summation .mean()."""
    total = 0.0
    for v in row:
        total += v
    return total / len(row)


def load_columns(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order), and finally center each row like the CLI's own
    per-column variance()-based centering in main(). Returns a (dim,
    length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    series = raw.T.copy()
    for row in series:
        row -= variance_mean(row)
    return series


# label, extra CLI args (dim/emb/delay/columns encoded separately below),
# datafile, columns (1-indexed, defines dim too), length, exclude, emb, delay
CASES = [
    ("defaults", [], HENON, [1, 2], None, 0, 1, 1),
    ("dim1", ["-m1,1"], AR_RUN, [1], None, 0, 1, 1),
    ("dim3", ["-m3,1"], LORENZ, [1, 2, 3], None, 0, 1, 1),
    ("emb2", ["-m2,2"], HENON, [1, 2], None, 0, 2, 1),
    ("emb2_delay2", ["-m2,2", "-d2"], HENON, [1, 2], None, 0, 2, 2),
    ("columns_reordered", ["-m2,1", "-c2,1"], HENON, [2, 1], None, 0, 1, 1),
    ("column_subset", ["-m2,1", "-c3,1"], LORENZ, [3, 1], None, 0, 1, 1),
    ("length", ["-m2,1", "-l500"], HENON, [1, 2], 500, 0, 1, 1),
    ("exclude", ["-m2,1", "-x100"], HENON, [1, 2], None, 100, 1, 1),
    ("length_and_exclude", ["-m2,1", "-l500", "-x100"], HENON, [1, 2], 500, 100, 1, 1),
    ("verbosity_0", ["-V0"], HENON, [1, 2], None, 0, 1, 1),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,length,exclude,emb,delay",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli_eigenvalues_and_eigenvectors(
    label, args, datafile, columns, length, exclude, emb, delay
):
    dim = len(columns)

    values_out = run_cli(args + ["-W0", datafile])
    cli_values = parse_values(values_out)

    vectors_out = run_cli(args + ["-W1", datafile])
    cli_vectors = parse_vectors(vectors_out)

    series = load_columns(datafile, columns, length=length, exclude=exclude)
    pca = tisean.pca.compute(series, emb=emb, delay=delay)

    assert pca.dimemb == dim * emb
    np.testing.assert_allclose(pca.eigenvalues, cli_values, **CLI_TEXT_TOL)
    np.testing.assert_allclose(pca.eigenvectors, cli_vectors, **CLI_TEXT_TOL)


def test_compute_default_emb_and_delay_are_one():
    series = load_columns(HENON, [1, 2])

    default = tisean.pca.compute(series)
    explicit = tisean.pca.compute(series, emb=1, delay=1)

    assert default.dimemb == explicit.dimemb == 2
    np.testing.assert_array_equal(default.eigenvalues, explicit.eigenvalues)
    np.testing.assert_array_equal(default.eigenvectors, explicit.eigenvectors)


def test_cli_rejects_constant_column_like_original(tmp_path):
    # The CLI centers each column with variance()'s own zero-variance guard
    # (source_c/pca.c's main(), untouched by the pca_compute() extraction)
    # before ever reaching the PCA math, so a constant column is rejected
    # at the CLI level with the same exit code as before the refactor.
    datafile = tmp_path / "constant.txt"
    rows = "\n".join(f"1.5 {i}" for i in range(20))
    datafile.write_text(rows + "\n")

    result = subprocess.run(
        [PCA_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO


def test_compute_rejects_emb_zero():
    # emb=0 makes the CLI's own covariance-matrix build divide component
    # indices by EMB (source_c/pca.c's make_pca(), now pca_compute()'s
    # i/emb), which is undefined behaviour in C (the CLI actually crashes
    # with a SIGABRT on this input rather than exiting cleanly). The
    # Python binding guards against it up front instead of crashing the
    # interpreter.
    series = load_columns(HENON, [1, 2])

    with pytest.raises(ValueError):
        tisean.pca.compute(series, emb=0)


def test_compute_rejects_series_too_short_for_emb_delay():
    # emb=2, delay=2 needs length > (emb-1)*delay == 2.
    series = load_columns(HENON, [1, 2], length=2)

    with pytest.raises(ValueError):
        tisean.pca.compute(series, emb=2, delay=2)


def test_compute_rejects_non_2d_series():
    with pytest.raises(ValueError):
        tisean.pca.compute(np.zeros(10))
