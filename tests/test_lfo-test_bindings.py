import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# figures), so comparisons against numbers parsed from its stdout can never
# be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11
ONESTEP_TOO_FEW_POINTS = 81

AR_RUN = "tests/refs/ar-run_l1000.txt"
LORENZ = "tests/refs/lorenz_l1000.txt"
LFO_TEST_BIN = os.path.abspath("./bin/lfo-test")


def run_cli(args, datafile):
    result = subprocess.run(
        [LFO_TEST_BIN] + args + [datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_output(text, comp):
    """lfo-test always prints a "#Relative forecast errors..." header
    followed by `comp` rms-error lines (one float each, "# %e" under -V2 /
    "%e" otherwise), and under -V2 additionally one row of `comp`
    space-separated %e individual-error values per scanned point."""
    lines = [line for line in text.splitlines() if line.strip() != ""]
    body = lines[1:]

    rms = []
    idx = 0
    for _ in range(comp):
        line = body[idx].strip()
        if line.startswith("#"):
            line = line[1:].strip()
        rms.append(float(line))
        idx += 1

    individual_rows = []
    for line in body[idx:]:
        parts = line.strip().split()
        if len(parts) != comp:
            continue
        individual_rows.append([float(p) for p in parts])

    return np.array(rms), np.array(individual_rows)


def load_multi_series(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (dim, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


# label, cli_args, datafile, columns, kwargs (only overrides from
# tisean.lfo_test.compute's defaults need to be listed)
CASES = [
    ("defaults", ["-m1,2", "-d1", "-n50", "-k10"], AR_RUN, [1],
     dict(minn=10, iterations=50)),
    ("embed_3", ["-m1,3", "-d1", "-n50", "-k10"], AR_RUN, [1],
     dict(embed=3, minn=10, iterations=50)),
    ("delay_2", ["-m1,2", "-d2", "-n50", "-k10"], AR_RUN, [1],
     dict(delay=2, minn=10, iterations=50)),
    ("step_2", ["-m1,2", "-d1", "-n50", "-k10", "-s2"], AR_RUN, [1],
     dict(minn=10, iterations=50, step=2)),
    ("causal_3", ["-m1,2", "-d1", "-n50", "-k10", "-s2", "-C3"], AR_RUN, [1],
     dict(minn=10, iterations=50, step=2, causal=3)),
    ("eps0_raw", ["-m1,2", "-d1", "-n50", "-k10", "-r0.01"], AR_RUN, [1],
     dict(minn=10, iterations=50, eps0=0.01, epsset=True)),
    ("epsf_1_5", ["-m1,2", "-d1", "-n50", "-k10", "-f1.5"], AR_RUN, [1],
     dict(minn=10, iterations=50, epsf=1.5)),
    ("multivariate_lorenz_2col", ["-m2,2", "-d1", "-c1,2", "-n100", "-k10"], LORENZ, [1, 2],
     dict(embed=2, minn=10, iterations=100)),
    ("length_and_exclude", ["-m1,2", "-d1", "-l500", "-x100", "-n50", "-k10"], AR_RUN, [1],
     dict(minn=10, iterations=50)),
    ("verbosity_0", ["-m1,2", "-d1", "-n50", "-k10", "-V0"], AR_RUN, [1],
     dict(minn=10, iterations=50)),
]


@pytest.mark.parametrize(
    "label,args,datafile,columns,kwargs", CASES, ids=[c[0] for c in CASES]
)
def test_compute_matches_cli(label, args, datafile, columns, kwargs):
    exclude = 100 if label == "length_and_exclude" else 0
    length_kw = 500 if label == "length_and_exclude" else None

    cli_text = run_cli(args, datafile)
    comp = len(columns)
    cli_rms, _ = parse_output(cli_text, comp)

    series = load_multi_series(datafile, columns, length=length_kw, exclude=exclude)
    result = tisean.lfo_test.compute(series, **kwargs)

    assert result.comp == comp
    assert result.length == series.shape[1]
    np.testing.assert_allclose(result.rms_error, cli_rms, **CLI_TEXT_TOL)


def test_compute_individual_matches_cli_verbosity_2():
    args = ["-m1,2", "-d1", "-n50", "-k10", "-V2"]
    cli_text = run_cli(args, AR_RUN)
    cli_rms, cli_individual = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lfo_test.compute(series, minn=10, iterations=50)

    np.testing.assert_allclose(result.rms_error, cli_rms, **CLI_TEXT_TOL)

    embed, delay, step = 2, 1, 1
    hdim = (embed - 1) * delay
    clength = 50 - step
    py_individual = result.individual[:, hdim:clength].T
    np.testing.assert_allclose(py_individual, cli_individual, **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.fce"
    subprocess.run(
        [LFO_TEST_BIN, "-m1,2", "-d1", "-n50", "-k10", "-o", str(outfile), AR_RUN],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_rms, _ = parse_output(outfile.read_text(), 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lfo_test.compute(series, minn=10, iterations=50)

    np.testing.assert_allclose(result.rms_error, cli_rms, **CLI_TEXT_TOL)


def test_compute_causal_defaults_to_step():
    args = ["-m1,2", "-d1", "-n50", "-k10", "-s2"]
    cli_text = run_cli(args, AR_RUN)
    cli_rms, _ = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lfo_test.compute(series, minn=10, iterations=50, step=2)

    np.testing.assert_allclose(result.rms_error, cli_rms, **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default comp,embed=1,2 (comp comes from
    # series.shape[0]), -d default 1, -k default 30, -s default 1, -C
    # default steps, -n default whole file (iterations=len(series)), -r/-f
    # default 1.e-3/1.2.
    series = load_multi_series(AR_RUN, [1])

    default = tisean.lfo_test.compute(series)
    explicit = tisean.lfo_test.compute(
        series, embed=2, delay=1, minn=30, step=1, causal=1, iterations=1000,
        eps0=1.0e-3, epsset=False, epsf=1.2,
    )

    assert default.comp == explicit.comp
    assert default.length == explicit.length
    np.testing.assert_array_equal(default.rms_error, explicit.rms_error)
    np.testing.assert_array_equal(default.individual, explicit.individual)


def test_compute_default_kwargs_match_cli_with_no_flags():
    cli_text = run_cli([], AR_RUN)
    cli_rms, _ = parse_output(cli_text, 1)

    series = load_multi_series(AR_RUN, [1])
    result = tisean.lfo_test.compute(series)

    assert result.length == 1000
    np.testing.assert_allclose(result.rms_error, cli_rms, **CLI_TEXT_TOL)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["5.0"] * 100) + "\n")

    result = subprocess.run(
        [LFO_TEST_BIN, str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 5.000000e+00 to 5.000000e+00" in result.stderr

    series = np.full((1, 100), 5.0)
    with pytest.raises(ValueError):
        tisean.lfo_test.compute(series)


def test_compute_rejects_too_short_for_minn_like_cli(tmp_path):
    datafile = tmp_path / "short.txt"
    data = np.sin(np.linspace(0, 10, 20))
    np.savetxt(datafile, data)

    result = subprocess.run(
        [LFO_TEST_BIN, "-m1,2", "-k30", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == ONESTEP_TOO_FEW_POINTS

    series = data.reshape(1, -1)
    with pytest.raises(ValueError):
        tisean.lfo_test.compute(series, minn=30)


def test_compute_rejects_2d_shape_mismatch():
    series = load_multi_series(AR_RUN, [1]).reshape(-1)
    with pytest.raises(ValueError):
        tisean.lfo_test.compute(series)


def test_compute_rejects_embed_zero():
    series = load_multi_series(AR_RUN, [1])
    with pytest.raises(ValueError):
        tisean.lfo_test.compute(series, embed=0)
