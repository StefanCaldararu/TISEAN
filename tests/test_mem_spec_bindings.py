import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

MEM_SPEC_TOO_MANY_POLES = 82
VARIANCE_VAR_EQ_ZERO = 23

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
MEM_SPEC_BIN = os.path.abspath("./bin/mem_spec")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [MEM_SPEC_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns (freq, spec) arrays parsed from the plain (non-comment)
    "%e %e" data rows; "#..." header/coefficient rows (only present with
    -V2/-V3) are skipped."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    data = np.array(rows)
    return data[:, 0], data[:, 1]


def load_series(path, column=1, length=None, exclude=0):
    """Replicates get_series()'s -x/-l/-c handling: skip `exclude` lines,
    keep up to `length` of the rest, then pick `column` (1-indexed)."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


CASES = [
    # label, args, datafile, column, poles, out, samplingrate, length, exclude
    ("defaults", [], AR_RUN, 1, 128, 2000, 1.0, None, 0),
    ("poles_1", ["-p1", "-P200"], AR_RUN, 1, 1, 200, 1.0, None, 0),
    ("poles_5", ["-p5", "-P200"], AR_RUN, 1, 5, 200, 1.0, None, 0),
    ("poles_20", ["-p20", "-P200"], AR_RUN, 1, 20, 200, 1.0, None, 0),
    ("out_1", ["-p5", "-P1"], AR_RUN, 1, 5, 1, 1.0, None, 0),
    ("out_100", ["-p5", "-P100"], AR_RUN, 1, 5, 100, 1.0, None, 0),
    ("samplingrate", ["-p5", "-P100", "-f2.5"], AR_RUN, 1, 5, 100, 2.5, None, 0),
    ("column_2", ["-p5", "-P100", "-c2"], HENON, 2, 5, 100, 1.0, None, 0),
    ("length", ["-p5", "-P100", "-l300"], AR_RUN, 1, 5, 100, 1.0, 300, 0),
    ("exclude", ["-p5", "-P100", "-x50"], AR_RUN, 1, 5, 100, 1.0, None, 50),
    (
        "length_and_exclude",
        ["-p5", "-P100", "-l300", "-x50"],
        AR_RUN,
        1,
        5,
        100,
        1.0,
        300,
        50,
    ),
    ("verbosity_0", ["-p5", "-P100", "-V0"], AR_RUN, 1, 5, 100, 1.0, None, 0),
    ("verbosity_3_prints_coeffs", ["-p5", "-P100", "-V3"], AR_RUN, 1, 5, 100, 1.0, None, 0),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,poles,out,samplingrate,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_spectrum_matches_cli(
    label, args, datafile, column, poles, out, samplingrate, length, exclude
):
    cli_out = run_cli(args + [datafile])
    cli_freq, cli_spec = parse_output(cli_out)

    series = load_series(datafile, column=column, length=length, exclude=exclude)
    model = tisean.mem_spec.fit(series, poles=poles)
    freq, spec = model.spectrum(out=out, samplingrate=samplingrate)

    assert model.poles == poles
    np.testing.assert_allclose(freq, cli_freq, **CLI_TEXT_TOL)
    np.testing.assert_allclose(spec, cli_spec, **CLI_TEXT_TOL)


def test_spectrum_matches_cli_dash_o_output_file_scaling(tmp_path):
    # Writing to a file (-o) divides the printed spectrum by sqrt(length)
    # on top of the stdout value - mem_spec.c's main() does this only in
    # its file-output branch (a longstanding quirk of the CLI, preserved
    # verbatim by the refactor). The Python binding always returns the
    # stdout-style (unscaled) values, so callers who want the -o scaling
    # have to apply it themselves.
    outfile = tmp_path / "out.spec"
    poles, out = 5, 100
    run_cli(["-p" + str(poles), "-P" + str(out), "-o" + str(outfile), AR_RUN])

    cli_freq, cli_spec = parse_output(outfile.read_text())

    series = load_series(AR_RUN)
    model = tisean.mem_spec.fit(series, poles=poles)
    freq, spec = model.spectrum(out=out)

    np.testing.assert_allclose(freq, cli_freq, **CLI_TEXT_TOL)
    np.testing.assert_allclose(
        spec / np.sqrt(len(series)), cli_spec, **CLI_TEXT_TOL
    )


def test_fit_exposes_sigma2_and_coef():
    series = load_series(AR_RUN)
    model = tisean.mem_spec.fit(series, poles=10)

    assert model.poles == 10
    assert model.coef.shape == (10,)
    assert isinstance(model.sigma2, float)
    assert model.sigma2 > 0


def test_fit_default_poles_matches_cli_default():
    # No -p at all: the CLI's documented default is 128.
    out_pts = 50
    cli_out = run_cli(["-P" + str(out_pts), AR_RUN])
    cli_freq, cli_spec = parse_output(cli_out)

    series = load_series(AR_RUN)
    default_model = tisean.mem_spec.fit(series)
    explicit_model = tisean.mem_spec.fit(series, poles=128)

    assert default_model.poles == explicit_model.poles == 128
    np.testing.assert_allclose(default_model.coef, explicit_model.coef)

    freq, spec = default_model.spectrum(out=out_pts)
    np.testing.assert_allclose(freq, cli_freq, **CLI_TEXT_TOL)
    np.testing.assert_allclose(spec, cli_spec, **CLI_TEXT_TOL)


def test_spectrum_default_out_and_samplingrate_match_cli_defaults():
    # No -P/-f at all: the CLI's documented defaults are out=2000, f=1.0.
    cli_out = run_cli(["-p5", AR_RUN])
    cli_freq, cli_spec = parse_output(cli_out)
    assert len(cli_freq) == 2000

    series = load_series(AR_RUN)
    model = tisean.mem_spec.fit(series, poles=5)

    default_freq, default_spec = model.spectrum()
    explicit_freq, explicit_spec = model.spectrum(out=2000, samplingrate=1.0)

    np.testing.assert_array_equal(default_freq, explicit_freq)
    np.testing.assert_array_equal(default_spec, explicit_spec)
    np.testing.assert_allclose(default_freq, cli_freq, **CLI_TEXT_TOL)
    np.testing.assert_allclose(default_spec, cli_spec, **CLI_TEXT_TOL)


def test_fit_rejects_too_many_poles_like_cli():
    poles, length = 20, 15

    result = subprocess.run(
        [MEM_SPEC_BIN, f"-p{poles}", f"-l{length}", AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == MEM_SPEC_TOO_MANY_POLES

    series = load_series(AR_RUN, length=length)
    with pytest.raises(ValueError):
        tisean.mem_spec.fit(series, poles=poles)


def test_fit_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 200) + "\n")

    result = subprocess.run(
        [MEM_SPEC_BIN, "-p5", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VARIANCE_VAR_EQ_ZERO

    series = np.full(200, 1.5)
    with pytest.raises(ValueError):
        tisean.mem_spec.fit(series, poles=5)


def test_fit_does_not_modify_input_series():
    series = load_series(AR_RUN)
    original = series.copy()

    tisean.mem_spec.fit(series, poles=10)

    np.testing.assert_array_equal(series, original)
