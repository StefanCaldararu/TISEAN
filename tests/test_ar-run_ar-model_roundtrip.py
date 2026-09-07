import subprocess
import numpy as np

# The model ar-run simulates from (tests/refs/ar-run_model.txt): variance
# (driving noise amplitude) = 1.0, AR(2) coefficients = [0.5, -0.3].
TRUE_VARIANCE = 1.0
TRUE_COEFFS = [0.5, -0.3]

# Generous but still meaningful: this is a statistical fit, not an exact
# reproduction, but both ar-run and ar-model are deterministic (fixed
# default seed), so the fitted values are exactly reproducible run to run -
# this only needs to comfortably cover normal AR-estimation error for
# l=1000, not run-to-run variance.
COEFF_ATOL = 0.1
VARIANCE_ATOL = 0.2


def run_ar_run():
    result = subprocess.run(
        ["./bin/ar-run", "-l", "1000", "./tests/refs/ar-run_model.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def run_ar_model(series_file, order):
    result = subprocess.run(
        ["./bin/ar-model", "-m1", f"-p{order}", "1000", series_file],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_fit(text, order):
    """ar-model's stdout starts with '#average forcast error= X', then
    '#individual forecast errors: X', then exactly `order` lines of the
    form '# <coefficient>', then the (non-#) residuals/data. Returns
    (avg_forecast_error, [coefficients])."""
    avg_error = None
    coeffs = []
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        if line.startswith("#average forcast error="):
            avg_error = float(line.split("=", 1)[1])
            continue
        rest = line[1:].split()
        if len(rest) == 1:
            try:
                coeffs.append(float(rest[0]))
            except ValueError:
                continue
        if len(coeffs) == order:
            break
    return avg_error, coeffs


def test_ar_model_recovers_ar_run_coefficients(tmp_path):
    series_file = tmp_path / "ar_run_series.txt"
    series_file.write_text(run_ar_run())

    fit_out = run_ar_model(str(series_file), order=len(TRUE_COEFFS))
    avg_error, fitted_coeffs = parse_fit(fit_out, order=len(TRUE_COEFFS))

    assert len(fitted_coeffs) == len(TRUE_COEFFS)
    np.testing.assert_allclose(fitted_coeffs, TRUE_COEFFS, atol=COEFF_ATOL)

    assert avg_error is not None
    np.testing.assert_allclose(avg_error, TRUE_VARIANCE, atol=VARIANCE_ATOL)
