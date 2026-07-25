// pybind11 bindings for TISEAN's reentrant C APIs (source_c/api/*.c). More
// routines get their own wrapper class/functions here (or their own .cpp
// file) as they get the same "extract a reentrant API" treatment.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "ar-model.h"
#include "low121.h"

namespace py = pybind11;

namespace {

// Owns an ARModel* and exposes its fields as numpy arrays. Not copyable
// since ARModel doesn't support that; pybind11 holds it by unique_ptr.
class ARModelWrapper {
public:
  explicit ARModelWrapper(ARModel *model) : model_(model) {}
  ARModelWrapper(const ARModelWrapper &) = delete;
  ARModelWrapper &operator=(const ARModelWrapper &) = delete;
  ~ARModelWrapper() { ar_model_free(model_); }

  unsigned int dim() const { return model_->dim; }
  unsigned int poles() const { return model_->poles; }
  unsigned long length() const { return model_->length; }

  py::array_t<double> coeff() const {
    unsigned int dim = model_->dim, ncoeff = model_->dim * model_->poles;
    py::array_t<double> out({(py::ssize_t)dim, (py::ssize_t)ncoeff});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < dim; i++)
      for (unsigned int j = 0; j < ncoeff; j++)
	buf(i, j) = model_->coeff[i][j];
    return out;
  }

  py::array_t<double> rms_error() const {
    py::array_t<double> out((py::ssize_t)model_->dim);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < model_->dim; i++)
      buf(i) = model_->rms_error[i];
    return out;
  }

  py::array_t<double> residuals() const {
    py::array_t<double> out({(py::ssize_t)model_->dim, (py::ssize_t)model_->length});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < model_->dim; i++)
      for (unsigned long j = 0; j < model_->length; j++)
	buf(i, j) = model_->residuals[i][j];
    return out;
  }

  py::array_t<double> iterate(unsigned long ilength, unsigned long seed) const {
    double **out = ar_model_iterate(model_, ilength, seed);
    py::array_t<double> result({(py::ssize_t)ilength, (py::ssize_t)model_->dim});
    auto buf = result.mutable_unchecked<2>();
    for (unsigned long n = 0; n < ilength; n++)
      for (unsigned int d = 0; d < model_->dim; d++)
	buf(n, d) = out[n][d];
    ar_model_iterate_free(out, ilength);
    return result;
  }

private:
  ARModel *model_;
};

std::unique_ptr<ARModelWrapper>
fit(py::array_t<double, py::array::c_style | py::array::forcecast> series,
    unsigned int poles)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  ARModel *model = ar_model_fit(rows.data(), length, dim, poles);
  if (model == nullptr)
    throw std::invalid_argument("poles must be >= 1 and < series.shape[1]");

  return std::make_unique<ARModelWrapper>(model);
}

py::array_t<double>
low121_filter_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		       unsigned int iterations)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);
  if (length < 2)
    throw std::invalid_argument("series must have at least 2 points");

  py::array_t<double> out((py::ssize_t)length);
  low121_filter(series.data(), length, iterations, out.mutable_data());
  return out;
}

} // namespace

PYBIND11_MODULE(_tisean, m)
{
  m.doc() = "Python bindings for TISEAN routines";

  auto ar_model = m.def_submodule(
      "ar_model", "Multivariate AR model fitting (source_c/ar-model.c)");

  py::class_<ARModelWrapper>(ar_model, "ARModel")
      .def_property_readonly("dim", &ARModelWrapper::dim)
      .def_property_readonly("poles", &ARModelWrapper::poles)
      .def_property_readonly("length", &ARModelWrapper::length)
      .def_property_readonly("coeff", &ARModelWrapper::coeff,
			      "AR coefficients, shape (dim, dim*poles)")
      .def_property_readonly("rms_error", &ARModelWrapper::rms_error,
			      "RMS one-step forecast error per component, shape (dim,)")
      .def_property_readonly("residuals", &ARModelWrapper::residuals,
			      "One-step-ahead residual series, shape (dim, length)")
      .def("iterate", &ARModelWrapper::iterate, py::arg("ilength"),
	   py::arg("seed") = 0x44325UL,
	   "Iterate the fitted model forward `ilength` steps, returning "
	   "an (ilength, dim) array.");

  ar_model.def(
      "fit", &fit, py::arg("series"), py::arg("poles") = 1,
      "Fit a multivariate AR model to `series` (shape (dim, length)).\n\n"
      "series is expected to already be centered (zero mean per row), the\n"
      "same way the ar-model CLI centers its input before fitting.");

  auto low121 = m.def_submodule(
      "low121", "Simple [1,2,1]/4 lowpass filter (source_c/low121.c)");

  low121.def(
      "filter", &low121_filter_binding, py::arg("series"), py::arg("iterations") = 1,
      "Apply the [1,2,1]/4 lowpass filter to `series` `iterations` times, "
      "returning a new 1D array of the same length.");
}
