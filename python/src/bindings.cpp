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
#include "histogram.h"
#include "polypar.h"
#include "corr.h"
#include "xcor.h"
#include "av-d2.h"
#include "mutual.h"
#include "extrema.h"
#include "xzero.h"

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

// Owns a Histogram* and exposes its fields as numpy arrays. Not copyable
// since Histogram doesn't support that; pybind11 holds it by unique_ptr.
class HistogramWrapper {
public:
  explicit HistogramWrapper(Histogram *hist) : hist_(hist) {}
  HistogramWrapper(const HistogramWrapper &) = delete;
  HistogramWrapper &operator=(const HistogramWrapper &) = delete;
  ~HistogramWrapper() { histogram_free(hist_); }

  unsigned long base() const { return hist_->base; }
  double min() const { return hist_->min; }
  double interval() const { return hist_->interval; }
  double average() const { return hist_->average; }
  double var() const { return hist_->var; }

  py::array_t<long> box() const {
    py::array_t<long> out((py::ssize_t)hist_->base);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < hist_->base; i++)
      buf(i) = hist_->box[i];
    return out;
  }

private:
  Histogram *hist_;
};

std::unique_ptr<HistogramWrapper>
histogram_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			   unsigned long base)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);
  Histogram *hist = histogram_compute(series.data(), length, base);
  if (hist == nullptr)
    throw std::invalid_argument("series must be non-empty and non-constant");

  return std::make_unique<HistogramWrapper>(hist);
}

// Owns a PolyParResult* and exposes its fields as numpy arrays. Not
// copyable since PolyParResult doesn't support that; pybind11 holds it by
// unique_ptr.
class PolyParResultWrapper {
public:
  explicit PolyParResultWrapper(PolyParResult *result) : result_(result) {}
  PolyParResultWrapper(const PolyParResultWrapper &) = delete;
  PolyParResultWrapper &operator=(const PolyParResultWrapper &) = delete;
  ~PolyParResultWrapper() { polypar_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned int order() const { return result_->order; }
  unsigned long count() const { return result_->count; }

  py::array_t<unsigned int> params() const {
    unsigned int dim = result_->dim;
    unsigned long count = result_->count;
    py::array_t<unsigned int> out({(py::ssize_t)count, (py::ssize_t)dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < count; i++)
      for (unsigned int j = 0; j < dim; j++)
	buf(i, j) = result_->params[i * dim + j];
    return out;
  }

private:
  PolyParResult *result_;
};

std::unique_ptr<PolyParResultWrapper> generate(unsigned int dim, unsigned int order)
{
  if (dim < 1)
    throw std::invalid_argument("dim must be >= 1");

  PolyParResult *result = polypar_generate(dim, order);
  return std::make_unique<PolyParResultWrapper>(result);
}

// Owns a CorrResult* and exposes its fields as numpy arrays. Not copyable
// since CorrResult doesn't support that; pybind11 holds it by unique_ptr.
class CorrResultWrapper {
public:
  explicit CorrResultWrapper(CorrResult *result) : result_(result) {}
  CorrResultWrapper(const CorrResultWrapper &) = delete;
  CorrResultWrapper &operator=(const CorrResultWrapper &) = delete;
  ~CorrResultWrapper() { corr_free(result_); }

  unsigned long length() const { return result_->length; }
  unsigned long tau() const { return result_->tau; }
  double average() const { return result_->average; }
  double stddev() const { return result_->stddev; }

  py::array_t<double> values() const {
    py::array_t<double> out((py::ssize_t)(result_->tau + 1));
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i <= result_->tau; i++)
      buf(i) = result_->values[i];
    return out;
  }

private:
  CorrResult *result_;
};

std::unique_ptr<CorrResultWrapper>
corr_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		      unsigned long tau, bool normalize)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);
  CorrResult *result = corr_compute(series.data(), length, tau, normalize ? 1 : 0);
  if (result == nullptr)
    throw std::invalid_argument("series must be non-empty and non-constant");

  return std::make_unique<CorrResultWrapper>(result);
}

// Owns an XcorResult* and exposes its fields as numpy arrays. Not copyable
// since XcorResult doesn't support that; pybind11 holds it by unique_ptr.
class XcorResultWrapper {
public:
  explicit XcorResultWrapper(XcorResult *result) : result_(result) {}
  XcorResultWrapper(const XcorResultWrapper &) = delete;
  XcorResultWrapper &operator=(const XcorResultWrapper &) = delete;
  ~XcorResultWrapper() { xcor_free(result_); }

  unsigned long length() const { return result_->length; }
  unsigned long tau() const { return result_->tau; }
  double average1() const { return result_->average1; }
  double stddev1() const { return result_->stddev1; }
  double average2() const { return result_->average2; }
  double stddev2() const { return result_->stddev2; }

  py::array_t<double> values() const {
    py::array_t<double> out((py::ssize_t)(2 * result_->tau + 1));
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < 2 * result_->tau + 1; i++)
      buf(i) = result_->values[i];
    return out;
  }

private:
  XcorResult *result_;
};

std::unique_ptr<XcorResultWrapper>
xcor_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series1,
		      py::array_t<double, py::array::c_style | py::array::forcecast> series2,
		      long tau)
{
  if (series1.ndim() != 1 || series2.ndim() != 1)
    throw std::invalid_argument("series1 and series2 must be 1D arrays");
  if (series1.shape(0) != series2.shape(0))
    throw std::invalid_argument("series1 and series2 must have the same length");

  auto length = (unsigned long)series1.shape(0);
  XcorResult *result = xcor_compute(series1.data(), series2.data(), length, tau);
  if (result == nullptr)
    throw std::invalid_argument("series1 and series2 must be non-empty and non-constant");

  return std::make_unique<XcorResultWrapper>(result);
}

// Owns an AvD2Result* and exposes its fields as numpy arrays. Not copyable
// since AvD2Result doesn't support that; pybind11 holds it by unique_ptr.
class AvD2ResultWrapper {
public:
  explicit AvD2ResultWrapper(AvD2Result *result) : result_(result) {}
  AvD2ResultWrapper(const AvD2ResultWrapper &) = delete;
  AvD2ResultWrapper &operator=(const AvD2ResultWrapper &) = delete;
  ~AvD2ResultWrapper() { av_d2_free(result_); }

  unsigned long n_points() const { return result_->n_points; }

  py::array_t<double> avg_eps() const {
    py::array_t<double> out((py::ssize_t)result_->n_points);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_points; i++)
      buf(i) = result_->avg_eps[i];
    return out;
  }

  py::array_t<double> avg_y() const {
    py::array_t<double> out((py::ssize_t)result_->n_points);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_points; i++)
      buf(i) = result_->avg_y[i];
    return out;
  }

private:
  AvD2Result *result_;
};

std::unique_ptr<AvD2ResultWrapper>
av_d2_average_binding(py::array_t<double, py::array::c_style | py::array::forcecast> eps,
		       py::array_t<double, py::array::c_style | py::array::forcecast> y,
		       int aver)
{
  if (eps.ndim() != 1 || y.ndim() != 1)
    throw std::invalid_argument("eps and y must be 1D arrays");
  if (eps.shape(0) != y.shape(0))
    throw std::invalid_argument("eps and y must have the same length");

  auto howmany = (unsigned long)eps.shape(0);
  AvD2Result *result = av_d2_average(eps.data(), y.data(), howmany, aver);
  if (result == nullptr)
    throw std::invalid_argument("aver must be >= 0");

  return std::make_unique<AvD2ResultWrapper>(result);
}

// Owns a MutualResult* and exposes its fields as numpy arrays. Not copyable
// since MutualResult doesn't support that; pybind11 holds it by
// unique_ptr.
class MutualResultWrapper {
public:
  explicit MutualResultWrapper(MutualResult *result) : result_(result) {}
  MutualResultWrapper(const MutualResultWrapper &) = delete;
  MutualResultWrapper &operator=(const MutualResultWrapper &) = delete;
  ~MutualResultWrapper() { mutual_free(result_); }

  unsigned long length() const { return result_->length; }
  long partitions() const { return result_->partitions; }
  long corrlength() const { return result_->corrlength; }

  py::array_t<double> values() const {
    py::array_t<double> out((py::ssize_t)(result_->corrlength + 1));
    auto buf = out.mutable_unchecked<1>();
    for (long i = 0; i <= result_->corrlength; i++)
      buf(i) = result_->values[i];
    return out;
  }

private:
  MutualResult *result_;
};

std::unique_ptr<MutualResultWrapper>
mutual_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			long partitions, long corrlength)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");
  if (partitions < 1)
    throw std::invalid_argument("partitions must be >= 1");
  if (corrlength < 0)
    throw std::invalid_argument("corrlength must be >= 0");

  auto length = (unsigned long)series.shape(0);
  MutualResult *result = mutual_compute(series.data(), length, partitions, corrlength);
  if (result == nullptr)
    throw std::invalid_argument("series must be non-empty and non-constant");

  return std::make_unique<MutualResultWrapper>(result);
}

// Owns an ExtremaResult* and exposes its fields as numpy arrays. Not
// copyable since ExtremaResult doesn't support that; pybind11 holds it by
// unique_ptr.
class ExtremaResultWrapper {
public:
  explicit ExtremaResultWrapper(ExtremaResult *result) : result_(result) {}
  ExtremaResultWrapper(const ExtremaResultWrapper &) = delete;
  ExtremaResultWrapper &operator=(const ExtremaResultWrapper &) = delete;
  ~ExtremaResultWrapper() { extrema_free(result_); }

  unsigned long count() const { return result_->count; }
  unsigned int dim() const { return result_->dim; }

  py::array_t<double> point() const {
    py::array_t<double> out({(py::ssize_t)result_->count, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->count; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->point[i * result_->dim + j];
    return out;
  }

  py::array_t<double> dt() const {
    py::array_t<double> out((py::ssize_t)result_->count);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->count; i++)
      buf(i) = result_->dt[i];
    return out;
  }

private:
  ExtremaResult *result_;
};

std::unique_ptr<ExtremaResultWrapper>
extrema_find_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		       unsigned int which, bool maxima, double mintime)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  ExtremaResult *result = extrema_find(rows.data(), length, dim, which,
					 maxima ? 1 : 0, mintime);
  if (result == nullptr)
    throw std::invalid_argument("which must be < series.shape[0]");

  return std::make_unique<ExtremaResultWrapper>(result);
}

// Owns an XZeroResult* and exposes its fields as numpy arrays. Not copyable
// since XZeroResult doesn't support that; pybind11 holds it by unique_ptr.
class XZeroResultWrapper {
public:
  explicit XZeroResultWrapper(XZeroResult *result) : result_(result) {}
  XZeroResultWrapper(const XZeroResultWrapper &) = delete;
  XZeroResultWrapper &operator=(const XZeroResultWrapper &) = delete;
  ~XZeroResultWrapper() { xzero_free(result_); }

  unsigned int steps() const { return result_->steps; }
  unsigned long clength() const { return result_->clength; }

  py::array_t<double> error() const {
    py::array_t<double> out((py::ssize_t)result_->steps);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < result_->steps; i++)
      buf(i) = result_->error[i];
    return out;
  }

private:
  XZeroResult *result_;
};

std::unique_ptr<XZeroResultWrapper>
xzero_forecast_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series1,
			py::array_t<double, py::array::c_style | py::array::forcecast> series2,
			unsigned int dim, unsigned int delay, py::object n_ref,
			int minn, double eps0, double epsf, unsigned int step,
			bool epsset)
{
  if (series1.ndim() != 1 || series2.ndim() != 1)
    throw std::invalid_argument("series1 and series2 must be 1D arrays");
  if (series1.shape(0) != series2.shape(0))
    throw std::invalid_argument("series1 and series2 must have the same length");
  if (dim < 1)
    throw std::invalid_argument("dim must be >= 1");

  auto length = (unsigned long)series1.shape(0);
  unsigned long resolved_ref = n_ref.is_none() ? length : n_ref.cast<unsigned long>();
  unsigned long effective_ref = resolved_ref < length ? resolved_ref : length;
  if (step > effective_ref)
    throw std::invalid_argument("step must be <= min(n_ref, len(series1))");

  XZeroResult *result = xzero_forecast(series1.data(), series2.data(), length, dim, delay,
					resolved_ref, minn, eps0, epsf, step, epsset ? 1 : 0);
  if (result == nullptr)
    throw std::invalid_argument("series1 and series2 must be non-empty and non-constant");

  return std::make_unique<XZeroResultWrapper>(result);
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

  auto histogram = m.def_submodule(
      "histogram", "Data histogram over a rescaled [0,1) range (source_c/histogram.c)");

  py::class_<HistogramWrapper>(histogram, "Histogram")
      .def_property_readonly("base", &HistogramWrapper::base)
      .def_property_readonly("min", &HistogramWrapper::min,
			      "Minimum of the raw (un-rescaled) series")
      .def_property_readonly("interval", &HistogramWrapper::interval,
			      "max - min of the raw series")
      .def_property_readonly("average", &HistogramWrapper::average)
      .def_property_readonly("var", &HistogramWrapper::var,
			      "Standard deviation of the raw series")
      .def_property_readonly("box", &HistogramWrapper::box,
			      "Bin counts, shape (base,)");

  histogram.def(
      "compute", &histogram_compute_binding, py::arg("series"), py::arg("base") = 50,
      "Bin `series` into `base` equal-width intervals over its own "
      "[min,max] range, the same way the histogram CLI does it.");

  auto polypar = m.def_submodule(
      "polypar", "Polynomial exponent enumeration (source_c/polypar.c)");

  py::class_<PolyParResultWrapper>(polypar, "PolyParResult")
      .def_property_readonly("dim", &PolyParResultWrapper::dim)
      .def_property_readonly("order", &PolyParResultWrapper::order)
      .def_property_readonly("count", &PolyParResultWrapper::count)
      .def_property_readonly("params", &PolyParResultWrapper::params,
			      "Exponent combinations, shape (count, dim)");

  polypar.def(
      "generate", &generate, py::arg("dim") = 2, py::arg("order") = 3,
      "Enumerate every combination of `dim` non-negative integer exponents "
      "that sum to at most `order`.");

  auto corr = m.def_submodule(
      "corr", "Autocorrelation estimation (source_c/corr.c)");

  py::class_<CorrResultWrapper>(corr, "CorrResult")
      .def_property_readonly("length", &CorrResultWrapper::length)
      .def_property_readonly("tau", &CorrResultWrapper::tau,
			      "Maximum lag actually computed")
      .def_property_readonly("average", &CorrResultWrapper::average,
			      "Mean of the raw (un-centered) series")
      .def_property_readonly("stddev", &CorrResultWrapper::stddev,
			      "Standard deviation of the raw series")
      .def_property_readonly("values", &CorrResultWrapper::values,
			      "Correlation values for lags 0..tau, shape (tau+1,)");

  corr.def(
      "compute", &corr_compute_binding, py::arg("series"), py::arg("tau") = 100,
      py::arg("normalize") = true,
      "Estimate the autocorrelation of `series` for lags 0..tau (tau is\n"
      "clamped to len(series)-1 if too large). If normalize is True\n"
      "(default), the series is centered by its own mean and each lag is\n"
      "divided by the variance; if False, the raw series is used\n"
      "unscaled, matching the CLI's -n flag.");

  auto xcor = m.def_submodule(
      "xcor", "Crosscorrelation estimation of two data sets (source_c/xcor.c)");

  py::class_<XcorResultWrapper>(xcor, "XcorResult")
      .def_property_readonly("length", &XcorResultWrapper::length)
      .def_property_readonly("tau", &XcorResultWrapper::tau,
			      "Maximum lag actually computed")
      .def_property_readonly("average1", &XcorResultWrapper::average1,
			      "Mean of the raw (un-centered) first series")
      .def_property_readonly("stddev1", &XcorResultWrapper::stddev1,
			      "Standard deviation of the raw first series")
      .def_property_readonly("average2", &XcorResultWrapper::average2,
			      "Mean of the raw (un-centered) second series")
      .def_property_readonly("stddev2", &XcorResultWrapper::stddev2,
			      "Standard deviation of the raw second series")
      .def_property_readonly("values", &XcorResultWrapper::values,
			      "Crosscorrelation values for lags -tau..tau, "
			      "shape (2*tau+1,)");

  xcor.def(
      "compute", &xcor_compute_binding, py::arg("series1"), py::arg("series2"),
      py::arg("tau") = 100,
      "Estimate the crosscorrelation of `series1` against `series2` for\n"
      "lags -tau..tau (tau is clamped to len(series)-1 if too large or\n"
      "negative). Both series are centered by their own mean and each lag\n"
      "is divided by both series' standard deviations, matching xcor.c's\n"
      "main().");

  auto av_d2 = m.def_submodule(
      "av_d2", "Centered moving-average smoothing of d2 program output (source_c/av-d2.c)");

  py::class_<AvD2ResultWrapper>(av_d2, "AvD2Result")
      .def_property_readonly("n_points", &AvD2ResultWrapper::n_points)
      .def_property_readonly("avg_eps", &AvD2ResultWrapper::avg_eps,
			      "Window-averaged eps values, shape (n_points,)")
      .def_property_readonly("avg_y", &AvD2ResultWrapper::avg_y,
			      "Window-averaged y values, shape (n_points,)");

  av_d2.def(
      "average", &av_d2_average_binding, py::arg("eps"), py::arg("y"), py::arg("aver") = 1,
      "Smooth `eps`/`y` pairs from one dimension-block of a d2 program\n"
      "output file with a centered moving average over a (2*aver+1)-point\n"
      "window, matching av-d2.c's main() (the CLI's -a option, default 1).\n"
      "The first and last `aver` points are dropped since they can't be\n"
      "centered within the array.");

  auto mutual = m.def_submodule(
      "mutual", "Time-delayed mutual information estimation (source_c/mutual.c)");

  py::class_<MutualResultWrapper>(mutual, "MutualResult")
      .def_property_readonly("length", &MutualResultWrapper::length)
      .def_property_readonly("partitions", &MutualResultWrapper::partitions,
			      "Number of histogram bins per dimension")
      .def_property_readonly("corrlength", &MutualResultWrapper::corrlength,
			      "Maximum lag actually computed")
      .def_property_readonly("values", &MutualResultWrapper::values,
			      "Conditional-entropy estimate of the mutual "
			      "information for lags 0..corrlength, shape "
			      "(corrlength+1,)");

  mutual.def(
      "compute", &mutual_compute_binding, py::arg("series"), py::arg("partitions") = 16,
      py::arg("corrlength") = 20,
      "Estimate the time-delayed mutual information of `series` by binning "
      "it (rescaled to its own [min,max] range) into `partitions` "
      "equal-width boxes and computing the conditional entropy against its "
      "own lag-t copy for each t in 0..corrlength (corrlength is clamped "
      "to len(series)-1 if too large), matching the mutual CLI's default "
      "-b/-D options.");

  auto extrema = m.def_submodule(
      "extrema", "Local maxima/minima detection via parabola fit (source_c/extrema.c)");

  py::class_<ExtremaResultWrapper>(extrema, "ExtremaResult")
      .def_property_readonly("count", &ExtremaResultWrapper::count)
      .def_property_readonly("dim", &ExtremaResultWrapper::dim)
      .def_property_readonly("point", &ExtremaResultWrapper::point,
			      "Interpolated series values at each extremum, "
			      "shape (count, dim)")
      .def_property_readonly("dt", &ExtremaResultWrapper::dt,
			      "Time since the previous extremum, shape (count,)");

  extrema.def(
      "find", &extrema_find_binding, py::arg("series"), py::arg("which") = 0,
      py::arg("maxima") = true, py::arg("mintime") = 0.0,
      "Find local maxima (or minima if maxima=False) of component `which`\n"
      "(0-based; the CLI's -w is 1-based and defaults to its first "
      "component, i.e. which=0 here) of `series` (shape (dim, length)) by\n"
      "fitting a parabola through each candidate triple of points, "
      "matching the extrema CLI's -w/-z/-t options. For every extremum "
      "found, every component of series is interpolated at the extremum's "
      "fractional time via the same parabola fit.");

  auto xzero = m.def_submodule(
      "xzero", "Average cross forecast error of a zeroth-order fit between two series (source_c/xzero.c)");

  py::class_<XZeroResultWrapper>(xzero, "XZeroResult")
      .def_property_readonly("steps", &XZeroResultWrapper::steps)
      .def_property_readonly("clength", &XZeroResultWrapper::clength,
			      "Number of reference points actually scanned")
      .def_property_readonly("error", &XZeroResultWrapper::error,
			      "Normalized RMS cross-forecast error for forecast "
			      "horizons 1..steps, shape (steps,)");

  xzero.def(
      "forecast", &xzero_forecast_binding, py::arg("series1"), py::arg("series2"),
      py::arg("dim") = 3, py::arg("delay") = 1, py::arg("n_ref") = py::none(),
      py::arg("minn") = 30, py::arg("eps0") = 1.e-3, py::arg("epsf") = 1.2,
      py::arg("step") = 1, py::arg("epsset") = false,
      "Estimate the average cross forecast error of a zeroth-order fit\n"
      "between `series1` and `series2` for forecast horizons 1..step,\n"
      "matching the xzero CLI's -m/-d/-n/-k/-r/-f/-s options. Both series\n"
      "are independently rescaled to their own [0,1) range before a\n"
      "box-assisted neighbor search over series1 (growing the search\n"
      "radius until at least `minn` neighbors are found for a given\n"
      "reference point). n_ref (the CLI's -n) defaults to len(series1) if\n"
      "not given (None). eps0 is the starting search radius in rescaled\n"
      "[0,1) units, unless epsset=True, in which case eps0 is interpreted\n"
      "in the same units as the raw input data and divided by the average\n"
      "of series1's/series2's own raw data ranges, matching the CLI's -r\n"
      "flag.");
}
