// pybind11 bindings for TISEAN's reentrant C APIs (source_c/api/*.c). More
// routines get their own wrapper class/functions here (or their own .cpp
// file) as they get the same "extract a reentrant API" treatment.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <limits>
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
#include "recurr.h"
#include "mem_spec.h"
#include "lyap_r.h"
#include "makenoise.h"
#include "sav_gol.h"
#include "lzo-gm.h"
#include "polynomp.h"
#include "fsle.h"
#include "false_nearest.h"
#include "pca.h"
#include "delay.h"
#include "lyap_k.h"
#include "lzo-test.h"
#include "boxcount.h"
#include "nrlazy.h"
#include "rbf.h"
#include "lfo-ar.h"
#include "lzo-run.h"
#include "lfo-run.h"
#include "lfo-test.h"
#include "nstat_z.h"
#include "ghkss.h"

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

// Owns a RecurrResult* and exposes its fields as numpy arrays. Not copyable
// since RecurrResult doesn't support that; pybind11 holds it by
// unique_ptr.
class RecurrResultWrapper {
public:
  explicit RecurrResultWrapper(RecurrResult *result) : result_(result) {}
  RecurrResultWrapper(const RecurrResultWrapper &) = delete;
  RecurrResultWrapper &operator=(const RecurrResultWrapper &) = delete;
  ~RecurrResultWrapper() { recurr_free(result_); }

  unsigned long count() const { return result_->count; }

  py::array_t<long> point() const {
    py::array_t<long> out((py::ssize_t)result_->count);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->count; i++)
      buf(i) = result_->point[i];
    return out;
  }

  py::array_t<long> neighbor() const {
    py::array_t<long> out((py::ssize_t)result_->count);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->count; i++)
      buf(i) = result_->neighbor[i];
    return out;
  }

private:
  RecurrResult *result_;
};

std::unique_ptr<RecurrResultWrapper>
recurr_find_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		     unsigned int embed, unsigned int delay, double eps,
		     bool eps_is_raw, double fraction)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  double bad_value = 0.0;
  RecurrResult *result = recurr_find(rows.data(), length, dim, embed, delay, eps,
				      eps_is_raw ? 1 : 0, fraction, &bad_value);
  if (result == nullptr)
    throw std::invalid_argument(
	"series must be non-empty and have no constant dimension (a dimension "
	"ranges from " + std::to_string(bad_value) + " to " + std::to_string(bad_value) + ")");

  return std::make_unique<RecurrResultWrapper>(result);
}

// Owns a MemSpecModel* and exposes its fields as numpy arrays. Not
// copyable since MemSpecModel doesn't support that; pybind11 holds it by
// unique_ptr.
class MemSpecModelWrapper {
public:
  explicit MemSpecModelWrapper(MemSpecModel *model) : model_(model) {}
  MemSpecModelWrapper(const MemSpecModelWrapper &) = delete;
  MemSpecModelWrapper &operator=(const MemSpecModelWrapper &) = delete;
  ~MemSpecModelWrapper() { mem_spec_free(model_); }

  unsigned long poles() const { return model_->poles; }
  double sigma2() const { return model_->sigma2; }

  py::array_t<double> coef() const {
    py::array_t<double> out((py::ssize_t)model_->poles);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < model_->poles; i++)
      buf(i) = model_->coef[i];
    return out;
  }

  std::pair<py::array_t<double>, py::array_t<double>>
  spectrum(unsigned long out, double samplingrate) const {
    py::array_t<double> freq((py::ssize_t)out), spec((py::ssize_t)out);
    mem_spec_spectrum(model_, out, samplingrate, freq.mutable_data(), spec.mutable_data());
    return {freq, spec};
  }

private:
  MemSpecModel *model_;
};

std::unique_ptr<MemSpecModelWrapper>
mem_spec_fit_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		      unsigned long poles)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);
  MemSpecModel *model = mem_spec_fit(series.data(), length, poles);
  if (model == nullptr)
    throw std::invalid_argument(
	"poles must be < len(series), and series must not be constant "
	"(zero variance)");

  return std::make_unique<MemSpecModelWrapper>(model);
}

// Owns a LyapR* and exposes its fields as numpy arrays. Not copyable since
// LyapR doesn't support that; pybind11 holds it by unique_ptr.
class LyapRWrapper {
public:
  explicit LyapRWrapper(LyapR *result) : result_(result) {}
  LyapRWrapper(const LyapRWrapper &) = delete;
  LyapRWrapper &operator=(const LyapRWrapper &) = delete;
  ~LyapRWrapper() { lyap_r_free(result_); }

  unsigned int steps() const { return result_->steps; }

  py::array_t<long> found() const {
    py::array_t<long> out((py::ssize_t)(result_->steps + 1));
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i <= result_->steps; i++)
      buf(i) = result_->found[i];
    return out;
  }

  py::array_t<double> lyap() const {
    py::array_t<double> out((py::ssize_t)(result_->steps + 1));
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i <= result_->steps; i++)
      buf(i) = result_->lyap[i];
    return out;
  }

private:
  LyapR *result_;
};

std::unique_ptr<LyapRWrapper>
lyap_r_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			unsigned int dim, unsigned int delay, unsigned int mindist,
			unsigned int steps, double eps0, bool epsset)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");
  if (dim < 1)
    throw std::invalid_argument("dim must be >= 1");

  auto length = (unsigned long)series.shape(0);
  if (length == 0)
    throw std::invalid_argument("series must be non-empty");

  unsigned long del = (unsigned long)delay * (dim - 1);
  if (del + steps >= length)
    throw std::invalid_argument(
	"series too short for the given dim/delay/steps (length must be "
	"> delay*(dim-1)+steps)");

  LyapR *result = lyap_r_compute(series.data(), length, dim, delay, mindist, steps,
				  eps0, epsset ? 1 : 0, nullptr, nullptr);
  if (result == nullptr)
    throw std::invalid_argument("series must be non-constant");

  return std::make_unique<LyapRWrapper>(result);
}

// Owns a MakeNoise* and exposes its fields as numpy arrays. Not copyable
// since MakeNoise doesn't support that; pybind11 holds it by unique_ptr.
class MakeNoiseWrapper {
public:
  explicit MakeNoiseWrapper(MakeNoise *noise) : noise_(noise) {}
  MakeNoiseWrapper(const MakeNoiseWrapper &) = delete;
  MakeNoiseWrapper &operator=(const MakeNoiseWrapper &) = delete;
  ~MakeNoiseWrapper() { makenoise_free(noise_); }

  unsigned int dim() const { return noise_->dim; }
  unsigned long length() const { return noise_->length; }

  py::array_t<double> series() const {
    py::array_t<double> out({(py::ssize_t)noise_->dim, (py::ssize_t)noise_->length});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < noise_->dim; i++)
      for (unsigned long j = 0; j < noise_->length; j++)
	buf(i, j) = noise_->series[i][j];
    return out;
  }

private:
  MakeNoise *noise_;
};

std::unique_ptr<MakeNoiseWrapper>
makenoise_add_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		       double noiselevel, bool absolute, bool gaussian, unsigned long seed)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (dim == 0 || length == 0)
    throw std::invalid_argument("series must be non-empty");

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  MakeNoise *noise = makenoise_add(rows.data(), length, dim, noiselevel,
				    absolute ? 1 : 0, gaussian ? 1 : 0, seed);
  if (noise == nullptr)
    throw std::invalid_argument(
	"absolute=False requires every row of series to have non-zero variance");

  return std::make_unique<MakeNoiseWrapper>(noise);
}

// Owns a SavGol* and exposes its fields as numpy arrays. Not copyable since
// SavGol doesn't support that; pybind11 holds it by unique_ptr.
class SavGolWrapper {
public:
  explicit SavGolWrapper(SavGol *result) : result_(result) {}
  SavGolWrapper(const SavGolWrapper &) = delete;
  SavGolWrapper &operator=(const SavGolWrapper &) = delete;
  ~SavGolWrapper() { sav_gol_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned long length() const { return result_->length; }

  py::array_t<double> data() const {
    py::array_t<double> out({(py::ssize_t)result_->dim, (py::ssize_t)result_->length});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < result_->dim; i++)
      for (unsigned long j = 0; j < result_->length; j++)
	buf(i, j) = result_->data[i][j];
    return out;
  }

private:
  SavGol *result_;
};

std::unique_ptr<SavGolWrapper>
sav_gol_filter_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			unsigned int nb, unsigned int nf, unsigned int power,
			unsigned int deriv)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  SavGol *result = sav_gol_filter(rows.data(), length, dim, nb, nf, power, deriv);
  if (result == nullptr)
    throw std::invalid_argument(
	"power must be < nb+nf+1 (the fit would be underdetermined), and "
	"deriv must be <= power");

  return std::make_unique<SavGolWrapper>(result);
}

// Owns an LzoGmResult* and exposes its fields as numpy arrays. Not copyable
// since LzoGmResult doesn't support that; pybind11 holds it by unique_ptr.
class LzoGmResultWrapper {
public:
  explicit LzoGmResultWrapper(LzoGmResult *result) : result_(result) {}
  LzoGmResultWrapper(const LzoGmResultWrapper &) = delete;
  LzoGmResultWrapper &operator=(const LzoGmResultWrapper &) = delete;
  ~LzoGmResultWrapper() { lzo_gm_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned long n_rows() const { return result_->n_rows; }

  py::array_t<double> epsilon() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->epsilon[i];
    return out;
  }

  py::array_t<double> avg_error() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->avg_error[i];
    return out;
  }

  py::array_t<double> error() const {
    py::array_t<double> out({(py::ssize_t)result_->n_rows, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->error[i * result_->dim + j];
    return out;
  }

  py::array_t<double> fraction() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->fraction[i];
    return out;
  }

  py::array_t<double> avneighbors() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->avneighbors[i];
    return out;
  }

private:
  LzoGmResult *result_;
};

std::unique_ptr<LzoGmResultWrapper>
lzo_gm_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			unsigned int embed, unsigned int delay, int step,
			py::object causal, py::object iterations,
			double eps0, bool eps0_raw, double eps1, bool eps1_raw,
			double epsf)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  unsigned long resolved_causal =
      causal.is_none() ? (step < 0 ? 0UL : (unsigned long)step) : causal.cast<unsigned long>();
  unsigned long resolved_iterations = iterations.is_none() ? length : iterations.cast<unsigned long>();

  double bad_value = 0.0;
  LzoGmResult *result = lzo_gm_compute(rows.data(), length, dim, embed, delay, step,
					resolved_causal, resolved_iterations,
					eps0, eps0_raw ? 1 : 0, eps1, eps1_raw ? 1 : 0, epsf,
					&bad_value);
  if (result == nullptr)
    throw std::invalid_argument(
	"either a dimension of series is constant (ranges from " +
	std::to_string(bad_value) + " to " + std::to_string(bad_value) +
	"), or dim/embed/series.shape[1] is 0, or step is not in "
	"[0, series.shape[1]), or iterations < step");

  return std::make_unique<LzoGmResultWrapper>(result);
}

// Owns an LfoArResult* and exposes its fields as numpy arrays. Not copyable
// since LfoArResult doesn't support that; pybind11 holds it by unique_ptr.
class LfoArResultWrapper {
public:
  explicit LfoArResultWrapper(LfoArResult *result) : result_(result) {}
  LfoArResultWrapper(const LfoArResultWrapper &) = delete;
  LfoArResultWrapper &operator=(const LfoArResultWrapper &) = delete;
  ~LfoArResultWrapper() { lfo_ar_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned long n_rows() const { return result_->n_rows; }

  py::array_t<double> epsilon() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->epsilon[i];
    return out;
  }

  py::array_t<double> avg_error() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->avg_error[i];
    return out;
  }

  py::array_t<double> error() const {
    py::array_t<double> out({(py::ssize_t)result_->n_rows, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->error[i * result_->dim + j];
    return out;
  }

  py::array_t<double> fraction() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->fraction[i];
    return out;
  }

  py::array_t<double> avneighbors() const {
    py::array_t<double> out((py::ssize_t)result_->n_rows);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_rows; i++)
      buf(i) = result_->avneighbors[i];
    return out;
  }

private:
  LfoArResult *result_;
};

std::unique_ptr<LfoArResultWrapper>
lfo_ar_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			unsigned int embed, unsigned int delay, int step,
			py::object causal, py::object iterations,
			double eps0, bool eps0_raw, double eps1, bool eps1_raw,
			double epsf)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  unsigned long resolved_causal =
      causal.is_none() ? (step < 0 ? 0UL : (unsigned long)step) : causal.cast<unsigned long>();
  unsigned long resolved_iterations = iterations.is_none() ? length : iterations.cast<unsigned long>();

  double bad_value = 0.0;
  LfoArResult *result = lfo_ar_compute(rows.data(), length, dim, embed, delay, step,
					resolved_causal, resolved_iterations,
					eps0, eps0_raw ? 1 : 0, eps1, eps1_raw ? 1 : 0, epsf,
					&bad_value);
  if (result == nullptr)
    throw std::invalid_argument(
	"either a dimension of series is constant (ranges from " +
	std::to_string(bad_value) + " to " + std::to_string(bad_value) +
	"), or dim/embed/series.shape[1] is 0, or step is not in "
	"[0, series.shape[1]), or iterations < step");

  return std::make_unique<LfoArResultWrapper>(result);
}

// Owns a PolynompResult* and exposes its fields as numpy arrays. Not
// copyable since PolynompResult doesn't support that; pybind11 holds it by
// unique_ptr.
class PolynompResultWrapper {
public:
  explicit PolynompResultWrapper(PolynompResult *result) : result_(result) {}
  PolynompResultWrapper(const PolynompResultWrapper &) = delete;
  PolynompResultWrapper &operator=(const PolynompResultWrapper &) = delete;
  ~PolynompResultWrapper() { polynomp_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned int delay() const { return result_->delay; }
  unsigned int plength() const { return result_->plength; }
  double fce_insample() const { return result_->fce_insample; }
  bool has_outsample() const { return result_->has_outsample != 0; }
  double fce_outsample() const { return result_->fce_outsample; }
  unsigned long step() const { return result_->step; }

  py::array_t<double> param() const {
    py::array_t<double> out((py::ssize_t)result_->plength);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < result_->plength; i++)
      buf(i) = result_->param[i];
    return out;
  }

  py::array_t<double> forecast() const {
    py::array_t<double> out((py::ssize_t)result_->step);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->step; i++)
      buf(i) = result_->forecast[i];
    return out;
  }

private:
  PolynompResult *result_;
};

std::unique_ptr<PolynompResultWrapper>
polynomp_fit_binding(
    py::array_t<double, py::array::c_style | py::array::forcecast> series,
    py::array_t<unsigned int, py::array::c_style | py::array::forcecast> order,
    unsigned int delay, unsigned long insample, unsigned long step)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");
  if (order.ndim() != 2)
    throw std::invalid_argument("order must be a 2D array of shape (plength, dim)");

  auto length = (unsigned long)series.shape(0);
  auto plength = (unsigned int)order.shape(0);
  auto dim = (unsigned int)order.shape(1);

  if (plength < 1)
    throw std::invalid_argument("order must have at least one row (plength >= 1)");
  if (dim < 1)
    throw std::invalid_argument("order must have at least one column (dim >= 1)");
  if (length <= (unsigned long)(dim - 1) * delay)
    throw std::invalid_argument(
	"series is too short for dim/delay: length must be > (dim - 1) * delay");

  PolynompError error;
  PolynompResult *result = polynomp_fit(series.data(), length, order.data(),
					 plength, dim, delay, insample, step,
					 &error);
  if (result == nullptr) {
    if (error == POLYNOMP_ERR_ZERO_VARIANCE)
      throw std::invalid_argument("series has zero variance");
    throw std::invalid_argument("normal-equations matrix is singular");
  }

  return std::make_unique<PolynompResultWrapper>(result);
}

// Owns an FSLEResult* and exposes its fields as numpy arrays. Not copyable
// since FSLEResult doesn't support that; pybind11 holds it by unique_ptr.
class FSLEResultWrapper {
public:
  explicit FSLEResultWrapper(FSLEResult *result) : result_(result) {}
  FSLEResultWrapper(const FSLEResultWrapper &) = delete;
  FSLEResultWrapper &operator=(const FSLEResultWrapper &) = delete;
  ~FSLEResultWrapper() { fsle_free(result_); }

  unsigned long n() const { return result_->n; }

  py::array_t<double> eps() const {
    py::array_t<double> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->eps[i];
    return out;
  }

  py::array_t<double> lyapunov() const {
    py::array_t<double> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->lyapunov[i];
    return out;
  }

  py::array_t<long> count() const {
    py::array_t<long> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->count[i];
    return out;
  }

private:
  FSLEResult *result_;
};

std::unique_ptr<FSLEResultWrapper>
fsle_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		      unsigned int dim, unsigned int delay, unsigned int mindist,
		      double eps0, bool epsset)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");
  if (dim < 1)
    throw std::invalid_argument("dim must be >= 1");

  auto length = (unsigned long)series.shape(0);
  if (length == 0)
    throw std::invalid_argument("series must be non-empty");

  unsigned long del = (unsigned long)delay * (dim - 1);
  if (del + 1 + (unsigned long)mindist > length)
    throw std::invalid_argument(
	"series too short for the given dim/delay/mindist (length must be "
	"> delay*(dim-1)+mindist)");

  FSLEError error;
  FSLEResult *result = fsle_compute(series.data(), length, dim, delay, mindist,
				     eps0, epsset ? 1 : 0, &error);
  if (result == nullptr) {
    if (error == FSLE_ERR_ZERO_VARIANCE)
      throw std::invalid_argument("series has zero variance");
    if (error == FSLE_ERR_ZERO_INTERVAL)
      throw std::invalid_argument("series is constant (zero range)");
    throw std::invalid_argument(
	"starting epsilon is too large relative to the data's own scale");
  }

  return std::make_unique<FSLEResultWrapper>(result);
}

// Owns a FalseNearest* and exposes its fields as numpy arrays. Not copyable
// since FalseNearest doesn't support that; pybind11 holds it by unique_ptr.
class FalseNearestWrapper {
public:
  explicit FalseNearestWrapper(FalseNearest *result) : result_(result) {}
  FalseNearestWrapper(const FalseNearestWrapper &) = delete;
  FalseNearestWrapper &operator=(const FalseNearestWrapper &) = delete;
  ~FalseNearestWrapper() { false_nearest_free(result_); }

  unsigned long n() const { return result_->n; }

  py::array_t<unsigned int> dimension() const {
    py::array_t<unsigned int> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->dimension[i];
    return out;
  }

  py::array_t<double> fraction() const {
    py::array_t<double> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->fraction[i];
    return out;
  }

  py::array_t<double> avg_eps() const {
    py::array_t<double> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->avg_eps[i];
    return out;
  }

  py::array_t<double> sigma_eps() const {
    py::array_t<double> out((py::ssize_t)result_->n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n; i++)
      buf(i) = result_->sigma_eps[i];
    return out;
  }

private:
  FalseNearest *result_;
};

std::unique_ptr<FalseNearestWrapper>
false_nearest_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			       unsigned int minemb, unsigned int maxemb,
			       unsigned int delay, unsigned long theiler,
			       double rt, double eps0)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (comp, length)");

  auto comp = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (comp < 1)
    throw std::invalid_argument("series must have at least one component (shape[0] >= 1)");
  if (length < 1)
    throw std::invalid_argument("series must be non-empty");
  if (minemb < 1)
    throw std::invalid_argument("minemb must be >= 1");

  if (minemb <= maxemb) {
    unsigned long minlen = (unsigned long)(maxemb + 1) * delay;
    if (length <= minlen)
      throw std::invalid_argument(
	  "series too short for the given maxemb/delay (length must be "
	  "> (maxemb+1)*delay)");
  }

  std::vector<double *> rows(comp);
  for (unsigned int i = 0; i < comp; i++)
    rows[i] = series.mutable_data(i, 0);

  FalseNearestError error;
  FalseNearest *result = false_nearest_compute(rows.data(), length, comp, delay,
						minemb, maxemb, theiler, rt, eps0,
						&error);
  if (result == nullptr) {
    if (error == FALSE_NEAREST_ERR_ZERO_INTERVAL)
      throw std::invalid_argument("series is constant (zero range) for some component");
    if (error == FALSE_NEAREST_ERR_ZERO_VARIANCE)
      throw std::invalid_argument("series has zero variance for some component");
    throw std::invalid_argument(
	"not enough neighbor points found for some embedding dimension");
  }

  return std::make_unique<FalseNearestWrapper>(result);
}

// Owns a PCA* and exposes its fields as numpy arrays. Not copyable since
// PCA doesn't support that; pybind11 holds it by unique_ptr.
class PCAWrapper {
public:
  explicit PCAWrapper(PCA *pca) : pca_(pca) {}
  PCAWrapper(const PCAWrapper &) = delete;
  PCAWrapper &operator=(const PCAWrapper &) = delete;
  ~PCAWrapper() { pca_free(pca_); }

  unsigned int dimemb() const { return pca_->dimemb; }

  py::array_t<double> eigenvalues() const {
    py::array_t<double> out((py::ssize_t)pca_->dimemb);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < pca_->dimemb; i++)
      buf(i) = pca_->eigenvalues[i];
    return out;
  }

  py::array_t<double> eigenvectors() const {
    py::array_t<double> out({(py::ssize_t)pca_->dimemb, (py::ssize_t)pca_->dimemb});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < pca_->dimemb; i++)
      for (unsigned int j = 0; j < pca_->dimemb; j++)
	buf(i, j) = pca_->eigenvectors[i][j];
    return out;
  }

private:
  PCA *pca_;
};

std::unique_ptr<PCAWrapper>
pca_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		     unsigned int emb, unsigned int delay)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (dim < 1)
    throw std::invalid_argument("series must have at least one row (shape[0] >= 1)");
  if (emb < 1)
    throw std::invalid_argument("emb must be >= 1");

  unsigned long minlen = (unsigned long)(emb - 1) * delay;
  if (length <= minlen)
    throw std::invalid_argument(
	"series too short for the given emb/delay (length must be > "
	"(emb-1)*delay)");

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  PCA *pca = pca_compute(rows.data(), length, dim, emb, delay);
  if (pca == nullptr)
    throw std::invalid_argument("PCA eigenvalue solver failed to converge");

  return std::make_unique<PCAWrapper>(pca);
}

// Owns a DelayResult* and exposes its fields as numpy arrays. Not copyable
// since DelayResult doesn't support that; pybind11 holds it by unique_ptr.
class DelayResultWrapper {
public:
  explicit DelayResultWrapper(DelayResult *result) : result_(result) {}
  DelayResultWrapper(const DelayResultWrapper &) = delete;
  DelayResultWrapper &operator=(const DelayResultWrapper &) = delete;
  ~DelayResultWrapper() { delay_free(result_); }

  unsigned int alldim() const { return result_->alldim; }
  unsigned long n_vectors() const { return result_->n_vectors; }

  py::array_t<double> vectors() const {
    py::array_t<double> out({(py::ssize_t)result_->n_vectors, (py::ssize_t)result_->alldim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->n_vectors; i++)
      for (unsigned int j = 0; j < result_->alldim; j++)
	buf(i, j) = result_->vectors[i * result_->alldim + j];
    return out;
  }

private:
  DelayResult *result_;
};

std::unique_ptr<DelayResultWrapper>
delay_embed_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		     unsigned int embdim, py::object format, unsigned int delay,
		     py::object multidelay)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (indim, length)");

  auto indim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (indim == 0)
    throw std::invalid_argument("series must have at least one row (shape[0] >= 1)");

  std::vector<double *> rows(indim);
  for (unsigned int i = 0; i < indim; i++)
    rows[i] = series.mutable_data(i, 0);

  std::vector<unsigned int> fmt(indim);
  if (format.is_none()) {
    if (embdim % indim != 0)
      throw std::invalid_argument(
	  "embdim is not a multiple of series.shape[0]; pass an explicit "
	  "format array instead");
    unsigned int per = embdim / indim;
    for (unsigned int i = 0; i < indim; i++)
      fmt[i] = per;
  } else {
    auto fmt_arr =
	format.cast<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>>();
    if (fmt_arr.ndim() != 1 || (unsigned int)fmt_arr.shape(0) != indim)
      throw std::invalid_argument("format must be a 1D array of length series.shape[0]");
    for (unsigned int i = 0; i < indim; i++) {
      fmt[i] = fmt_arr.data()[i];
      if (fmt[i] < 1)
	throw std::invalid_argument("every entry of format must be >= 1");
    }
  }

  unsigned int alldim = 0;
  for (unsigned int i = 0; i < indim; i++)
    alldim += fmt[i];

  std::vector<unsigned int> delays(alldim);
  unsigned int rundel = 0;
  if (multidelay.is_none()) {
    if (delay < 1)
      throw std::invalid_argument("delay must be >= 1");
    for (unsigned int i = 0; i < indim; i++) {
      unsigned int delsum = 0;
      delays[rundel++] = delsum;
      for (unsigned int j = 1; j < fmt[i]; j++) {
	delsum += delay;
	delays[rundel++] = delsum;
      }
    }
  } else {
    auto md =
	multidelay.cast<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>>();
    if (md.ndim() != 1 || (unsigned int)md.shape(0) != alldim - indim)
      throw std::invalid_argument(
	  "multidelay must be a 1D array of length sum(format) - series.shape[0]");
    unsigned int runmdel = 0;
    for (unsigned int i = 0; i < indim; i++) {
      unsigned int delsum = 0;
      delays[rundel++] = delsum;
      for (unsigned int j = 1; j < fmt[i]; j++) {
	delsum += md.data()[runmdel++];
	delays[rundel++] = delsum;
      }
    }
  }

  DelayResult *result = delay_compute(rows.data(), length, indim, fmt.data(), delays.data());
  if (result == nullptr)
    throw std::invalid_argument("format must sum to at least 1");

  return std::make_unique<DelayResultWrapper>(result);
}

// Owns a LyapK* and exposes its fields as numpy arrays. Not copyable since
// LyapK doesn't support that; pybind11 holds it by unique_ptr.
class LyapKWrapper {
public:
  explicit LyapKWrapper(LyapK *result) : result_(result) {}
  LyapKWrapper(const LyapKWrapper &) = delete;
  LyapKWrapper &operator=(const LyapKWrapper &) = delete;
  ~LyapKWrapper() { lyap_k_free(result_); }

  unsigned int epscount() const { return result_->epscount; }
  unsigned int mindim() const { return result_->mindim; }
  unsigned int maxdim() const { return result_->maxdim; }
  unsigned int maxiter() const { return result_->maxiter; }

  py::array_t<double> epsilon() const {
    py::array_t<double> out((py::ssize_t)result_->epscount);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int e = 0; e < result_->epscount; e++)
      buf(e) = result_->epsilon[e];
    return out;
  }

  py::array_t<long> count() const {
    unsigned int ndim = result_->maxdim - result_->mindim + 1;
    py::array_t<long> out({(py::ssize_t)result_->epscount, (py::ssize_t)ndim,
			    (py::ssize_t)(result_->maxiter + 1)});
    auto buf = out.mutable_unchecked<3>();
    for (unsigned int e = 0; e < result_->epscount; e++)
      for (unsigned int d = 0; d < ndim; d++)
	for (unsigned int j = 0; j <= result_->maxiter; j++)
	  buf(e, d, j) = result_->count[e][d][j];
    return out;
  }

  py::array_t<double> lyap() const {
    unsigned int ndim = result_->maxdim - result_->mindim + 1;
    py::array_t<double> out({(py::ssize_t)result_->epscount, (py::ssize_t)ndim,
			      (py::ssize_t)(result_->maxiter + 1)});
    auto buf = out.mutable_unchecked<3>();
    for (unsigned int e = 0; e < result_->epscount; e++)
      for (unsigned int d = 0; d < ndim; d++)
	for (unsigned int j = 0; j <= result_->maxiter; j++)
	  buf(e, d, j) = result_->lyap[e][d][j];
    return out;
  }

private:
  LyapK *result_;
};

std::unique_ptr<LyapKWrapper>
lyap_k_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			unsigned int mindim, unsigned int maxdim, unsigned int delay,
			double epsmin, double epsmax, bool eps0set, bool eps1set,
			unsigned int epscount, unsigned long reference,
			unsigned int maxiter, unsigned int window)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);
  if (length == 0)
    throw std::invalid_argument("series must be non-empty");

  /* Mirrors the clamping lyap_k_compute() itself applies to mindim/maxdim,
     but computed with the *actual* (post-clamp) maxdim, unlike the CLI's
     own too-few-points check (which runs before that clamp and so can
     under-count the required length when maxdim < 2 or mindim > maxdim -
     see lyap_k.h). Using the clamped value here is what actually keeps the
     box-building/neighbor-search reads in bounds. */
  unsigned int cmindim = mindim < 2 ? 2 : mindim;
  unsigned int cmaxdim = maxdim < 2 ? 2 : maxdim;
  if (cmindim > cmaxdim)
    cmaxdim = cmindim;
  unsigned long need = (unsigned long)maxiter + (unsigned long)(cmaxdim - 1) * delay;
  if (need >= length)
    throw std::invalid_argument(
	"series too short for the given mindim/maxdim/delay/maxiter (length "
	"must be > maxiter+(maxdim-1)*delay once mindim/maxdim are clamped "
	"to >= 2)");

  LyapKError error;
  LyapK *result = lyap_k_compute(series.data(), length, mindim, maxdim, delay,
				  epsmin, epsmax, eps0set ? 1 : 0, eps1set ? 1 : 0,
				  epscount, reference, maxiter, window,
				  nullptr, nullptr, &error);
  if (result == nullptr) {
    if (error == LYAP_K_ERR_ZERO_INTERVAL)
      throw std::invalid_argument("series is constant (zero range)");
    throw std::invalid_argument(
	"series too short for the given mindim/maxdim/delay/maxiter");
  }

  return std::make_unique<LyapKWrapper>(result);
}

// Owns a LzoTest* and exposes its fields as numpy arrays. Not copyable since
// LzoTest doesn't support that; pybind11 holds it by unique_ptr.
class LzoTestWrapper {
public:
  explicit LzoTestWrapper(LzoTest *result) : result_(result) {}
  LzoTestWrapper(const LzoTestWrapper &) = delete;
  LzoTestWrapper &operator=(const LzoTestWrapper &) = delete;
  ~LzoTestWrapper() { lzo_test_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned long step() const { return result_->step; }
  unsigned long n_ref() const { return result_->n_ref; }

  py::array_t<double> error() const {
    py::array_t<double> out({(py::ssize_t)result_->step, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->step; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->error[i * result_->dim + j];
    return out;
  }

  py::array_t<double> diffs() const {
    py::array_t<double> out({(py::ssize_t)result_->n_ref, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->n_ref; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->diffs[i * result_->dim + j];
    return out;
  }

private:
  LzoTest *result_;
};

std::unique_ptr<LzoTestWrapper>
lzo_test_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			  unsigned int embed, unsigned int delay, unsigned int minn,
			  unsigned long step, unsigned long refstep, py::object causal,
			  py::object n_ref, double eps0, bool epsset, double epsf)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (dim < 1)
    throw std::invalid_argument("series must have at least one component (shape[0] >= 1)");
  if (length < 1)
    throw std::invalid_argument("series must be non-empty");
  if (embed < 1)
    throw std::invalid_argument("embed must be >= 1");
  if (refstep < 1)
    throw std::invalid_argument("refstep must be >= 1");
  if (step >= length)
    throw std::invalid_argument("step must be < series.shape[1]");

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  unsigned long resolved_causal = causal.is_none() ? step : causal.cast<unsigned long>();
  char n_ref_set = n_ref.is_none() ? 0 : 1;
  unsigned long resolved_n_ref = n_ref.is_none() ? 0 : n_ref.cast<unsigned long>();

  LzoTestError error;
  LzoTest *result = lzo_test_compute(rows.data(), length, dim, embed, delay, minn,
				      step, refstep, resolved_causal, resolved_n_ref,
				      n_ref_set, eps0, epsset ? 1 : 0, epsf, &error);
  if (result == nullptr) {
    if (error == LZO_TEST_ERR_ZERO_INTERVAL)
      throw std::invalid_argument("series is constant (zero range) for some component");
    throw std::invalid_argument("series has zero variance for some component");
  }

  return std::make_unique<LzoTestWrapper>(result);
}

// Owns a BoxCount* and exposes its fields as numpy arrays. Not copyable
// since BoxCount doesn't support that; pybind11 holds it by unique_ptr.
class BoxCountWrapper {
public:
  explicit BoxCountWrapper(BoxCount *result) : result_(result) {}
  BoxCountWrapper(const BoxCountWrapper &) = delete;
  BoxCountWrapper &operator=(const BoxCountWrapper &) = delete;
  ~BoxCountWrapper() { boxcount_free(result_); }

  unsigned int dimension() const { return result_->dimension; }
  unsigned int maxembed() const { return result_->maxembed; }
  unsigned long epscount() const { return result_->epscount; }

  py::array_t<double> eps() const {
    py::array_t<double> out((py::ssize_t)result_->epscount);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long k = 0; k < result_->epscount; k++)
      buf(k) = result_->eps[k];
    return out;
  }

  py::array_t<double> entropy() const {
    unsigned long epscount = result_->epscount;
    unsigned long n = (unsigned long)result_->dimension * result_->maxembed;
    py::array_t<double> out({(py::ssize_t)epscount, (py::ssize_t)n});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long k = 0; k < epscount; k++)
      for (unsigned long i = 0; i < n; i++)
	buf(k, i) = result_->entropy[k][i];
    return out;
  }

  py::array_t<unsigned int> which_component() const {
    unsigned long n = (unsigned long)result_->dimension * result_->maxembed;
    py::array_t<unsigned int> out((py::ssize_t)n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < n; i++)
      buf(i) = result_->which_component[i];
    return out;
  }

  py::array_t<unsigned int> which_embed() const {
    unsigned long n = (unsigned long)result_->dimension * result_->maxembed;
    py::array_t<unsigned int> out((py::ssize_t)n);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < n; i++)
      buf(i) = result_->which_embed[i];
    return out;
  }

private:
  BoxCount *result_;
};

std::unique_ptr<BoxCountWrapper>
boxcount_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			  unsigned int maxembed, unsigned int delay, double q,
			  double epsmin, bool epsmin_absolute, double epsmax,
			  bool epsmax_absolute, unsigned int epscount)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dimension, length)");

  auto dimension = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (dimension < 1)
    throw std::invalid_argument("series must have at least one component (shape[0] >= 1)");
  if (length < 1)
    throw std::invalid_argument("series must be non-empty");
  if (maxembed < 1)
    throw std::invalid_argument("maxembed must be >= 1");
  if (delay < 1)
    throw std::invalid_argument("delay must be >= 1");

  unsigned long minlen = (unsigned long)(maxembed - 1) * delay;
  if (length <= minlen)
    throw std::invalid_argument(
	"series too short for the given maxembed/delay (length must be "
	"> (maxembed-1)*delay)");

  std::vector<double *> rows(dimension);
  for (unsigned int i = 0; i < dimension; i++)
    rows[i] = series.mutable_data(i, 0);

  BoxCountError error;
  BoxCount *result = boxcount_compute(rows.data(), length, dimension, maxembed,
				       delay, q, epsmin, epsmin_absolute ? 1 : 0,
				       epsmax, epsmax_absolute ? 1 : 0, epscount,
				       &error);
  if (result == nullptr)
    throw std::invalid_argument("series is constant (zero range) for some component");

  return std::make_unique<BoxCountWrapper>(result);
}

// Owns an NRLazyResult* and exposes its fields as numpy arrays. Not
// copyable since NRLazyResult doesn't support that; pybind11 holds it by
// unique_ptr.
class NRLazyResultWrapper {
public:
  explicit NRLazyResultWrapper(NRLazyResult *result) : result_(result) {}
  NRLazyResultWrapper(const NRLazyResultWrapper &) = delete;
  NRLazyResultWrapper &operator=(const NRLazyResultWrapper &) = delete;
  ~NRLazyResultWrapper() { nrlazy_free(result_); }

  unsigned int comp() const { return result_->comp; }
  unsigned long length() const { return result_->length; }

  py::array_t<double> series() const {
    py::array_t<double> out({(py::ssize_t)result_->comp, (py::ssize_t)result_->length});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < result_->comp; i++)
      for (unsigned long n = 0; n < result_->length; n++)
	buf(i, n) = result_->series[i][n];
    return out;
  }

  py::array_t<unsigned int> neighbors() const {
    py::array_t<unsigned int> out((py::ssize_t)result_->length);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long n = 0; n < result_->length; n++)
      buf(n) = result_->neighbors[n];
    return out;
  }

private:
  NRLazyResult *result_;
};

std::unique_ptr<NRLazyResultWrapper>
nrlazy_correct_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			unsigned int embed, unsigned int delay, unsigned int iterations,
			py::object r, py::object v)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (comp, length)");

  auto comp = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (comp < 1)
    throw std::invalid_argument("series must have at least one component (shape[0] >= 1)");
  if (embed < 1)
    throw std::invalid_argument("embed must be >= 1");
  if (delay < 1)
    throw std::invalid_argument("delay must be >= 1");
  if (iterations < 1)
    throw std::invalid_argument("iterations must be >= 1");

  unsigned long minlen = (unsigned long)(embed - 1) * delay;
  if (length <= minlen)
    throw std::invalid_argument(
	"series too short for the given embed/delay (length must be > "
	"(embed-1)*delay)");

  std::vector<double *> rows(comp);
  for (unsigned int i = 0; i < comp; i++)
    rows[i] = series.mutable_data(i, 0);

  double eps_r = r.is_none() ? std::numeric_limits<double>::quiet_NaN() : r.cast<double>();
  double eps_v = v.is_none() ? std::numeric_limits<double>::quiet_NaN() : v.cast<double>();

  double bad_value = 0.0;
  NRLazyResult *result = nrlazy_correct(rows.data(), length, comp, embed, delay, iterations,
					  eps_r, eps_v, &bad_value, nullptr, nullptr);
  if (result == nullptr)
    throw std::invalid_argument(
	"series is constant (ranges from " + std::to_string(bad_value) + " to " +
	std::to_string(bad_value) + ") for some component");

  return std::make_unique<NRLazyResultWrapper>(result);
}

// Owns an RBFResult* and exposes its fields as numpy arrays. Not copyable
// since RBFResult doesn't support that; pybind11 holds it by unique_ptr.
class RBFResultWrapper {
public:
  explicit RBFResultWrapper(RBFResult *result) : result_(result) {}
  RBFResultWrapper(const RBFResultWrapper &) = delete;
  RBFResultWrapper &operator=(const RBFResultWrapper &) = delete;
  ~RBFResultWrapper() { rbf_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned int delay() const { return result_->delay; }
  unsigned int centers() const { return result_->centers; }
  unsigned long step() const { return result_->step; }
  unsigned long insample() const { return result_->insample; }
  unsigned long length() const { return result_->length; }
  double variance() const { return result_->variance; }
  double insample_error() const { return result_->insample_error; }
  bool has_outsample_error() const { return result_->has_outsample_error != 0; }
  double outsample_error() const { return result_->outsample_error; }
  unsigned long cast_length() const { return result_->cast_length; }

  py::array_t<double> center() const {
    py::array_t<double> out({(py::ssize_t)result_->centers, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < result_->centers; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->center[i][j];
    return out;
  }

  py::array_t<double> coefs() const {
    py::array_t<double> out((py::ssize_t)(result_->centers + 1));
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i <= result_->centers; i++)
      buf(i) = result_->coefs[i];
    return out;
  }

  py::array_t<double> cast() const {
    py::array_t<double> out((py::ssize_t)result_->cast_length);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->cast_length; i++)
      buf(i) = result_->cast[i];
    return out;
  }

private:
  RBFResult *result_;
};

std::unique_ptr<RBFResultWrapper>
rbf_fit_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		 unsigned int dim, unsigned int delay, unsigned int centers,
		 bool drift, unsigned long step, unsigned long insample,
		 unsigned long cast_length)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);

  if (dim < 1)
    throw std::invalid_argument("dim must be >= 1");
  if (length <= (unsigned long)(dim - 1) * delay)
    throw std::invalid_argument(
	"series is too short for dim/delay: length must be > (dim - 1) * delay");

  unsigned int effective_centers = centers > length ? (unsigned int)length : centers;
  if (effective_centers < 2)
    throw std::invalid_argument(
	"not enough centers: min(centers, series length) must be >= 2");

  unsigned long effective_insample = insample > length ? length : insample;
  if (effective_insample < step)
    throw std::invalid_argument(
	"insample (after clamping to the series length) must be >= step");

  RBFError error;
  RBFResult *result = rbf_fit(series.data(), length, dim, delay, centers,
			       drift ? 1 : 0, step, insample, cast_length, &error);
  if (result == nullptr) {
    if (error == RBF_ERR_ZERO_VARIANCE)
      throw std::invalid_argument("series has zero variance");
    throw std::invalid_argument("normal-equations matrix is singular");
  }

  return std::make_unique<RBFResultWrapper>(result);
}

// Owns an LzoRun* and exposes its fields as numpy arrays. Not copyable
// since LzoRun doesn't support that; pybind11 holds it by unique_ptr.
class LzoRunWrapper {
public:
  explicit LzoRunWrapper(LzoRun *result) : result_(result) {}
  LzoRunWrapper(const LzoRunWrapper &) = delete;
  LzoRunWrapper &operator=(const LzoRunWrapper &) = delete;
  ~LzoRunWrapper() { lzo_run_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned long length() const { return result_->length; }

  py::array_t<double> series() const {
    py::array_t<double> out({(py::ssize_t)result_->length, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->length; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->series[i * result_->dim + j];
    return out;
  }

private:
  LzoRun *result_;
};

std::unique_ptr<LzoRunWrapper>
lzo_run_forecast_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			   unsigned int embed, unsigned int delay, unsigned int minn,
			   bool fix_neighbors, unsigned long flength, double eps0,
			   bool epsset, double epsf, py::object noise_pct,
			   unsigned long seed)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (dim < 1)
    throw std::invalid_argument("series must have at least one component (shape[0] >= 1)");
  if (embed < 1)
    throw std::invalid_argument("embed must be >= 1");

  unsigned long minlen = (unsigned long)(embed - 1) * delay;
  if (length <= minlen)
    throw std::invalid_argument(
	"series too short for the given embed/delay (length must be > "
	"(embed-1)*delay)");

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  double noise_value = noise_pct.is_none() ? 0.0 : noise_pct.cast<double>();
  char setnoise = (!noise_pct.is_none() && noise_value > 0.0) ? 1 : 0;

  LzoRunError error;
  LzoRun *result = lzo_run_forecast(rows.data(), length, dim, embed, delay, minn,
				     fix_neighbors ? 1 : 0, flength, eps0,
				     epsset ? 1 : 0, epsf, noise_value, setnoise,
				     seed, &error);
  if (result == nullptr) {
    if (error == LZO_RUN_ERR_ZERO_INTERVAL)
      throw std::invalid_argument("series is constant (zero range) for some component");
    throw std::invalid_argument("series has zero variance for some component");
  }

  return std::make_unique<LzoRunWrapper>(result);
}

// Owns an LfoRun* and exposes its fields as numpy arrays. Not copyable
// since LfoRun doesn't support that; pybind11 holds it by unique_ptr.
class LfoRunWrapper {
public:
  explicit LfoRunWrapper(LfoRun *result) : result_(result) {}
  LfoRunWrapper(const LfoRunWrapper &) = delete;
  LfoRunWrapper &operator=(const LfoRunWrapper &) = delete;
  ~LfoRunWrapper() { lfo_run_free(result_); }

  unsigned int dim() const { return result_->dim; }
  unsigned long length() const { return result_->length; }

  py::array_t<double> series() const {
    py::array_t<double> out({(py::ssize_t)result_->length, (py::ssize_t)result_->dim});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned long i = 0; i < result_->length; i++)
      for (unsigned int j = 0; j < result_->dim; j++)
	buf(i, j) = result_->series[i * result_->dim + j];
    return out;
  }

private:
  LfoRun *result_;
};

std::unique_ptr<LfoRunWrapper>
lfo_run_forecast_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			   unsigned int embed, unsigned int delay, unsigned int minn,
			   bool zeroth_order, unsigned long flength, double eps0,
			   bool epsset, double epsf)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (dim, length)");

  auto dim = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);
  if (dim < 1)
    throw std::invalid_argument("series must have at least one component (shape[0] >= 1)");
  if (embed < 1)
    throw std::invalid_argument("embed must be >= 1");

  unsigned long minlen = (unsigned long)(embed - 1) * delay;
  if (length <= minlen)
    throw std::invalid_argument(
	"series too short for the given embed/delay (length must be > "
	"(embed-1)*delay)");

  std::vector<double *> rows(dim);
  for (unsigned int i = 0; i < dim; i++)
    rows[i] = series.mutable_data(i, 0);

  LfoRunError error;
  LfoRun *result = lfo_run_forecast(rows.data(), length, dim, embed, delay, minn,
				     zeroth_order ? 1 : 0, flength, eps0,
				     epsset ? 1 : 0, epsf, &error);
  if (result == nullptr)
    throw std::invalid_argument("series is constant (zero range) for some component");
  if (error == LFO_RUN_ERR_ESCAPED_REGION) {
    lfo_run_free(result);
    throw std::invalid_argument(
	"forecast escaped the data region before completing flength iterations");
  }

  return std::make_unique<LfoRunWrapper>(result);
}

// Owns an LfoTest* and exposes its fields as numpy arrays. Not copyable
// since LfoTest doesn't support that; pybind11 holds it by unique_ptr.
class LfoTestWrapper {
public:
  explicit LfoTestWrapper(LfoTest *result) : result_(result) {}
  LfoTestWrapper(const LfoTestWrapper &) = delete;
  LfoTestWrapper &operator=(const LfoTestWrapper &) = delete;
  ~LfoTestWrapper() { lfo_test_free(result_); }

  unsigned int comp() const { return result_->comp; }
  unsigned long length() const { return result_->length; }

  py::array_t<double> rms_error() const {
    py::array_t<double> out((py::ssize_t)result_->comp);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < result_->comp; i++)
      buf(i) = result_->rms_error[i];
    return out;
  }

  py::array_t<double> individual() const {
    py::array_t<double> out({(py::ssize_t)result_->comp, (py::ssize_t)result_->length});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int i = 0; i < result_->comp; i++)
      for (unsigned long j = 0; j < result_->length; j++)
	buf(i, j) = result_->individual[i * result_->length + j];
    return out;
  }

private:
  LfoTest *result_;
};

std::unique_ptr<LfoTestWrapper>
lfo_test_forecast_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			   unsigned int embed, unsigned int delay, unsigned int minn,
			   unsigned int step, py::object causal, py::object iterations,
			   double eps0, bool epsset, double epsf)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (comp, length)");

  auto comp = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(comp);
  for (unsigned int i = 0; i < comp; i++)
    rows[i] = series.mutable_data(i, 0);

  unsigned long resolved_causal = causal.is_none() ? (unsigned long)step : causal.cast<unsigned long>();
  unsigned long resolved_iterations = iterations.is_none() ? length : iterations.cast<unsigned long>();

  double bad_value = 0.0;
  LfoTest *result = lfo_test_forecast(rows.data(), length, comp, embed, delay, minn, step,
				       resolved_causal, resolved_iterations,
				       eps0, epsset ? 1 : 0, epsf, &bad_value);
  if (result == nullptr)
    throw std::invalid_argument(
	"either a component of series is constant (ranges from " +
	std::to_string(bad_value) + " to " + std::to_string(bad_value) +
	"), or comp/embed/series.shape[1] is 0, or length - (embed-1)*delay "
	"< minn (too few points to find enough neighbors for the fit)");

  return std::make_unique<LfoTestWrapper>(result);
}

// Owns an NstatZ* and exposes its fields as numpy arrays. Not copyable
// since NstatZ doesn't support that; pybind11 holds it by unique_ptr.
class NstatZWrapper {
public:
  explicit NstatZWrapper(NstatZ *result) : result_(result) {}
  NstatZWrapper(const NstatZWrapper &) = delete;
  NstatZWrapper &operator=(const NstatZWrapper &) = delete;
  ~NstatZWrapper() { nstat_z_free(result_); }

  unsigned int pieces() const { return result_->pieces; }
  unsigned long n_pairs() const { return result_->n_pairs; }

  py::array_t<unsigned int> first() const {
    py::array_t<unsigned int> out((py::ssize_t)result_->n_pairs);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_pairs; i++)
      buf(i) = result_->first[i];
    return out;
  }

  py::array_t<unsigned int> second() const {
    py::array_t<unsigned int> out((py::ssize_t)result_->n_pairs);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_pairs; i++)
      buf(i) = result_->second[i];
    return out;
  }

  py::array_t<double> value() const {
    py::array_t<double> out((py::ssize_t)result_->n_pairs);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned long i = 0; i < result_->n_pairs; i++)
      buf(i) = result_->value[i];
    return out;
  }

private:
  NstatZ *result_;
};

std::unique_ptr<NstatZWrapper>
nstat_z_compute_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
			 unsigned int pieces, unsigned int dim, unsigned int delay,
			 unsigned int minn, unsigned long step, py::object causal,
			 py::object center, py::object first_window,
			 py::object second_window, py::object first_offset,
			 py::object second_offset, double eps0, bool epsset, double epsf)
{
  if (series.ndim() != 1)
    throw std::invalid_argument("series must be a 1D array");

  auto length = (unsigned long)series.shape(0);
  if (length < 1)
    throw std::invalid_argument("series must be non-empty");
  if (pieces < 1)
    throw std::invalid_argument("pieces must be >= 1");
  if (dim < 1)
    throw std::invalid_argument("dim must be >= 1");

  py::array_t<char, py::array::c_style | py::array::forcecast> fw_arr, sw_arr;
  const char *fw_ptr = nullptr;
  const char *sw_ptr = nullptr;
  if (!first_window.is_none()) {
    fw_arr = py::array_t<char, py::array::c_style | py::array::forcecast>::ensure(first_window);
    if (!fw_arr || fw_arr.ndim() != 1 || (unsigned int)fw_arr.shape(0) != pieces)
      throw std::invalid_argument("first_window must be a 1D array-like of length pieces");
    fw_ptr = fw_arr.data();
  }
  if (!second_window.is_none()) {
    sw_arr = py::array_t<char, py::array::c_style | py::array::forcecast>::ensure(second_window);
    if (!sw_arr || sw_arr.ndim() != 1 || (unsigned int)sw_arr.shape(0) != pieces)
      throw std::invalid_argument("second_window must be a 1D array-like of length pieces");
    sw_ptr = sw_arr.data();
  }

  int resolved_first_offset = first_offset.is_none() ? -1 : first_offset.cast<int>();
  int resolved_second_offset = second_offset.is_none() ? -1 : second_offset.cast<int>();
  unsigned long resolved_causal = causal.is_none() ? step : causal.cast<unsigned long>();
  char centerset = center.is_none() ? 0 : 1;
  unsigned long resolved_center = center.is_none() ? 0 : center.cast<unsigned long>();

  NstatZError error;
  NstatZ *result = nstat_z_compute(series.data(), length, pieces, fw_ptr, sw_ptr,
				    resolved_first_offset, resolved_second_offset, dim,
				    delay, minn, step, resolved_causal, resolved_center,
				    centerset, eps0, epsset ? 1 : 0, epsf, &error);
  if (result == nullptr) {
    if (error == NSTAT_Z_ERR_ZERO_INTERVAL)
      throw std::invalid_argument("series is constant (zero range)");
    if (error == NSTAT_Z_ERR_ZERO_VARIANCE)
      throw std::invalid_argument("some piece of series has zero variance");
    throw std::invalid_argument(
	"pieces is too large: a piece has fewer than minn usable reference points");
  }

  return std::make_unique<NstatZWrapper>(result);
}

// Owns a GHKSSResult* and exposes its per-iteration fields as numpy arrays.
// Not copyable since GHKSSResult doesn't support that; pybind11 holds it by
// unique_ptr.
class GHKSSResultWrapper {
public:
  explicit GHKSSResultWrapper(GHKSSResult *result) : result_(result) {}
  GHKSSResultWrapper(const GHKSSResultWrapper &) = delete;
  GHKSSResultWrapper &operator=(const GHKSSResultWrapper &) = delete;
  ~GHKSSResultWrapper() { ghkss_free(result_); }

  unsigned int comp() const { return result_->comp; }
  unsigned long length() const { return result_->length; }
  unsigned int n_iterations() const { return result_->iterations; }

  py::array_t<double> series() const {
    unsigned int iters = result_->iterations, comp = result_->comp;
    unsigned long length = result_->length;
    py::array_t<double> out({(py::ssize_t)iters, (py::ssize_t)comp, (py::ssize_t)length});
    auto buf = out.mutable_unchecked<3>();
    for (unsigned int it = 0; it < iters; it++)
      for (unsigned int c = 0; c < comp; c++)
	for (unsigned long i = 0; i < length; i++)
	  buf(it, c, i) = result_->iters[it].series[c][i];
    return out;
  }

  py::array_t<double> shift() const {
    unsigned int iters = result_->iterations, comp = result_->comp;
    py::array_t<double> out({(py::ssize_t)iters, (py::ssize_t)comp});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int it = 0; it < iters; it++)
      for (unsigned int c = 0; c < comp; c++)
	buf(it, c) = result_->iters[it].shift[c];
    return out;
  }

  py::array_t<double> rms() const {
    unsigned int iters = result_->iterations, comp = result_->comp;
    py::array_t<double> out({(py::ssize_t)iters, (py::ssize_t)comp});
    auto buf = out.mutable_unchecked<2>();
    for (unsigned int it = 0; it < iters; it++)
      for (unsigned int c = 0; c < comp; c++)
	buf(it, c) = result_->iters[it].rms[c];
    return out;
  }

  py::array_t<bool> mineps_reset() const {
    unsigned int iters = result_->iterations;
    py::array_t<bool> out((py::ssize_t)iters);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int it = 0; it < iters; it++)
      buf(it) = result_->iters[it].mineps_reset != 0;
    return out;
  }

  py::array_t<double> mineps_after() const {
    unsigned int iters = result_->iterations;
    py::array_t<double> out((py::ssize_t)iters);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int it = 0; it < iters; it++)
      buf(it) = result_->iters[it].mineps_after;
    return out;
  }

  std::pair<py::array_t<double>, py::array_t<long>> correction_steps(unsigned int iteration) const {
    if (iteration >= result_->iterations)
      throw std::out_of_range("iteration must be < n_iterations");
    const GHKSSIteration &it = result_->iters[iteration];
    py::array_t<double> eps((py::ssize_t)it.n_correction_steps);
    py::array_t<long> count((py::ssize_t)it.n_correction_steps);
    auto ebuf = eps.mutable_unchecked<1>();
    auto cbuf = count.mutable_unchecked<1>();
    for (unsigned long s = 0; s < it.n_correction_steps; s++) {
      ebuf(s) = it.correction_steps[s].epsilon;
      cbuf(s) = (long)it.correction_steps[s].count;
    }
    return {eps, count};
  }

  std::pair<py::array_t<double>, py::array_t<long>> trend_steps(unsigned int iteration) const {
    if (iteration >= result_->iterations)
      throw std::out_of_range("iteration must be < n_iterations");
    const GHKSSIteration &it = result_->iters[iteration];
    py::array_t<double> eps((py::ssize_t)it.n_trend_steps);
    py::array_t<long> count((py::ssize_t)it.n_trend_steps);
    auto ebuf = eps.mutable_unchecked<1>();
    auto cbuf = count.mutable_unchecked<1>();
    for (unsigned long s = 0; s < it.n_trend_steps; s++) {
      ebuf(s) = it.trend_steps[s].epsilon;
      cbuf(s) = (long)it.trend_steps[s].count;
    }
    return {eps, count};
  }

private:
  GHKSSResult *result_;
};

std::unique_ptr<GHKSSResultWrapper>
ghkss_reduce_binding(py::array_t<double, py::array::c_style | py::array::forcecast> series,
		      unsigned int embed, unsigned int delay, unsigned int qdim,
		      unsigned int minn, py::object mineps, unsigned int iterations,
		      bool euclidean)
{
  if (series.ndim() != 2)
    throw std::invalid_argument("series must be a 2D array of shape (comp, length)");

  auto comp = (unsigned int)series.shape(0);
  auto length = (unsigned long)series.shape(1);

  std::vector<double *> rows(comp);
  for (unsigned int i = 0; i < comp; i++)
    rows[i] = series.mutable_data(i, 0);

  char eps_set = mineps.is_none() ? 0 : 1;
  double mineps_val = mineps.is_none() ? 0.0 : mineps.cast<double>();

  GHKSSError error;
  double bad_value = 0.0;
  GHKSSResult *result = ghkss_reduce(rows.data(), length, comp, embed, delay, qdim, minn,
				      mineps_val, eps_set, iterations, euclidean ? 1 : 0,
				      &error, &bad_value);
  if (result == nullptr) {
    if (error == GHKSS_ERR_TOO_MANY_NEIGHBORS)
      throw std::invalid_argument("series.shape[1] must be >= minn (can never find minn neighbors)");
    if (error == GHKSS_ERR_ZERO_INTERVAL)
      throw std::invalid_argument(
	  "a component of series is constant (ranges from " + std::to_string(bad_value) +
	  " to " + std::to_string(bad_value) + ")");
    throw std::invalid_argument(
	"the eigenvalue solver failed to converge for some point's local correction matrix");
  }

  return std::make_unique<GHKSSResultWrapper>(result);
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
  auto recurr = m.def_submodule(
      "recurr", "Recurrence plot neighbor search (source_c/recurr.c)");

  py::class_<RecurrResultWrapper>(recurr, "RecurrResult")
      .def_property_readonly("count", &RecurrResultWrapper::count)
      .def_property_readonly("point", &RecurrResultWrapper::point,
			      "1-based index of the scanned point, shape (count,)")
      .def_property_readonly("neighbor", &RecurrResultWrapper::neighbor,
			      "1-based index of the recurrence neighbor, "
			      "shape (count,)");

  recurr.def(
      "find", &recurr_find_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("eps") = 1.e-3,
      py::arg("eps_is_raw") = false, py::arg("fraction") = 1.0,
      "Find recurrence-plot neighbor pairs in `series` (shape (dim, "
      "length)). Each dimension is independently rescaled to [0,1) the "
      "same way the CLI does it. If eps_is_raw is True, `eps` is "
      "interpreted in the original (raw) data units and divided by the "
      "largest per-dimension raw interval before use, matching the CLI's "
      "-r flag; if False (default), `eps` is used as-is in the "
      "already-rescaled [0,1) space, matching the CLI's default. `embed` "
      "and `delay` match the CLI's -m (embedding dimension part) and -d; "
      "`fraction` matches the CLI's -% (as a 0..1 fraction rather than a "
      "percentage).");

  auto mem_spec = m.def_submodule(
      "mem_spec", "AR power spectrum estimation via Burg's method (source_c/mem_spec.c)");

  py::class_<MemSpecModelWrapper>(mem_spec, "MemSpecModel")
      .def_property_readonly("poles", &MemSpecModelWrapper::poles)
      .def_property_readonly("sigma2", &MemSpecModelWrapper::sigma2,
			      "Residual variance of the Burg fit")
      .def_property_readonly("coef", &MemSpecModelWrapper::coef,
			      "AR (reflection) coefficients, shape (poles,)")
      .def("spectrum", &MemSpecModelWrapper::spectrum, py::arg("out") = 2000,
	   py::arg("samplingrate") = 1.0,
	   "Evaluate the power spectrum implied by this model at `out`\n"
	   "frequencies spaced over [0, samplingrate/2), matching the "
	   "mem_spec CLI's -P/-f options. Returns a (freq, spec) tuple of "
	   "1D arrays, each of length `out`. spec is unscaled, matching "
	   "what the CLI prints to stdout; the CLI additionally divides by "
	   "sqrt(len(series)) only when writing to a file with -o.");

  mem_spec.def(
      "fit", &mem_spec_fit_binding, py::arg("series"), py::arg("poles") = 128,
      "Fit an AR power-spectrum model to `series` via Burg's method, "
      "matching the mem_spec CLI's -p option (default 128). series is "
      "mean-centered internally the same way the CLI does it before "
      "fitting; the input array itself is not modified.");

  auto lyap_r = m.def_submodule(
      "lyap_r", "Maximal Lyapunov exponent via Rosenstein et al. (source_c/lyap_r.c)");

  py::class_<LyapRWrapper>(lyap_r, "LyapRResult")
      .def_property_readonly("steps", &LyapRWrapper::steps)
      .def_property_readonly("found", &LyapRWrapper::found,
			      "Number of point pairs contributing to lyap[i] "
			      "for each step, shape (steps+1,); 0 means no "
			      "data for that step")
      .def_property_readonly("lyap", &LyapRWrapper::lyap,
			      "Raw sum of log(squared divergence) for each "
			      "step, shape (steps+1,); divide by found[i] "
			      "and 2.0 to get the CLI's printed value "
			      "wherever found[i] > 0");

  lyap_r.def(
      "compute", &lyap_r_compute_binding, py::arg("series"), py::arg("dim") = 2,
      py::arg("delay") = 1, py::arg("mindist") = 0, py::arg("steps") = 10,
      py::arg("eps0") = 1.e-3, py::arg("epsset") = false,
      "Estimate the maximal Lyapunov exponent of `series` via the method "
      "of Rosenstein et al., matching the lyap_r CLI's -m/-d/-t/-s/-r "
      "options. series is rescaled to [0,1) internally (the input array "
      "is not modified); a box-assisted nearest-neighbor search at a "
      "growing radius (starting at eps0, in rescaled [0,1) units unless "
      "epsset=True, in which case eps0 is interpreted in the same units "
      "as the raw input data and divided by its own data range, matching "
      "the CLI's -r flag) accumulates the log divergence of nearby "
      "trajectories for `steps` iteration steps ahead.");

  auto makenoise = m.def_submodule(
      "makenoise", "Add uniform or Gaussian noise to a time series (source_c/makenoise.c)");

  py::class_<MakeNoiseWrapper>(makenoise, "MakeNoise")
      .def_property_readonly("dim", &MakeNoiseWrapper::dim)
      .def_property_readonly("length", &MakeNoiseWrapper::length)
      .def_property_readonly("series", &MakeNoiseWrapper::series,
			      "Noisy series, shape (dim, length)");

  makenoise.def(
      "add", &makenoise_add_binding, py::arg("series"), py::arg("noiselevel") = 0.05,
      py::arg("absolute") = false, py::arg("gaussian") = false,
      py::arg("seed") = 3441341UL,
      "Add uniform (or, if gaussian=True, Gaussian) noise to `series` "
      "(shape (dim, length)), matching the makenoise CLI's equidistri()/"
      "gauss(). noiselevel scales relative to each row's own standard "
      "deviation unless absolute=True (matching the CLI's -r flag), in "
      "which case it is used as-is. seed matches the CLI's -I option "
      "(default 3441341); the RNG is warmed up with 10000 discarded draws "
      "first, exactly like the CLI.");

  auto sav_gol = m.def_submodule(
      "sav_gol", "Savitzky-Golay filter/derivative estimation (source_c/sav_gol.c)");

  py::class_<SavGolWrapper>(sav_gol, "SavGol")
      .def_property_readonly("dim", &SavGolWrapper::dim)
      .def_property_readonly("length", &SavGolWrapper::length)
      .def_property_readonly("data", &SavGolWrapper::data,
			      "Filtered series (or estimated derivative, if "
			      "deriv != 0), shape (dim, length). The first "
			      "nb and last nf points per row are left "
			      "unfiltered (deriv == 0) or set to 0.0 "
			      "(deriv != 0).");

  sav_gol.def(
      "filter", &sav_gol_filter_binding, py::arg("series"), py::arg("nb") = 2,
      py::arg("nf") = 2, py::arg("power") = 2, py::arg("deriv") = 0,
      "Apply a Savitzky-Golay filter to `series` (shape (dim, length)): "
      "fits a degree-`power` polynomial through the `nb` points before and "
      "`nf` points after each point (except within `nb`/`nf` of either "
      "edge) and evaluates its `deriv`-th derivative there, normalized by "
      "1/deriv!, matching the sav_gol CLI's -n/-p/-D options.");

  auto lzo_gm = m.def_submodule(
      "lzo_gm", "Average local-constant forecast error vs. neighborhood size (source_c/lzo-gm.c)");

  py::class_<LzoGmResultWrapper>(lzo_gm, "LzoGmResult")
      .def_property_readonly("dim", &LzoGmResultWrapper::dim)
      .def_property_readonly("n_rows", &LzoGmResultWrapper::n_rows,
			      "Number of qualifying neighborhood sizes (only "
			      "sizes with more than one contributing point "
			      "are kept, matching the CLI's own row filter)")
      .def_property_readonly("epsilon", &LzoGmResultWrapper::epsilon,
			      "Neighborhood size in raw data units, shape (n_rows,)")
      .def_property_readonly("avg_error", &LzoGmResultWrapper::avg_error,
			      "Relative forecast error averaged over all dim "
			      "components, shape (n_rows,)")
      .def_property_readonly("error", &LzoGmResultWrapper::error,
			      "Relative forecast error per component, shape "
			      "(n_rows, dim)")
      .def_property_readonly("fraction", &LzoGmResultWrapper::fraction,
			      "Fraction of scanned points that had enough "
			      "neighbors to contribute, shape (n_rows,)")
      .def_property_readonly("avneighbors", &LzoGmResultWrapper::avneighbors,
			      "Average number of neighbors found per "
			      "contributing point, shape (n_rows,)");

  lzo_gm.def(
      "compute", &lzo_gm_compute_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("step") = 1, py::arg("causal") = py::none(),
      py::arg("iterations") = py::none(), py::arg("eps0") = 1.e-3,
      py::arg("eps0_raw") = false, py::arg("eps1") = 1.0, py::arg("eps1_raw") = false,
      py::arg("epsf") = 1.2,
      "Estimate, for a growing sequence of neighborhood sizes, the average\n"
      "forecast error of a local-constant fit on `series` (shape (dim,\n"
      "length)), matching the lzo-gm CLI's -m (embedding dimension part)/\n"
      "-d/-s/-C/-i/-r/-R/-f options. Each dimension is independently\n"
      "rescaled to [0,1) internally (the input array is not modified). causal\n"
      "(the CLI's -C) defaults to `step` if not given (None), matching the\n"
      "CLI's default (unset -C). iterations (the CLI's -i) defaults to\n"
      "series.shape[1] if not given (None), matching the CLI's default of\n"
      "the whole series. eps0/eps1 are in rescaled [0,1) units unless\n"
      "eps0_raw/eps1_raw is True, in which case the respective value is\n"
      "interpreted in the original (raw) data units and divided by the\n"
      "largest per-dimension raw interval before use, matching the CLI's\n"
      "-r/-R flags.");

  auto lfo_ar = m.def_submodule(
      "lfo_ar", "Average local-linear (AR) forecast error vs. neighborhood size (source_c/lfo-ar.c)");

  py::class_<LfoArResultWrapper>(lfo_ar, "LfoArResult")
      .def_property_readonly("dim", &LfoArResultWrapper::dim)
      .def_property_readonly("n_rows", &LfoArResultWrapper::n_rows,
			      "Number of qualifying neighborhood sizes (only "
			      "sizes with more than one contributing point "
			      "are kept, matching the CLI's own row filter)")
      .def_property_readonly("epsilon", &LfoArResultWrapper::epsilon,
			      "Neighborhood size in raw data units, shape (n_rows,)")
      .def_property_readonly("avg_error", &LfoArResultWrapper::avg_error,
			      "Relative forecast error averaged over all dim "
			      "components, shape (n_rows,)")
      .def_property_readonly("error", &LfoArResultWrapper::error,
			      "Relative forecast error per component, shape "
			      "(n_rows, dim)")
      .def_property_readonly("fraction", &LfoArResultWrapper::fraction,
			      "Fraction of scanned points that had enough "
			      "neighbors to contribute, shape (n_rows,)")
      .def_property_readonly("avneighbors", &LfoArResultWrapper::avneighbors,
			      "Average number of neighbors found per "
			      "contributing point, shape (n_rows,)");

  lfo_ar.def(
      "compute", &lfo_ar_compute_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("step") = 1, py::arg("causal") = py::none(),
      py::arg("iterations") = py::none(), py::arg("eps0") = 1.e-3,
      py::arg("eps0_raw") = false, py::arg("eps1") = 1.0, py::arg("eps1_raw") = false,
      py::arg("epsf") = 1.2,
      "Estimate, for a growing sequence of neighborhood sizes, the average\n"
      "forecast error of a local-linear (AR) fit on `series` (shape (dim,\n"
      "length)), matching the lfo-ar CLI's -m (embedding dimension part)/\n"
      "-d/-s/-C/-i/-r/-R/-f options. Each dimension is independently\n"
      "rescaled to [0,1) internally (the input array is not modified). causal\n"
      "(the CLI's -C) defaults to `step` if not given (None), matching the\n"
      "CLI's default (unset -C). iterations (the CLI's -i) defaults to\n"
      "series.shape[1] if not given (None), matching the CLI's default of\n"
      "the whole series. eps0/eps1 are in rescaled [0,1) units unless\n"
      "eps0_raw/eps1_raw is True, in which case the respective value is\n"
      "interpreted in the original (raw) data units and divided by the\n"
      "largest per-dimension raw interval before use, matching the CLI's\n"
      "-r/-R flags.");

  auto polynomp = m.def_submodule(
      "polynomp", "Polynomial fit and forecast of a scalar series (source_c/polynomp.c)");

  py::class_<PolynompResultWrapper>(polynomp, "PolynompResult")
      .def_property_readonly("dim", &PolynompResultWrapper::dim)
      .def_property_readonly("delay", &PolynompResultWrapper::delay)
      .def_property_readonly("plength", &PolynompResultWrapper::plength,
			      "Number of polynomial terms/coefficients")
      .def_property_readonly("param", &PolynompResultWrapper::param,
			      "Fitted coefficients, shape (plength,), in the "
			      "same term order as `order`")
      .def_property_readonly("fce_insample", &PolynompResultWrapper::fce_insample,
			      "In-sample forecast error, normalized by the "
			      "series' standard deviation")
      .def_property_readonly("has_outsample", &PolynompResultWrapper::has_outsample,
			      "Whether fce_outsample was computed, i.e. "
			      "whether insample < series.shape[0]")
      .def_property_readonly("fce_outsample", &PolynompResultWrapper::fce_outsample,
			      "Out-of-sample forecast error on [insample+1, "
			      "length); 0.0 if has_outsample is False")
      .def_property_readonly("step", &PolynompResultWrapper::step)
      .def_property_readonly("forecast", &PolynompResultWrapper::forecast,
			      "Values continuing the series, shape (step,)");

  polynomp.def(
      "fit", &polynomp_fit_binding, py::arg("series"), py::arg("order"),
      py::arg("delay") = 1, py::arg("insample") = std::numeric_limits<unsigned long>::max(),
      py::arg("step") = 1000,
      "Fit a polynomial to `series` (1D) and forecast it `step` points\n"
      "forward, matching the polynomp CLI's -d/-n/-L options. `order` is a\n"
      "2D array of shape (plength, dim) of non-negative exponents (e.g.\n"
      "tisean.polypar.generate(dim, order).params): term i of the\n"
      "polynomial is the product over j in [0, dim) of\n"
      "series[act - j*delay] ** order[i, j]. dim is taken from order's\n"
      "second dimension, matching the CLI's -m (which also controls how\n"
      "many columns are read from the parameter file). insample selects\n"
      "how much of the series is used to fit the model; leaving it unset\n"
      "uses the whole series and leaves fce_outsample/has_outsample\n"
      "unset, matching the CLI's default (-n unset). Raises ValueError if\n"
      "series is constant, if the normal-equations matrix is singular, or\n"
      "if order's shape or series' length is degenerate for the given\n"
      "dim/delay.");

  auto fsle = m.def_submodule(
      "fsle", "Finite-size Lyapunov exponent spectrum via Vulpiani et al. (source_c/fsle.c)");

  py::class_<FSLEResultWrapper>(fsle, "FSLEResult")
      .def_property_readonly("n", &FSLEResultWrapper::n,
			      "Number of populated epsilon bins (bins with at "
			      "least one divergence event)")
      .def_property_readonly("eps", &FSLEResultWrapper::eps,
			      "Epsilon of each bin, in the original series' "
			      "units, shape (n,)")
      .def_property_readonly("lyapunov", &FSLEResultWrapper::lyapunov,
			      "Finite-size Lyapunov exponent estimate for "
			      "each bin, shape (n,)")
      .def_property_readonly("count", &FSLEResultWrapper::count,
			      "Number of divergence events contributing to "
			      "each bin, shape (n,)");

  fsle.def(
      "compute", &fsle_compute_binding, py::arg("series"), py::arg("dim") = 2,
      py::arg("delay") = 1, py::arg("mindist") = 0, py::arg("eps0") = 1.e-3,
      py::arg("epsset") = false,
      "Estimate the finite-size Lyapunov exponent spectrum of `series` via\n"
      "the method of Vulpiani et al., matching the fsle CLI's -m/-d/-t/-r\n"
      "options. series is centered/rescaled to [0,1] internally (the input\n"
      "array is not modified); pairs of nearby trajectory points are\n"
      "tracked as they diverge across exponentially-spaced (factor\n"
      "sqrt(2)) epsilon bins starting from eps0. If epsset=False (the\n"
      "default), eps0 is treated as a fraction of the rescaled series'\n"
      "standard deviation; if epsset=True, eps0 is interpreted in the same\n"
      "units as the raw input data, matching the CLI's -r flag. Raises\n"
      "ValueError if series is constant, or if the resulting starting\n"
      "epsilon is not smaller than the data's own maximal epsilon.");

  auto false_nearest = m.def_submodule(
      "false_nearest", "Fraction of false nearest neighbors (source_c/false_nearest.c)");

  py::class_<FalseNearestWrapper>(false_nearest, "FalseNearestResult")
      .def_property_readonly("n", &FalseNearestWrapper::n,
			      "Number of embedding dimensions computed")
      .def_property_readonly("dimension", &FalseNearestWrapper::dimension,
			      "Total embedding dimension (comp * emb) for "
			      "each row, shape (n,)")
      .def_property_readonly("fraction", &FalseNearestWrapper::fraction,
			      "Fraction of false nearest neighbors, shape (n,)")
      .def_property_readonly("avg_eps", &FalseNearestWrapper::avg_eps,
			      "Average neighbor distance at which a false "
			      "neighbor was decided, in the original series' "
			      "units, shape (n,)")
      .def_property_readonly("sigma_eps", &FalseNearestWrapper::sigma_eps,
			      "Standard deviation of that distance, in the "
			      "original series' units, shape (n,)");

  false_nearest.def(
      "compute", &false_nearest_compute_binding, py::arg("series"),
      py::arg("minemb") = 1, py::arg("maxemb") = 5, py::arg("delay") = 1,
      py::arg("theiler") = 0, py::arg("rt") = 2.0, py::arg("eps0") = 1.0e-5,
      "Estimate the fraction of false nearest neighbors of `series`\n"
      "(shape (comp, length)) for every total embedding dimension\n"
      "comp*emb, emb running from minemb to maxemb inclusive, matching the\n"
      "false_nearest CLI's -m/-M/-d/-t/-f options. series is internally\n"
      "rescaled to [0,1) per component (the input array is not modified);\n"
      "`rt` is the escape factor and `theiler` is the Theiler window.\n"
      "Raises ValueError if some component is constant, or if no neighbor\n"
      "pair is found for some embedding dimension.");

  auto pca = m.def_submodule(
      "pca", "Global PCA of the delay-embedded covariance matrix (source_c/pca.c)");

  py::class_<PCAWrapper>(pca, "PCA")
      .def_property_readonly("dimemb", &PCAWrapper::dimemb,
			      "Size of the covariance matrix (dim*emb)")
      .def_property_readonly("eigenvalues", &PCAWrapper::eigenvalues,
			      "Eigenvalues sorted descending, shape (dimemb,)")
      .def_property_readonly("eigenvectors", &PCAWrapper::eigenvectors,
			      "Eigenvectors, shape (dimemb, dimemb); "
			      "eigenvectors[i, j] is component i of the "
			      "eigenvector for eigenvalues[j]");

  pca.def(
      "compute", &pca_compute_binding, py::arg("series"), py::arg("emb") = 1,
      py::arg("delay") = 1,
      "Compute the eigenvalues/eigenvectors of the covariance matrix of\n"
      "`series` (shape (dim, length)), matching the pca CLI's -m (embedding\n"
      "dimension part)/-d options. series is expected to already be\n"
      "centered (zero mean per row), the same way the pca CLI centers its\n"
      "input before computing. Each of the dim rows is embedded with `emb`\n"
      "delayed copies (spaced `delay` apart) before the dim*emb by dim*emb\n"
      "covariance matrix is built. Raises ValueError if the eigenvalue\n"
      "solver fails to converge.");

  auto delay = m.def_submodule(
      "delay", "Delay-embedding vector construction (source_c/delay.c)");

  py::class_<DelayResultWrapper>(delay, "DelayResult")
      .def_property_readonly("alldim", &DelayResultWrapper::alldim,
			      "Total embedding dimension, sum(format)")
      .def_property_readonly("n_vectors", &DelayResultWrapper::n_vectors,
			      "Number of delay vectors produced")
      .def_property_readonly("vectors", &DelayResultWrapper::vectors,
			      "Delay vectors, shape (n_vectors, alldim)");

  delay.def(
      "embed", &delay_embed_binding, py::arg("series"), py::arg("embdim") = 2,
      py::arg("format") = py::none(), py::arg("delay") = 1,
      py::arg("multidelay") = py::none(),
      "Build delay-embedding vectors from `series` (shape (indim, "
      "length)), matching the delay CLI's stdout output (its -m/-F/-d/-D "
      "options; -M is series.shape[0]).\n\n"
      "format (the CLI's -F) is a 1D array of indim positive ints giving "
      "how many embedded coordinates to take from each row of series. If "
      "not given (None, the default), it is built from embdim (the CLI's "
      "-m, default 2) split evenly across series.shape[0] rows, matching "
      "the CLI's default behavior; embdim must then be a multiple of "
      "series.shape[0].\n\n"
      "delay (the CLI's -d, default 1) sets a uniform lag spacing between "
      "consecutive embedded coordinates of every row. multidelay (the "
      "CLI's -D) overrides this with an explicit 1D array of "
      "sum(format) - series.shape[0] per-coordinate lags (one for every "
      "embedded coordinate after the first of each row, concatenated "
      "across rows in row order); delay is ignored if multidelay is "
      "given.\n\n"
      "Output row t is built from input time index t + max(lags): "
      "coordinate k of the row is series[row][t + max(lags) - lag_k]. "
      "n_vectors is series.shape[1] - max(lags), or 0 if series.shape[1] "
      "is not larger than max(lags) (matching the CLI, which then prints "
      "nothing rather than erroring).");

  auto lyap_k = m.def_submodule(
      "lyap_k", "Maximal Lyapunov exponent via Kantz (source_c/lyap_k.c)");

  py::class_<LyapKWrapper>(lyap_k, "LyapKResult")
      .def_property_readonly("epscount", &LyapKWrapper::epscount,
			      "Number of epsilon values actually used (may "
			      "be forced to 1 if mineps is not smaller than "
			      "maxeps)")
      .def_property_readonly("mindim", &LyapKWrapper::mindim,
			      "Clamped mindim actually used (>= 2)")
      .def_property_readonly("maxdim", &LyapKWrapper::maxdim,
			      "Clamped maxdim actually used (>= mindim)")
      .def_property_readonly("maxiter", &LyapKWrapper::maxiter,
			      "Number of iteration steps; count/lyap hold "
			      "entries for steps 0..maxiter")
      .def_property_readonly("epsilon", &LyapKWrapper::epsilon,
			      "Neighborhood radius used for each row, in "
			      "the original series' units, shape "
			      "(epscount,)")
      .def_property_readonly("count", &LyapKWrapper::count,
			      "Number of reference points contributing to "
			      "lyap[e,d,j], shape (epscount, "
			      "maxdim-mindim+1, maxiter+1); 0 means no data "
			      "for that step, mirroring the CLI's skipping "
			      "that row entirely in its output")
      .def_property_readonly("lyap", &LyapKWrapper::lyap,
			      "Raw sum of already-averaged-per-reference-"
			      "point log divergences, shape (epscount, "
			      "maxdim-mindim+1, maxiter+1); divide by "
			      "count[e,d,j] to get the CLI's printed value, "
			      "only meaningful where count[e,d,j] > 0");

  lyap_k.def(
      "compute", &lyap_k_compute_binding, py::arg("series"), py::arg("mindim") = 2,
      py::arg("maxdim") = 2, py::arg("delay") = 1, py::arg("epsmin") = 1.e-3,
      py::arg("epsmax") = 1.e-2, py::arg("eps0set") = false,
      py::arg("eps1set") = false, py::arg("epscount") = 5,
      py::arg("reference") = std::numeric_limits<unsigned long>::max(),
      py::arg("maxiter") = 50, py::arg("window") = 0,
      "Estimate the maximal Lyapunov exponent of `series` via the method "
      "of Kantz, matching the lyap_k CLI's -m/-M/-d/-r/-R/-#/-n/-s/-t "
      "options. series is rescaled to [0,1) internally (the input array "
      "is not modified); for each of epscount neighborhood radii "
      "(geometrically spaced between epsmin and epsmax, in rescaled [0,1) "
      "units unless eps0set/eps1set is True, in which case epsmin/epsmax "
      "are interpreted in the same units as the raw input data and "
      "divided by its own data range, matching the CLI's -r/-R flags), a "
      "box-assisted nearest-neighbor search finds, for the first "
      "`reference` points, neighbors within that radius (skipping any "
      "within `window` samples of the reference point) for every "
      "embedding dimension between mindim and maxdim, and accumulates "
      "the log divergence of the two trajectories for maxiter iteration "
      "steps ahead. reference defaults to using every point (the CLI's "
      "default of '# of data'). Raises ValueError if series is constant, "
      "or if series is too short for the given mindim/maxdim/delay/"
      "maxiter.");

  auto lzo_test = m.def_submodule(
      "lzo_test", "Zeroth-order forecast error vs. horizon (source_c/lzo-test.c)");

  py::class_<LzoTestWrapper>(lzo_test, "LzoTestResult")
      .def_property_readonly("dim", &LzoTestWrapper::dim)
      .def_property_readonly("step", &LzoTestWrapper::step)
      .def_property_readonly("n_ref", &LzoTestWrapper::n_ref)
      .def_property_readonly("error", &LzoTestWrapper::error,
			      "Relative forecast error per horizon/component, "
			      "shape (step, dim); row i is horizon i+1 - the "
			      "same values the CLI's default output prints")
      .def_property_readonly("diffs", &LzoTestWrapper::diffs,
			      "Per-reference-point one-step forecast "
			      "differences, scaled back into the original "
			      "series' units, shape (n_ref, dim) - matches "
			      "the CLI's extra output rows under -V2");

  lzo_test.def(
      "compute", &lzo_test_compute_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("minn") = 30, py::arg("step") = 1,
      py::arg("refstep") = 1, py::arg("causal") = py::none(),
      py::arg("n_ref") = py::none(), py::arg("eps0") = 1.e-3,
      py::arg("epsset") = false, py::arg("epsf") = 1.2,
      "Estimate the average forecast error of a zeroth-order (local-\n"
      "constant) fit on `series` (shape (dim, length)) for forecast\n"
      "horizons 1..step, matching the lzo-test CLI's -m (embedding\n"
      "dimension part)/-d/-k/-s/-S/-C/-n/-r/-f options. Each dimension is\n"
      "independently rescaled to [0,1) internally (the input array is not\n"
      "modified). causal (the CLI's -C) defaults to `step` if not given\n"
      "(None), matching the CLI's default (unset -C). n_ref (the CLI's\n"
      "-n) defaults to using the whole series if not given (None),\n"
      "matching the CLI's default of 'length'; the number of reference\n"
      "points actually scanned is then clamped to fit within length given\n"
      "step/refstep, same as the CLI. eps0 is the starting search radius\n"
      "in rescaled [0,1) units, unless epsset=True, in which case eps0 is\n"
      "interpreted in the same units as the raw input data and divided by\n"
      "the average of the per-component raw intervals before use,\n"
      "matching the CLI's -r flag. epsf is the growth factor for the\n"
      "search radius. Raises ValueError if some component is constant or\n"
      "has zero variance after rescaling.");

  auto boxcount = m.def_submodule(
      "boxcount", "Renyi entropy via box partition (source_c/boxcount.c)");

  py::class_<BoxCountWrapper>(boxcount, "BoxCountResult")
      .def_property_readonly("dimension", &BoxCountWrapper::dimension)
      .def_property_readonly("maxembed", &BoxCountWrapper::maxembed)
      .def_property_readonly("epscount", &BoxCountWrapper::epscount)
      .def_property_readonly("eps", &BoxCountWrapper::eps,
			      "Epsilon values in the original series' units, "
			      "shape (epscount,)")
      .def_property_readonly("entropy", &BoxCountWrapper::entropy,
			      "Generalized entropy of order q, shape "
			      "(epscount, dimension*maxembed); column i "
			      "corresponds to (which_component[i], "
			      "which_embed[i])")
      .def_property_readonly("which_component", &BoxCountWrapper::which_component,
			      "0-based component index per output column, "
			      "shape (dimension*maxembed,)")
      .def_property_readonly("which_embed", &BoxCountWrapper::which_embed,
			      "0-based embedding index per output column, "
			      "shape (dimension*maxembed,)");

  boxcount.def(
      "compute", &boxcount_compute_binding, py::arg("series"), py::arg("maxembed") = 10,
      py::arg("delay") = 1, py::arg("q") = 2.0, py::arg("epsmin") = 1.e-3,
      py::arg("epsmin_absolute") = false, py::arg("epsmax") = 1.0,
      py::arg("epsmax_absolute") = false, py::arg("epscount") = 20,
      "Estimate the generalized (Renyi) entropy of order q of `series`\n"
      "(shape (dimension, length)) via a recursive box partition, matching\n"
      "the boxcount CLI's -M (maxembed part)/-d/-Q/-r/-R/-# options. Each\n"
      "component is independently rescaled to [0,1) internally (the input\n"
      "array is not modified). epsmin/epsmax are box sizes as fractions of\n"
      "that rescaled range, unless epsmin_absolute/epsmax_absolute is\n"
      "True, in which case they are interpreted in the same units as the\n"
      "raw input data and divided by the largest per-component raw range,\n"
      "matching the CLI's -r/-R flags (which set epsminset/epsmaxset).\n"
      "Raises ValueError if series is constant (zero range) for some\n"
      "component, or too short for the given maxembed/delay.");

  auto nrlazy = m.def_submodule(
      "nrlazy", "Simple multivariate nonlinear noise reduction (source_c/nrlazy.c)");

  py::class_<NRLazyResultWrapper>(nrlazy, "NRLazyResult")
      .def_property_readonly("comp", &NRLazyResultWrapper::comp)
      .def_property_readonly("length", &NRLazyResultWrapper::length)
      .def_property_readonly("series", &NRLazyResultWrapper::series,
			      "Corrected data, scaled back to the original units, "
			      "shape (comp, length)")
      .def_property_readonly("neighbors", &NRLazyResultWrapper::neighbors,
			      "Number of neighbors found for each point during the "
			      "last iteration, shape (length,)");

  nrlazy.def(
      "correct", &nrlazy_correct_binding, py::arg("series"), py::arg("embed") = 5,
      py::arg("delay") = 1, py::arg("iterations") = 1, py::arg("r") = py::none(),
      py::arg("v") = py::none(),
      "Replace every embedded point of `series` (shape (comp, length)) by "
      "the average of its neighbors in delay-embedding space, matching the "
      "nrlazy CLI's -m (embedding dim part)/-d/-i/-r/-v options. Each "
      "component is independently rescaled to [0,1) internally (the input "
      "array is not modified) before `iterations` correction passes.\n\n"
      "r is the neighborhood size as a fraction of the largest "
      "per-component raw data interval (matching the CLI's -r); if not "
      "given, it defaults to a plain 1.e-3 in the rescaled [0,1) space. v "
      "is the neighborhood size in units of the largest per-component "
      "standard deviation of the rescaled data (matching the CLI's -v); if "
      "given, it overwrites whatever r would have produced.\n\n"
      "Raises ValueError if series is constant (zero range) for some "
      "component, or too short for the given embed/delay.");

  auto rbf = m.def_submodule(
      "rbf", "Radial-basis-function model fit and forecast of a scalar series (source_c/rbf.c)");

  py::class_<RBFResultWrapper>(rbf, "RBFResult")
      .def_property_readonly("dim", &RBFResultWrapper::dim)
      .def_property_readonly("delay", &RBFResultWrapper::delay)
      .def_property_readonly("centers", &RBFResultWrapper::centers,
			      "Number of RBF centers actually used, i.e. "
			      "min(centers, series length)")
      .def_property_readonly("step", &RBFResultWrapper::step)
      .def_property_readonly("insample", &RBFResultWrapper::insample,
			      "Number of points actually used to fit the "
			      "model, i.e. min(insample, series length)")
      .def_property_readonly("length", &RBFResultWrapper::length)
      .def_property_readonly("center", &RBFResultWrapper::center,
			      "RBF center coordinates, shape (centers, dim), "
			      "in original (unscaled) data units")
      .def_property_readonly("variance", &RBFResultWrapper::variance,
			      "RBF kernel width parameter, in original "
			      "(unscaled) data units")
      .def_property_readonly("coefs", &RBFResultWrapper::coefs,
			      "Fitted coefficients, shape (centers+1,); "
			      "coefs[0] is the intercept, coefs[1:] are the "
			      "per-center weights")
      .def_property_readonly("insample_error", &RBFResultWrapper::insample_error,
			      "Normalized in-sample RMS forecast error")
      .def_property_readonly("has_outsample_error", &RBFResultWrapper::has_outsample_error,
			      "Whether outsample_error was computed, i.e. "
			      "whether insample < series.shape[0]")
      .def_property_readonly("outsample_error", &RBFResultWrapper::outsample_error,
			      "Normalized out-of-sample RMS forecast error; "
			      "0.0 if has_outsample_error is False")
      .def_property_readonly("cast_length", &RBFResultWrapper::cast_length)
      .def_property_readonly("cast", &RBFResultWrapper::cast,
			      "Forecasted values continuing the series, "
			      "shape (cast_length,), in original units");

  rbf.def(
      "fit", &rbf_fit_binding, py::arg("series"), py::arg("dim") = 2, py::arg("delay") = 1,
      py::arg("centers") = 10, py::arg("drift") = true, py::arg("step") = 1,
      py::arg("insample") = std::numeric_limits<unsigned long>::max(),
      py::arg("cast_length") = 0,
      "Fit a radial-basis-function model to `series` (1D) and optionally\n"
      "forecast it cast_length points forward, matching the rbf CLI's\n"
      "-m/-d/-p/-X/-s/-n/-L options. centers is clamped to len(series);\n"
      "the number of centers actually used is returned in the result as\n"
      "`centers`. drift=True (the CLI's default, drift=False matches -X)\n"
      "applies a repulsion optimization to the initial evenly-spaced\n"
      "center placement before fitting. insample selects how much of the\n"
      "series is used to fit the model; leaving it unset uses the whole\n"
      "series and leaves outsample_error/has_outsample_error unset,\n"
      "matching the CLI's default (-n unset). Leaving cast_length at 0\n"
      "(the CLI's default, i.e. no -L) skips forecasting.\n\n"
      "Raises ValueError if series is constant, if the normal-equations\n"
      "matrix for the RBF weights is singular, if dim < 1, if series is\n"
      "too short for dim/delay, if min(centers, len(series)) < 2, or if\n"
      "min(insample, len(series)) < step.");

  auto lzo_run = m.def_submodule(
      "lzo_run", "Iterated local zeroth-order (nearest-neighbor) forecast (source_c/lzo-run.c)");

  py::class_<LzoRunWrapper>(lzo_run, "LzoRun")
      .def_property_readonly("dim", &LzoRunWrapper::dim)
      .def_property_readonly("length", &LzoRunWrapper::length,
			      "Number of iterated forecast points (the CLI's -L)")
      .def_property_readonly("series", &LzoRunWrapper::series,
			      "Iterated forecast trajectory, shape (length, dim), "
			      "in original (unscaled) data units");

  lzo_run.def(
      "forecast", &lzo_run_forecast_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("minn") = 50, py::arg("fix_neighbors") = true,
      py::arg("flength") = 1000, py::arg("eps0") = 1.e-3, py::arg("epsset") = false,
      py::arg("epsf") = 1.2, py::arg("noise_pct") = py::none(),
      py::arg("seed") = 0x9074325UL,
      "Iterate a local zeroth-order forecast forward `flength` steps from\n"
      "`series` (shape (dim, length)), matching the lzo-run CLI's\n"
      "-m/-d/-k/-K/-L/-r/-f/-%/-I options.\n\n"
      "fix_neighbors mirrors the CLI's -K, but the shipped CLI always\n"
      "behaves as if -K were given (its setsort global defaults to 1 and\n"
      "is never reset to 0), so fix_neighbors=True is the default here too\n"
      "and is what CLI output actually corresponds to.\n\n"
      "noise_pct adds Gaussian noise (percentage of each component's own "
      "rescaled variance) to every forecast point when given a value > 0, "
      "matching the CLI's -%; the default of None matches the CLI not "
      "being passed -% at all (no noise), regardless of the flag's own "
      "documented default of 10.0.\n\n"
      "Raises ValueError if series is constant (zero range) for some "
      "component, if series has zero variance for some component, if "
      "dim < 1, embed < 1, or series is too short for embed/delay.");

  auto lfo_run = m.def_submodule(
      "lfo_run", "Iterated local-linear (or zeroth order) forecast (source_c/lfo-run.c)");

  py::class_<LfoRunWrapper>(lfo_run, "LfoRun")
      .def_property_readonly("dim", &LfoRunWrapper::dim)
      .def_property_readonly("length", &LfoRunWrapper::length,
			      "Number of iterated forecast points (the CLI's -L)")
      .def_property_readonly("series", &LfoRunWrapper::series,
			      "Iterated forecast trajectory, shape (length, dim), "
			      "in original (unscaled) data units");

  lfo_run.def(
      "forecast", &lfo_run_forecast_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("minn") = 30, py::arg("zeroth_order") = false,
      py::arg("flength") = 1000, py::arg("eps0") = 1.e-3, py::arg("epsset") = false,
      py::arg("epsf") = 1.2,
      "Iterate a local-linear (or, if zeroth_order, a zeroth order) forecast\n"
      "forward `flength` steps from `series` (shape (dim, length)), matching\n"
      "the lfo-run CLI's -m/-d/-k/-0/-L/-r/-f options.\n\n"
      "Raises ValueError if series is constant (zero range) for some "
      "component, if the iterated forecast escapes the data region before "
      "completing flength iterations (matching the CLI's own \"Forecast "
      "failed. Escaping data region!\" exit), if dim < 1, embed < 1, or "
      "series is too short for embed/delay.");

  auto lfo_test = m.def_submodule(
      "lfo_test",
      "Average local-linear forecast error at a growing neighborhood size "
      "per point (source_c/lfo-test.c)");

  py::class_<LfoTestWrapper>(lfo_test, "LfoTest")
      .def_property_readonly("comp", &LfoTestWrapper::comp)
      .def_property_readonly("length", &LfoTestWrapper::length,
			      "Length of the input series")
      .def_property_readonly("rms_error", &LfoTestWrapper::rms_error,
			      "Relative forecast error per component, shape (comp,)")
      .def_property_readonly(
	  "individual", &LfoTestWrapper::individual,
	  "Per-point forecast error in original (unscaled) data units, "
	  "shape (comp, length); zero outside the scanned range "
	  "[(embed-1)*delay, min(iterations, length) - step)");

  lfo_test.def(
      "compute", &lfo_test_forecast_binding, py::arg("series"), py::arg("embed") = 2,
      py::arg("delay") = 1, py::arg("minn") = 30, py::arg("step") = 1,
      py::arg("causal") = py::none(), py::arg("iterations") = py::none(),
      py::arg("eps0") = 1.e-3, py::arg("epsset") = false, py::arg("epsf") = 1.2,
      "Estimate the average forecast error of a local-linear fit over "
      "`series` (shape (comp, length)), matching the lfo-test CLI's "
      "-m/-d/-k/-s/-C/-n/-r/-f options.\n\n"
      "causal defaults to step (the CLI's default when -C is not given). "
      "iterations defaults to len(series) (the CLI's default when -n is "
      "not given). If epsset is True, eps0 is interpreted in the original "
      "(raw) data units, matching the CLI's -r; otherwise it is used as-is "
      "in the internally-rescaled [0,1) space.\n\n"
      "Raises ValueError if a component of series is constant (zero "
      "range), if comp < 1, embed < 1, series is empty, or series is too "
      "short for embed/delay/minn.");

  auto nstat_z = m.def_submodule(
      "nstat_z", "Nonstationarity test via zeroth-order forecast error between "
		 "piece pairs (source_c/nstat_z.c)");

  py::class_<NstatZWrapper>(nstat_z, "NstatZ")
      .def_property_readonly("pieces", &NstatZWrapper::pieces)
      .def_property_readonly("n_pairs", &NstatZWrapper::n_pairs)
      .def_property_readonly("first", &NstatZWrapper::first,
			      "0-indexed piece used to fit the zeroth-order model, "
			      "shape (n_pairs,)")
      .def_property_readonly("second", &NstatZWrapper::second,
			      "0-indexed piece the fit is tested against, "
			      "shape (n_pairs,)")
      .def_property_readonly("value", &NstatZWrapper::value,
			      "rms forecast error of the `first` piece's fit on "
			      "`second`'s reference points, divided by `second`'s "
			      "own rms, shape (n_pairs,)");

  nstat_z.def(
      "compute", &nstat_z_compute_binding, py::arg("series"), py::arg("pieces"),
      py::arg("dim") = 3, py::arg("delay") = 1, py::arg("minn") = 30,
      py::arg("step") = 1, py::arg("causal") = py::none(), py::arg("center") = py::none(),
      py::arg("first_window") = py::none(), py::arg("second_window") = py::none(),
      py::arg("first_offset") = py::none(), py::arg("second_offset") = py::none(),
      py::arg("eps0") = 1.e-3, py::arg("epsset") = false, py::arg("epsf") = 1.2,
      "Test `series` (1D) for nonstationarity by splitting it into `pieces` "
      "equal-length segments and, for each selected pair of segments "
      "(first, second), fitting a zeroth-order model on `first` and "
      "measuring its rms forecast error on `second`'s reference points, "
      "scaled by `second`'s own rms - matching the nstat_z CLI's "
      "-#/-m/-d/-k/-s/-C/-n/-r/-f options.\n\n"
      "causal defaults to step (the CLI's default when -C is not given). "
      "center defaults to using every point of a piece as a reference "
      "point, clamped to fit (the CLI's default when -n is not given).\n\n"
      "first_window/second_window are optional length-`pieces` 0/1 "
      "arrays selecting which pieces are candidates for the 'first' "
      "(fitted) and 'second' (tested-against) role, matching the CLI's "
      "-1/-2 options with a plain (non \"+offset\") argument; None means "
      "all pieces (the CLI's default). first_offset/second_offset are the "
      "CLI's -1/-2 in \"+N\" form (a window of pieces within N of the "
      "other index); when given, they take precedence over "
      "first_window/second_window for the corresponding role, exactly "
      "like the CLI.\n\n"
      "If epsset is True, eps0 is interpreted in the original (raw) data "
      "units, matching the CLI's -r; otherwise it is used as-is in the "
      "internally-rescaled [0,1) space.\n\n"
      "Raises ValueError if series is constant (zero range), if some "
      "piece has zero variance, if pieces leaves fewer than minn usable "
      "reference points per piece, if pieces < 1 or dim < 1, or if "
      "first_window/second_window is not a length-`pieces` array.");

  auto ghkss = m.def_submodule(
      "ghkss", "Multivariate noise reduction using the GHKSS algorithm (source_c/ghkss.c)");

  py::class_<GHKSSResultWrapper>(ghkss, "GHKSSResult")
      .def_property_readonly("comp", &GHKSSResultWrapper::comp)
      .def_property_readonly("length", &GHKSSResultWrapper::length,
			      "Length of the input series")
      .def_property_readonly("n_iterations", &GHKSSResultWrapper::n_iterations)
      .def_property_readonly("series", &GHKSSResultWrapper::series,
			      "Corrected series after each iteration, shape "
			      "(n_iterations, comp, length), in original "
			      "(unscaled) data units")
      .def_property_readonly("shift", &GHKSSResultWrapper::shift,
			      "Average shift applied to each component by "
			      "each iteration, shape (n_iterations, comp), "
			      "in original (unscaled) data units")
      .def_property_readonly("rms", &GHKSSResultWrapper::rms,
			      "Average rms size of the correction applied to "
			      "each component by each iteration, shape "
			      "(n_iterations, comp), in original (unscaled) "
			      "data units")
      .def_property_readonly(
	  "mineps_reset", &GHKSSResultWrapper::mineps_reset,
	  "Whether the minimal neighborhood size was halved for the "
	  "following iteration, shape (n_iterations,)")
      .def_property_readonly(
	  "mineps_after", &GHKSSResultWrapper::mineps_after,
	  "Minimal neighborhood size used to start the following "
	  "iteration (or that would have been used, for the last "
	  "iteration), shape (n_iterations,), in original (unscaled) "
	  "data units")
      .def("correction_steps", &GHKSSResultWrapper::correction_steps, py::arg("iteration"),
	   "Returns (epsilon, count) for the given 0-indexed iteration: the "
	   "neighborhood sizes tried while searching for enough neighbors "
	   "to correct every point, and the cumulative number of points "
	   "corrected once each size was reached.")
      .def("trend_steps", &GHKSSResultWrapper::trend_steps, py::arg("iteration"),
	   "Returns (epsilon, count) for the given 0-indexed iteration: the "
	   "neighborhood sizes used while evaluating and removing the "
	   "trend, and the cumulative number of points trend-subtracted "
	   "once each size was reached.");

  ghkss.def(
      "reduce", &ghkss_reduce_binding, py::arg("series"), py::arg("embed") = 5,
      py::arg("delay") = 1, py::arg("qdim") = 2, py::arg("minn") = 50,
      py::arg("mineps") = py::none(), py::arg("iterations") = 1,
      py::arg("euclidean") = false,
      "Performs multivariate noise reduction on `series` (shape (comp, "
      "length)) via the GHKSS algorithm, matching the ghkss CLI's "
      "-m (embedding dim part)/-d/-q/-k/-r/-i/-2 options over `iterations` "
      "passes. Each component of series is independently rescaled to "
      "[0,1) internally (the input array is not modified).\n\n"
      "mineps is the minimal neighborhood size to start with, in original "
      "(raw) data units, matching the CLI's -r; the default of None uses "
      "1/1000 in the internally-rescaled [0,1) space, matching the CLI's "
      "own default (unset -r).\n\n"
      "Raises ValueError if series.shape[1] < minn, if some component of "
      "series is constant (zero range), or if the eigenvalue solver fails "
      "to converge for some point's local correction matrix.");
}
