/*
 *   This file is part of TISEAN
 *
 *   Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz, Thomas Schreiber
 *
 *   TISEAN is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   TISEAN is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with TISEAN; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Reentrant API for the arima-model routine: fits a multivariate AR model
   and, optionally, refines it into an ARMA model via an iterative
   procedure. Extracted out of source_c/arima-model.c so it can be called
   both from the arima-model CLI and from other bindings (e.g. Python)
   without going through global state, argv parsing, or the process-exiting
   error paths in variance()/rescale_data() that the original helpers it
   used to call (directly, or via rand_arb_dist()) relied on. */

#ifndef _ARIMA_MODEL_H
#define _ARIMA_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dim;
  unsigned long length;
  unsigned int poles;      /* order of the initial AR fit, as requested */
  unsigned int arpoles;    /* AR order of the ARMA refinement, 0 if none was requested */
  unsigned int mapoles;    /* MA order of the ARMA refinement, 0 if none was requested */
  unsigned int is_arima;   /* whether ARMA refinement was requested (and ran) */

  /* Effective history depth of the final model: `poles` for a plain AR fit
     (is_arima == 0), or max(arpoles,mapoles) once ARMA refinement has run
     (is_arima != 0). residuals[*][0..iter_poles) are zero (never written by
     the fit, matching the entries the arima-model CLI never prints either);
     this is also the history depth iterate() needs to seed itself. */
  unsigned int iter_poles;

  /* Number of coefficient columns: dim*poles for a plain AR fit,
     (arpoles+mapoles)*dim once ARMA refinement has run. */
  unsigned int size;

  /* [size]: for coefficient column i, aindex_id[i] is the series row it
     multiplies and aindex_lag[i] is the lag (row[n - aindex_lag[i]]).
     aindex_id[i] < dim means an AR tap on the raw (differenced/centered)
     series; aindex_id[i] >= dim (only possible once is_arima != 0) means an
     MA tap on the running residual estimate, referencing component
     aindex_id[i] - dim. */
  unsigned int *aindex_id;
  unsigned int *aindex_lag;

  double **coeff;       /* [dim][size] */
  double *rms_error;    /* [dim], one-step-ahead RMS forecast error per component. NOTE:
			    if ARMA refinement ran with mapoles large enough that an MA
			    tap's lookback reaches into the first `poles` entries of
			    the running residual estimate (always 0 here - see
			    `residuals` below), this can differ from the arima-model
			    CLI, which reads whatever uninitialized heap memory was
			    there instead; that CLI behavior is undefined, not a
			    contract this reimplementation can or should match. */
  double **residuals;   /* [dim][length]; entries [iter_poles..length-1] are
			    one-step-ahead residuals from the final fit, in
			    the same centering as the input series. Entries
			    before that are zero, EXCEPT when ARMA refinement
			    ran with max(arpoles,mapoles) > poles: then
			    entries [poles..iter_poles) are stale residuals
			    left over from the initial AR(poles) fit, exactly
			    like the CLI's own equivalent (undocumented)
			    output would be in that case. */

  /* Convergence history of the ARMA refinement, one row per iteration
     actually run. NULL (and realiter == 0) if is_arima == 0. */
  unsigned int realiter;
  double **xdiff;       /* [realiter][dim]: per-iteration RMS change of the residuals */
  double *diffcoeff;    /* [realiter]: per-iteration RMS change of coeff[0..dim-1][0..dim-1] */
} ArimaModel;

/* Fits a multivariate AR(`poles`) model to `series` ([dim][length]), the
   same way the arima-model CLI's initial fit does it. series is expected to
   already be differenced (if desired) and centered (zero mean per row) the
   same way the CLI does it via make_difference()/set_averages_to_zero().

   If run_arima is non-zero, the fit is then iteratively refined into an
   ARMA(arpoles,mapoles) model for up to max_iter iterations (stopping early
   once the largest per-component residual RMS change drops below
   convergence), matching the CLI's -P/-I/-e options.

   Returns NULL if poles < 1 or poles >= length, or if run_arima is
   non-zero and (arpoles >= length or mapoles >= length). */
ArimaModel *arima_model_fit(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int poles,
			     int run_arima, unsigned int arpoles,
			     unsigned int mapoles, unsigned int max_iter,
			     double convergence);
void arima_model_free(ArimaModel *model);

/* Iterates the fitted model forward for ilength steps, drawing innovations
   from the empirical distribution of model->residuals (via the same
   arbitrary-distribution resampling the CLI's -s uses) and seeding the
   generator with seed. Returns a newly allocated [ilength][dim] array (free
   with arima_model_iterate_free).

   Returns NULL if every entry of some residual row is identical (the
   resampled distribution would be a point mass), mirroring the process-exiting
   case in rescale_data() (called via the original's rand_arb_dist()) but
   without exiting the process; if bad_value is non-NULL, *bad_value is set
   to that row's constant value (matching what rescale_data()'s "data ranges
   from %e to %e" message would have printed - both %e are that same
   value). */
double **arima_model_iterate(const ArimaModel *model, unsigned long ilength,
			      unsigned long seed, double *bad_value);
void arima_model_iterate_free(double **iterated, unsigned long ilength);

#ifdef __cplusplus
}
#endif

#endif
