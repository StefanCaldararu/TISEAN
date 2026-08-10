cmake_minimum_required(VERSION 3.15)
project(tisean_python LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(pybind11 CONFIG REQUIRED)

# Repo root: python/ is directly under it.
set(TISEAN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Only the sources ar-model/low121/histogram/polypar/corr/xcor/av-d2/mutual/
# extrema/recurr/xzero/mem_spec/lyap_r/makenoise/sav_gol/lzo-gm/polynomp/fsle/
# false_nearest/pca/delay/lyap_k/lzo-test/boxcount/nrlazy/rbf/lfo-ar/lzo-run/
# lfo-run/lfo-test/nstat_z/ghkss/lyap_spec/d2 need. Other routines get added
# here as they get their own reentrant API extracted.
pybind11_add_module(_tisean
  src/bindings.cpp
  ${TISEAN_ROOT}/source_c/api/d2_api.c
  ${TISEAN_ROOT}/source_c/api/ar_model_api.c
  ${TISEAN_ROOT}/source_c/api/boxcount_api.c
  ${TISEAN_ROOT}/source_c/api/low121_api.c
  ${TISEAN_ROOT}/source_c/api/histogram_api.c
  ${TISEAN_ROOT}/source_c/api/polypar_api.c
  ${TISEAN_ROOT}/source_c/api/corr_api.c
  ${TISEAN_ROOT}/source_c/api/xcor_api.c
  ${TISEAN_ROOT}/source_c/api/av-d2_api.c
  ${TISEAN_ROOT}/source_c/api/mutual_api.c
  ${TISEAN_ROOT}/source_c/api/extrema_api.c
  ${TISEAN_ROOT}/source_c/api/xzero_api.c
  ${TISEAN_ROOT}/source_c/api/recurr_api.c
  ${TISEAN_ROOT}/source_c/api/mem_spec_api.c
  ${TISEAN_ROOT}/source_c/api/lyap_r_api.c
  ${TISEAN_ROOT}/source_c/api/makenoise_api.c
  ${TISEAN_ROOT}/source_c/api/sav_gol_api.c
  ${TISEAN_ROOT}/source_c/api/lzo-gm_api.c
  ${TISEAN_ROOT}/source_c/api/polynomp_api.c
  ${TISEAN_ROOT}/source_c/api/fsle_api.c
  ${TISEAN_ROOT}/source_c/api/false_nearest_api.c
  ${TISEAN_ROOT}/source_c/api/pca_api.c
  ${TISEAN_ROOT}/source_c/api/delay_api.c
  ${TISEAN_ROOT}/source_c/api/lyap_k_api.c
  ${TISEAN_ROOT}/source_c/api/lzo-test_api.c
  ${TISEAN_ROOT}/source_c/api/nrlazy_api.c
  ${TISEAN_ROOT}/source_c/api/rbf_api.c
  ${TISEAN_ROOT}/source_c/api/lfo-ar_api.c
  ${TISEAN_ROOT}/source_c/api/lzo-run_api.c
  ${TISEAN_ROOT}/source_c/api/lfo-run_api.c
  ${TISEAN_ROOT}/source_c/api/lfo-test_api.c
  ${TISEAN_ROOT}/source_c/api/nstat_z_api.c
  ${TISEAN_ROOT}/source_c/api/ghkss_api.c
  ${TISEAN_ROOT}/source_c/api/lyap_spec_api.c
  ${TISEAN_ROOT}/source_c/routines/check_alloc.c
  ${TISEAN_ROOT}/source_c/routines/variance.c
  ${TISEAN_ROOT}/source_c/routines/invert_matrix.c
  ${TISEAN_ROOT}/source_c/routines/solvele.c
  ${TISEAN_ROOT}/source_c/routines/rand.c
  ${TISEAN_ROOT}/source_c/routines/make_box.c
  ${TISEAN_ROOT}/source_c/routines/find_neighbors.c
  ${TISEAN_ROOT}/source_c/routines/make_multi_box.c
  ${TISEAN_ROOT}/source_c/routines/make_multi_box2.c
  ${TISEAN_ROOT}/source_c/routines/make_multi_index.c
  ${TISEAN_ROOT}/source_c/routines/find_multi_neighbors.c
  ${TISEAN_ROOT}/source_c/routines/exclude_interval.c
  ${TISEAN_ROOT}/source_c/routines/eigen.c
)

target_include_directories(_tisean PRIVATE ${TISEAN_ROOT}/include)

install(TARGETS _tisean DESTINATION tisean)
