cmake_minimum_required(VERSION 3.15)
project(tisean_python LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(pybind11 CONFIG REQUIRED)

# Repo root: python/ is directly under it.
set(TISEAN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Sources needed by ar-model and polypar. Other routines get added here as
# they get their own reentrant API extracted.
pybind11_add_module(_tisean
  src/bindings.cpp
  ${TISEAN_ROOT}/source_c/api/ar_model_api.c
  ${TISEAN_ROOT}/source_c/api/polypar_api.c
  ${TISEAN_ROOT}/source_c/routines/check_alloc.c
  ${TISEAN_ROOT}/source_c/routines/variance.c
  ${TISEAN_ROOT}/source_c/routines/invert_matrix.c
  ${TISEAN_ROOT}/source_c/routines/solvele.c
  ${TISEAN_ROOT}/source_c/routines/rand.c
)

target_include_directories(_tisean PRIVATE ${TISEAN_ROOT}/include)

install(TARGETS _tisean DESTINATION tisean)
