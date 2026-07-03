cmake_minimum_required(VERSION 3.16)
project(TISEAN LANGUAGES C Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The old ./configure only ever tried "-O" for FFLAGS (never -O2/-O3), so
# match that here: several of the routines (e.g. henon, lorenz) are chaotic
# systems whose output is sensitive to floating-point rounding, and building
# them at -O3 makes their regression tests diverge from the recorded
# reference output.
set(CMAKE_Fortran_FLAGS_RELEASE "-O" CACHE STRING "" FORCE)

# All programs are collected in <source>/bin, matching the historical
# BIN_DIR of the autotools build (and what tests/ expects).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(source_c)
add_subdirectory(source_f)
