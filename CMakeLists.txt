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

# For the same reason the C side must not contract a*b+c into a fused
# multiply-add. GCC and Clang do so by default whenever the target has an
# FMA instruction (always on arm64; on x86-64 only with -march=...), which
# changes the last bit of every dot product and Householder update. The
# Fortran at -O emits no FMAs and the reference outputs were recorded
# without them, so e.g. `project` (through the ported EISPACK rs in
# source_c/routines/rs.c) and `henon` drift from tests/refs on arm64 hosts
# while matching on x86-64 CI. See issue #161.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffp-contract=off>)
endif()

# All programs are collected in <source>/bin, matching the historical
# BIN_DIR of the autotools build (and what tests/ expects).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(source_c)
add_subdirectory(source_f)
