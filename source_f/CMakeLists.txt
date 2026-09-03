add_subdirectory(slatec)

# istdio.f writes error messages to Fortran unit ERRUNIT, which is the unit
# connected to stderr. This isn't standardized pre-Fortran 2003, so probe
# for it the same way the old ./configure script did: compile a one-line
# program per candidate unit and see which one's output lands on stderr.
set(_tisean_errunit 0)
foreach(_iu 0 1 2 3 4 6 7 8 9)
  set(_test_src "${CMAKE_CURRENT_BINARY_DIR}/errunit_test_${_iu}.f")
  file(WRITE "${_test_src}" "      write(${_iu},'(1ht)')\n      end\n")
  try_compile(_compiled "${CMAKE_CURRENT_BINARY_DIR}/errunit_check"
    SOURCES "${_test_src}"
    COPY_FILE "${CMAKE_CURRENT_BINARY_DIR}/errunit_test_${_iu}"
  )
  if(_compiled)
    execute_process(
      COMMAND "${CMAKE_CURRENT_BINARY_DIR}/errunit_test_${_iu}"
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
    )
    if(_err MATCHES "t")
      set(_tisean_errunit ${_iu})
      break()
    endif()
  endif()
endforeach()
set(ERRUNIT ${_tisean_errunit})
message(STATUS "Fortran stderr unit detected as ${ERRUNIT}")

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/istdio_temp.f" _istdio_content)
string(REPLACE "ERRUNIT" "${ERRUNIT}" _istdio_content "${_istdio_content}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/istdio.f" "${_istdio_content}")

add_library(tsa STATIC
  readfile.f
  xreadfile.f
  arguments.f
  commandline.f
  any_s.f
  "${CMAKE_CURRENT_BINARY_DIR}/istdio.f"
  help.f
  verbose.f
  d1.f
  neigh.f
  rank.f
  tospec.f
)

# normal.f, store_spec.f and nmore.f have been ported to C
# (source_c/routines/normal.c, store_spec.c, nmore.c); the Fortran routines
# still call them as `normal`/`rms`/`minmax`/`store_spec`/`nmore`/`nless`, so
# link the C library that now provides those symbols into every Fortran
# binary.
target_link_libraries(tsa PUBLIC ddtsa)

set(TISEAN_F_PROGRAMS
  c1 c2naive xc2
  c2d c2g c2t
  pc predict stp
  lazy project addnoise compare upo upoembed cluster
  choose rms notch autocor spectrum wiener1 wiener2
  surrogates endtoend timerev
  events intervals spikespec spikeauto
  ikeda lorenz xrecur
)

foreach(prog ${TISEAN_F_PROGRAMS})
  add_executable(${prog} ${prog}.f)
  target_link_libraries(${prog} PRIVATE tsa sla)
  install(TARGETS ${prog} RUNTIME DESTINATION bin)
endforeach()

add_subdirectory(randomize)
