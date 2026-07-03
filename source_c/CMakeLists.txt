add_subdirectory(routines)

set(TISEAN_C_PROGRAMS
  poincare extrema rescale recurr corr mutual false_nearest
  lyap_r lyap_k lyap_spec d2 av-d2 makenoise nrlazy low121
  lzo-test lfo-run lfo-test rbf polynom polyback polynomp polypar
  ar-model mem_spec pca ghkss lfo-ar xzero xcor boxcount fsle
  resample histogram nstat_z sav_gol delay lzo-gm arima-model
  lzo-run
)

foreach(prog ${TISEAN_C_PROGRAMS})
  add_executable(${prog} ${prog}.c)
  target_link_libraries(${prog} PRIVATE ddtsa)
  install(TARGETS ${prog} RUNTIME DESTINATION bin)
endforeach()
