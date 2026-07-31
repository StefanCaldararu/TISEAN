add_subdirectory(routines)

set(TISEAN_C_PROGRAMS
  poincare extrema rescale recurr mutual false_nearest
  lyap_r lyap_k lyap_spec d2 av-d2 makenoise nrlazy low121
  lzo-test lfo-run lfo-test rbf polynom polyback polynomp polypar
  mem_spec pca ghkss lfo-ar xzero xcor boxcount fsle
  resample histogram nstat_z sav_gol delay lzo-gm arima-model
  lzo-run ar-run
)

foreach(prog ${TISEAN_C_PROGRAMS})
  add_executable(${prog} ${prog}.c)
  target_link_libraries(${prog} PRIVATE ddtsa)
  install(TARGETS ${prog} RUNTIME DESTINATION bin)
endforeach()

# ar-model uses the reentrant API in api/ar_model_api.c (also used by the
# Python bindings under python/) instead of duplicating the math inline.
add_executable(ar-model ar-model.c api/ar_model_api.c)
target_link_libraries(ar-model PRIVATE ddtsa)
install(TARGETS ar-model RUNTIME DESTINATION bin)

# corr uses the reentrant API in api/corr_api.c (also used by the Python
# bindings under python/) instead of duplicating the math inline.
add_executable(corr corr.c api/corr_api.c)
target_link_libraries(corr PRIVATE ddtsa)
install(TARGETS corr RUNTIME DESTINATION bin)
