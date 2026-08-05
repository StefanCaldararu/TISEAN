add_subdirectory(routines)

set(TISEAN_C_PROGRAMS
  poincare rescale false_nearest
  lyap_r lyap_k lyap_spec d2 makenoise nrlazy
  lzo-test lfo-run lfo-test rbf polynom polyback polynomp
  pca ghkss lfo-ar boxcount fsle
  resample nstat_z sav_gol delay lzo-gm arima-model
  lzo-run ar-run
)

foreach(prog ${TISEAN_C_PROGRAMS})
  add_executable(${prog} ${prog}.c)
  target_link_libraries(${prog} PRIVATE ddtsa)
  install(TARGETS ${prog} RUNTIME DESTINATION bin)
endforeach()

# ar-model, low121, histogram, polypar, corr, xcor, av-d2, mutual, extrema,
# recurr, xzero, and mem_spec use the reentrant APIs in api/ (also used by
# the Python bindings under python/) instead of duplicating the math inline.
add_executable(ar-model ar-model.c api/ar_model_api.c)
target_link_libraries(ar-model PRIVATE ddtsa)
install(TARGETS ar-model RUNTIME DESTINATION bin)

add_executable(low121 low121.c api/low121_api.c)
target_link_libraries(low121 PRIVATE ddtsa)
install(TARGETS low121 RUNTIME DESTINATION bin)

add_executable(histogram histogram.c api/histogram_api.c)
target_link_libraries(histogram PRIVATE ddtsa)
install(TARGETS histogram RUNTIME DESTINATION bin)

add_executable(polypar polypar.c api/polypar_api.c)
target_link_libraries(polypar PRIVATE ddtsa)
install(TARGETS polypar RUNTIME DESTINATION bin)

add_executable(corr corr.c api/corr_api.c)
target_link_libraries(corr PRIVATE ddtsa)
install(TARGETS corr RUNTIME DESTINATION bin)

add_executable(xcor xcor.c api/xcor_api.c)
target_link_libraries(xcor PRIVATE ddtsa)
install(TARGETS xcor RUNTIME DESTINATION bin)

add_executable(av-d2 av-d2.c api/av-d2_api.c)
target_link_libraries(av-d2 PRIVATE ddtsa)
install(TARGETS av-d2 RUNTIME DESTINATION bin)

add_executable(mutual mutual.c api/mutual_api.c)
target_link_libraries(mutual PRIVATE ddtsa)
install(TARGETS mutual RUNTIME DESTINATION bin)

add_executable(extrema extrema.c api/extrema_api.c)
target_link_libraries(extrema PRIVATE ddtsa)
install(TARGETS extrema RUNTIME DESTINATION bin)

add_executable(xzero xzero.c api/xzero_api.c)
target_link_libraries(xzero PRIVATE ddtsa)
install(TARGETS xzero RUNTIME DESTINATION bin)
add_executable(recurr recurr.c api/recurr_api.c)
target_link_libraries(recurr PRIVATE ddtsa)
install(TARGETS recurr RUNTIME DESTINATION bin)

add_executable(mem_spec mem_spec.c api/mem_spec_api.c)
target_link_libraries(mem_spec PRIVATE ddtsa)
install(TARGETS mem_spec RUNTIME DESTINATION bin)
