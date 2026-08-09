add_subdirectory(routines)

set(TISEAN_C_PROGRAMS
  poincare rescale
  d2
  polynom polyback
  resample arima-model
  ar-run
)

foreach(prog ${TISEAN_C_PROGRAMS})
  add_executable(${prog} ${prog}.c)
  target_link_libraries(${prog} PRIVATE ddtsa)
  install(TARGETS ${prog} RUNTIME DESTINATION bin)
endforeach()

# ar-model, low121, histogram, polypar, corr, xcor, av-d2, mutual, extrema,
# recurr, xzero, mem_spec, lyap_r, makenoise, sav_gol, lzo-gm, polynomp, fsle,
# false_nearest, pca, delay, lyap_k, lzo-test, boxcount, nrlazy, rbf, lfo-ar,
# lzo-run, lfo-run, lfo-test, nstat_z, ghkss, and lyap_spec use the reentrant
# APIs in api/ (also used by the Python bindings under python/) instead of
# duplicating the math inline.
add_executable(ar-model ar-model.c api/ar_model_api.c)
target_link_libraries(ar-model PRIVATE ddtsa)
install(TARGETS ar-model RUNTIME DESTINATION bin)

add_executable(low121 low121.c api/low121_api.c)
target_link_libraries(low121 PRIVATE ddtsa)
install(TARGETS low121 RUNTIME DESTINATION bin)

add_executable(histogram histogram.c api/histogram_api.c)
target_link_libraries(histogram PRIVATE ddtsa)
install(TARGETS histogram RUNTIME DESTINATION bin)

add_executable(lyap_spec lyap_spec.c api/lyap_spec_api.c)
target_link_libraries(lyap_spec PRIVATE ddtsa)
install(TARGETS lyap_spec RUNTIME DESTINATION bin)

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

add_executable(lyap_r lyap_r.c api/lyap_r_api.c)
target_link_libraries(lyap_r PRIVATE ddtsa)
install(TARGETS lyap_r RUNTIME DESTINATION bin)

add_executable(makenoise makenoise.c api/makenoise_api.c)
target_link_libraries(makenoise PRIVATE ddtsa)
install(TARGETS makenoise RUNTIME DESTINATION bin)

add_executable(sav_gol sav_gol.c api/sav_gol_api.c)
target_link_libraries(sav_gol PRIVATE ddtsa)
install(TARGETS sav_gol RUNTIME DESTINATION bin)

add_executable(lzo-gm lzo-gm.c api/lzo-gm_api.c)
target_link_libraries(lzo-gm PRIVATE ddtsa)
install(TARGETS lzo-gm RUNTIME DESTINATION bin)

add_executable(polynomp polynomp.c api/polynomp_api.c)
target_link_libraries(polynomp PRIVATE ddtsa)
install(TARGETS polynomp RUNTIME DESTINATION bin)

add_executable(fsle fsle.c api/fsle_api.c)
target_link_libraries(fsle PRIVATE ddtsa)
install(TARGETS fsle RUNTIME DESTINATION bin)

add_executable(false_nearest false_nearest.c api/false_nearest_api.c)
target_link_libraries(false_nearest PRIVATE ddtsa)
install(TARGETS false_nearest RUNTIME DESTINATION bin)

add_executable(pca pca.c api/pca_api.c)
target_link_libraries(pca PRIVATE ddtsa)
install(TARGETS pca RUNTIME DESTINATION bin)

add_executable(delay delay.c api/delay_api.c)
target_link_libraries(delay PRIVATE ddtsa)
install(TARGETS delay RUNTIME DESTINATION bin)

add_executable(lyap_k lyap_k.c api/lyap_k_api.c)
target_link_libraries(lyap_k PRIVATE ddtsa)
install(TARGETS lyap_k RUNTIME DESTINATION bin)

add_executable(lzo-test lzo-test.c api/lzo-test_api.c)
target_link_libraries(lzo-test PRIVATE ddtsa)
install(TARGETS lzo-test RUNTIME DESTINATION bin)

add_executable(boxcount boxcount.c api/boxcount_api.c)
target_link_libraries(boxcount PRIVATE ddtsa)
install(TARGETS boxcount RUNTIME DESTINATION bin)

add_executable(nrlazy nrlazy.c api/nrlazy_api.c)
target_link_libraries(nrlazy PRIVATE ddtsa)
install(TARGETS nrlazy RUNTIME DESTINATION bin)

add_executable(rbf rbf.c api/rbf_api.c)
target_link_libraries(rbf PRIVATE ddtsa)
install(TARGETS rbf RUNTIME DESTINATION bin)

add_executable(lfo-ar lfo-ar.c api/lfo-ar_api.c)
target_link_libraries(lfo-ar PRIVATE ddtsa)
install(TARGETS lfo-ar RUNTIME DESTINATION bin)

add_executable(lzo-run lzo-run.c api/lzo-run_api.c)
target_link_libraries(lzo-run PRIVATE ddtsa)
install(TARGETS lzo-run RUNTIME DESTINATION bin)

add_executable(lfo-run lfo-run.c api/lfo-run_api.c)
target_link_libraries(lfo-run PRIVATE ddtsa)
install(TARGETS lfo-run RUNTIME DESTINATION bin)

add_executable(lfo-test lfo-test.c api/lfo-test_api.c)
target_link_libraries(lfo-test PRIVATE ddtsa)
install(TARGETS lfo-test RUNTIME DESTINATION bin)

add_executable(nstat_z nstat_z.c api/nstat_z_api.c)
target_link_libraries(nstat_z PRIVATE ddtsa)
install(TARGETS nstat_z RUNTIME DESTINATION bin)

add_executable(ghkss ghkss.c api/ghkss_api.c)
target_link_libraries(ghkss PRIVATE ddtsa)
install(TARGETS ghkss RUNTIME DESTINATION bin)
