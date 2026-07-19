add_library(ddtsa STATIC
  get_series.c
  rescale_data.c
  make_box.c
  find_neighbors.c
  scan_help.c
  variance.c
  get_multi_series.c
  search_datafile.c
  check_option.c
  solvele.c
  rand.c
  eigen.c
  test_outfile.c
  invert_matrix.c
  exclude_interval.c
  make_multi_box.c
  find_multi_neighbors.c
  check_alloc.c
  myfgets.c
  what_i_do.c
  make_multi_index.c
  make_multi_box2.c
  rand_arb_dist.c
  normal.c
  store_spec.c
  fft.c
)

target_link_libraries(ddtsa PUBLIC m)
