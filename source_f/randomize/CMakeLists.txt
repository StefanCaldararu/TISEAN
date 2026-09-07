function(add_randomize_variant cost cool perm)
  set(target_name randomize_${cost}_${cool}_${perm})
  add_executable(${target_name}
    ${CMAKE_CURRENT_SOURCE_DIR}/randomize.f
    ${CMAKE_CURRENT_SOURCE_DIR}/cost/${cost}.f
    ${CMAKE_CURRENT_SOURCE_DIR}/cool/${cool}.f
    ${CMAKE_CURRENT_SOURCE_DIR}/perm/${perm}.f
  )
  target_link_libraries(${target_name} PRIVATE tsa sla)
  install(TARGETS ${target_name} RUNTIME DESTINATION bin)
endfunction()

add_randomize_variant(auto      exp random)
add_randomize_variant(autop     exp random)
add_randomize_variant(spikeauto exp random)
add_randomize_variant(spikespec exp event)
add_randomize_variant(uneven    exp random)
# add more calls here for each module you write, e.g.
# add_randomize_variant(mycost exp random)
