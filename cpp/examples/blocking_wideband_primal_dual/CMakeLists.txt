add_example(wideband_pd_inpainting inpainting.cc LIBRARIES psi tools_for_tests LABELS wideband_pd)
set_target_properties(example_wideband_pd_inpainting PROPERTIES OUTPUT_NAME inpainting)
add_example(wideband_pd_reweighted NOTEST reweighted.cc LIBRARIES psi tools_for_tests LABELS wideband_pd)
set_target_properties(example_wideband_pd_reweighted PROPERTIES OUTPUT_NAME reweighted)
