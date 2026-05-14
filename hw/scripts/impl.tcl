source [file join [file dirname [info script]] "common.tcl"]

file mkdir $build_dir

read_checkpoint $post_synth

link_design -top $top_module -part $part

read_xdc [glob "$constrs_dir/*.xdc"]

opt_design
place_design
route_design

write_checkpoint -force $post_route
report_timing_summary -file $timing_sum


