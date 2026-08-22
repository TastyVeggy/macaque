source [file join [file dirname [info script]] "common.tcl"]
source [file join [file dirname [info script]] "top_config.tcl"]

if {![file exists $proj_file]} {
    error "project not found: $proj_file — run 'make synth' first"
}
open_project $proj_file

launch_runs impl_1 -to_step write_bitstream -jobs 7
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "IMPLEMENTATION/BITGEN FAILED — see impl_1 run log for '$top_module'"
}

puts "IMPL DONE: $top_module ($proj_dir)"
