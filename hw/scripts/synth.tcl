source [file join [file dirname [info script]] "project.tcl"]

launch_runs synth_1 -jobs 7
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "SYNTHESIS FAILED. See synth_1 run log for '$top_module'"
}

puts "SYNTH DONE: $top_module ($proj_dir)"
