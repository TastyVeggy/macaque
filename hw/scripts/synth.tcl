source [file join [file dirname [info script]] "common.tcl"]

file mkdir $build_dir

read_verilog -sv [glob_recursive "$hw_dir/rtl"]
read_xdc [glob "$constrs_dir/*.xdc"]

synth_design -top $top_module -part $part
write_checkpoint -force $post_synth

