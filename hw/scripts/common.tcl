set root_dir [lindex $argv 0]
set top_module [lindex $argv 1]

set hw_dir   "$root_dir/hw"
set build_dir "$hw_dir/build"
set rtl_dir "$hw_dir/rtl"
set constrs_dir "$hw_dir/constrs"

set hw_server "localhost:3121"

set part  "xc7a100tfgg676-1"
# for get_hw_devices
set device "xc7a100t_0"
set proj  "macaque"

set post_synth "$build_dir/${proj}_${top_module}_post_synth.dcp"
set post_route "$build_dir/${proj}_${top_module}_post_route.dcp"
set timing_sum "$build_dir/${proj}_${top_module}_timing_summary.txt"
set bitstream "$build_dir/${proj}_${top_module}.bit"

set proj_dir  "$build_dir/${proj}_${top_module}"
set proj_file [file join $proj_dir ${proj}.xpr]

source "$hw_dir/scripts/utils.tcl"
