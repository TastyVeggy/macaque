set root_dir [lindex $argv 0]
set hw_dir   "$root_dir/hw"
set build_dir "$hw_dir/build"
set rtl_dir "$hw_dir/rtl"
set constrs_dir "$hw_dir/constrs"

set top_module "uart_top"

set hw_server "localhost:3121"

set part  "xc7a100tfgg676-1"
# for get_hw_devices
set device "xc7a100t_0"
set proj  "macaque"

set post_synth "$build_dir/post_synth.dcp"
set post_route "$build_dir/post_route.dcp"
set timing_sum "$build_dir/timing_summary.txt"
set bitstream "$build_dir/$proj.bit"

source "$hw_dir/scripts/utils.tcl"
