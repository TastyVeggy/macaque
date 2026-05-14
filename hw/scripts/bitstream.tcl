source [file join [file dirname [info script]] "common.tcl"]

file mkdir $build_dir

open_checkpoint $post_route
write_bitstream -force $bitstream
