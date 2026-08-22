source [file join [file dirname [info script]] "common.tcl"]
source [file join [file dirname [info script]] "top_config.tcl"]

if {![file exists $proj_file]} {
    error "project not found: $proj_file — run 'make synth' first"
}
open_project $proj_file

set gen_bit [file join $proj_dir ${proj}.runs impl_1 ${top_module}.bit]
if {![file exists $gen_bit]} {
    error "bitstream not found: $gen_bit — run 'make impl' first"
}
file copy -force $gen_bit $bitstream

puts "BITSTREAM: $bitstream"
