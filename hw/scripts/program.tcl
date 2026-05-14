source [file join [file dirname [info script]] "common.tcl"]

open_hw_manager
connect_hw_server -url $hw_server
open_hw_target

set hw_device [get_hw_devices $device]
current_hw_device $hw_device

set_property PROGRAM.FILE $bitstream [current_hw_device]
program_hw_devices [current_hw_device]

close_hw_manager
