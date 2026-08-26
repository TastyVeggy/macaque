source [file join [file dirname [info script]] "common.tcl"]
source [file join [file dirname [info script]] "top_config.tcl"]

create_project $proj "$proj_dir" -part $part -force

add_files [glob_recursive "$rtl_dir"]

foreach f $TOP_XDC($top_module) {
    add_files -fileset constrs_1 [file join $constrs_dir $f]
}

foreach ip $TOP_IPS($top_module) {
    source [file join [file dirname [info script]] "ip" $ip]
}

set_property top $top_module [current_fileset]
update_compile_order -fileset sources_1

puts "PROJECT: created $proj_file (top=$top_module)"
