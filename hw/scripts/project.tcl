source [file join [file dirname [info script]] "common.tcl"]

file mkdir $build_dir

create_project $proj "$build_dir/vivado" -part $part -force
add_files [glob_recursive "$rtl_dir"]

add_files -fileset constrs_1 [glob "$constrs_dir/*.xdc"]
update_compile_order -fileset sources_1

source [file join [file dirname [info script]] "ip/create_mig.tcl"]
