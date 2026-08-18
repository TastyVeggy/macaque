set mig_prj  "$hw_dir/ip/mig_7series_0/mig.prj"
set mig_dir  "$build_dir/ip/mig_7series_0"

if {[file exists $mig_prj]} {
    file mkdir $mig_dir
    create_ip \
        -vendor xilinx.com \
        -library ip \
        -version 4.2 \
        -name mig_7series \
        -module_name mig_7series_0 \
        -dir $mig_dir

    set_property -dict [list \
        CONFIG.XML_INPUT_FILE $mig_prj \
    ] [get_ips mig_7series_0]

    generate_target all [get_ips mig_7series_0]
    puts "MIG: created and generated mig_7series_0 from $mig_prj"
} else {
    error "$mig_prj not found — cannot create MIG IP"
}
