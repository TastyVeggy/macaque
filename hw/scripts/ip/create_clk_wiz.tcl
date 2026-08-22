set clkwiz_dir "$build_dir/ip/clk_wiz_0"

file delete -force $clkwiz_dir
file mkdir $clkwiz_dir

create_ip \
    -vendor xilinx.com \
    -library ip \
    -version 6.0 \
    -name clk_wiz \
    -module_name clk_wiz_0 \
    -dir $clkwiz_dir

set_property -dict [list \
    CONFIG.PRIM_IN_FREQ                 {50.000} \
    CONFIG.PRIM_SOURCE                  {Single_ended_clock_capable_pin} \
    CONFIG.CLKIN1_JITTER_PS             {200.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ   {200.000} \
    CONFIG.CLKOUT1_DRIVES               {BUFG} \
    CONFIG.USE_LOCKED                   {true} \
    CONFIG.USE_RESET                    {true} \
] [get_ips clk_wiz_0]

generate_target all [get_ips clk_wiz_0]

puts "CLK WIZ: regenerated clk_wiz_0"
