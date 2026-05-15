set_property PACKAGE_PIN M21 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets -of_objects [get_ports clk]]
create_clock -period 20.000 [get_ports clk]

set_property PACKAGE_PIN F3 [get_ports rx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports rx_pin]

set_property PACKAGE_PIN E3 [get_ports tx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports tx_pin]

set_property PACKAGE_PIN H7 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
