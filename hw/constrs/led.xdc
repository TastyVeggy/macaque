# status LEDs (active-low: lit when driven 0).
# LED1 = G20, LED0 = G21
set_property PACKAGE_PIN G20 [get_ports led_calib]
set_property IOSTANDARD LVCMOS33 [get_ports led_calib]

set_property PACKAGE_PIN G21 [get_ports led_locked]
set_property IOSTANDARD LVCMOS33 [get_ports led_locked]
