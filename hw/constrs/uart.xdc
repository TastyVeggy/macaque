# uart.xdc — generic CH340 USB-UART pins, shared by any top with a UART.
# (uart_top, axi_ctrl_top, ddr3_uart_top)
set_property PACKAGE_PIN F3 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

set_property PACKAGE_PIN E3 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
