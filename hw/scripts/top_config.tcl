array set TOP_XDC {
    uart_top          {common.xdc uart.xdc}
    uart_ddr3_top     {common.xdc uart.xdc led.xdc}
    uart_mmio_top     {common.xdc uart.xdc led.xdc}
    npu_top           {common.xdc uart.xdc led.xdc}
}

array set TOP_IPS {
    uart_top          {}
    uart_ddr3_top     {create_clk_wiz.tcl create_mig.tcl}
    uart_mmio_top     {create_clk_wiz.tcl create_mig.tcl}
    npu_top           {create_clk_wiz.tcl create_mig.tcl}
}
