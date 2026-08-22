array set TOP_XDC {
    npu_top           {common.xdc}
    uart_top          {common.xdc uart.xdc}
    uart_ddr3_top     {common.xdc uart.xdc led.xdc}
    uart_mmio_top     {common.xdc uart.xdc led.xdc}
}

array set TOP_IPS {
    npu_top           {}
    uart_top          {}
    uart_ddr3_top     {create_clk_wiz.tcl create_mig.tcl}
    uart_mmio_top     {create_clk_wiz.tcl create_mig.tcl}
}
