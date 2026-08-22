array set TOP_XDC {
    npu_top           {common.xdc}
    uart_top          {common.xdc uart.xdc}
    ddr3_uart_top     {common.xdc uart.xdc led.xdc}
}

array set TOP_IPS {
    npu_top           {}
    uart_top          {}
    ddr3_uart_top     {create_clk_wiz.tcl create_mig.tcl}
}
