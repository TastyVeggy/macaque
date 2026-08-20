set dma_dir "$build_dir/ip/axi_dma_0"

file mkdir $dma_dir

create_ip \
    -vendor xilinx.com \
    -library ip \
    -version 7.1 \
    -name axi_dma \
    -module_name axi_dma_0 \
    -dir $dma_dir

set_property -dict [list \
    CONFIG.c_include_mm2s             {1} \
    CONFIG.c_include_s2mm             {1} \
    CONFIG.c_include_sg               {0} \
    CONFIG.c_m_axi_mm2s_data_width    {64} \
    CONFIG.c_m_axi_s2mm_data_width    {64} \
    CONFIG.c_mm2s_burst_size          {256} \
    CONFIG.c_s2mm_burst_size          {256} \
] [get_ips axi_dma_0]

# TODO(dma-integration): address width is left at the IP default (c_addr_width
# = 32, the minimum). The plan is for the AXI interconnect to narrow this to
# MIG's 28-bit (256 MB) address space so only the low 28 bits reach DDR3

generate_target all [get_ips axi_dma_0]

puts "AXI DMA: created and generated axi_dma_0"
