# DDR3 memory layout

This is a **software convention, not something the hardware enforces**. The compiler
(`sw/codegen`) and the runtime (`sw/runtime`) have to independently agree on the same layout
rules to interoperate at all, so this document sets the memory layout convention.

## Layout

Regions are **computed, not fixed** so each region starts right after the previous one ends,
aligned up to the next 8-byte boundary (the AXI4 beat size the DMA burst engine operates on):

```
weight_base    = 0x0000_1000
bias_base      = align_up(weight_base + weights_bytes, 8)
input_base     = align_up(bias_base   + biases_bytes,  8)
scratch_a_base = align_up(input_base  + input_bytes,   8)
scratch_b_base = align_up(scratch_a_base + scratch_bytes, 8)
output_base    = align_up(scratch_b_base + scratch_bytes, 8)
```


| Region | Contents | Notes |
|---|---|---|
| Weights | all layer weight tiles, concatenated in load order | |
| Biases | all layer bias tiles, concatenated in load order | |
| Input activations | the staged input tensor(s) |  |
| Scratch A / Scratch B | intermediate (layer-to-layer) activations | double-buffered, ping-pong |
| Output | final result tensor(s) | read back by the host after the run completes |
