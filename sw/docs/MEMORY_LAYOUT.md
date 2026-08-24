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
zero_bias_base = align_up(bias_base   + biases_bytes,  8)
input_base     = align_up(zero_bias_base + zero_bias_bytes, 8)
scratch_a_base = align_up(input_base  + input_bytes,   8)
scratch_b_base = align_up(scratch_a_base + scratch_bytes, 8)
output_base    = align_up(scratch_b_base + scratch_bytes, 8)
```

For a region holding more than one item (Weights, Biases, Input, Output), `align_up` applies **per item**, not once to the region's raw total so each item's own end gets rounded up to the next 8-byte boundary before the next item starts.

| Region | Contents | Notes |
|---|---|---|
| Weights | all layer weight tiles, concatenated in load order | |
| Biases | all layer bias tiles, concatenated in load order | |
| Zero-bias | a single always-zero slot | see below |
| Input activations | the staged input tensor(s) |  |
| Scratch A / Scratch B | intermediate (layer-to-layer) activations | double-buffered, ping-pong |
| Output | final result tensor(s) | read back by the host after the run completes |

### Zero-bias slot

`acc_mode=0` always seeds the systolic array's accumulator from the bias buffer - the ISA has
no "seed from zero" option. A matmul with no real bias (no `tosa.add` in the source model)
still needs *something* loaded there, or it silently inherits whatever a previous layer's
`load_bias` last put in the buffer. Rather than a bias-clear opcode (which doesn't exist), the
compiler points every no-bias matmul's `load_bias` at one shared, fixed, always-zero slot which is
sized to the largest single no-bias matmul's output-channel count. The runtime must stage this region as zero once, the same way it
stages any other region as it is not implicitly zeroed by hardware.
