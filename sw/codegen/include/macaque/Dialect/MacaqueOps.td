#ifndef MACAQUE_OPS
#define MACAQUE_OPS

include "macaque/Dialect/MacaqueDialect.td"
include "mlir/IR/OpBase.td"

class Macaque_Op<string mnemonic, list<Trait> traits = []> :
    Op<Macaque_Dialect, mnemonic, traits>;

def Macaque_LoadWeightOp : Macaque_Op<"load_weight"> {
  let summary = "Load a weight tile from DDR3 into the on-chip weight buffer";
  let description = [{
    Corresponds to OP_LOAD_WEIGHT (opcode 0x0).
    ddr3_addr occupies instruction bits [55:28] (28 bits) and byte_count
    occupies [27:12] (16 bits).
  }];

  let arguments = (ins
    UI32Attr:$ddr3_addr,
    UI16Attr:$byte_count
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Macaque_LoadBiasOp : Macaque_Op<"load_bias"> {
  let summary = "Load a bias tile from DDR3 into the on-chip bias buffer";
  let description = [{
    Corresponds to OP_LOAD_BIAS (opcode 0x1).
    ddr3_addr occupies instruction bits [55:28] (28 bits) and byte_count
    occupies [27:12] (16 bits).
  }];

  let arguments = (ins
    UI32Attr:$ddr3_addr,
    UI16Attr:$byte_count
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Macaque_LoadInputOp : Macaque_Op<"load_input"> {
  let summary = "Load an activation (input) tile from DDR3 into the on-chip activation buffer";
  let description = [{
    Corresponds to OP_LOAD_INPUT (opcode 0x2).
    ddr3_addr occupies instruction bits [55:28] (28 bits) and byte_count
    occupies [27:12] (16 bits).
  }];

  let arguments = (ins
    UI32Attr:$ddr3_addr,
    UI16Attr:$byte_count
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Macaque_MatmulOp : Macaque_Op<"matmul"> {
  let summary = "Run the systolic array over the currently-loaded weight/input tiles";
  let description = [{
    Corresponds to OP_MATMUL (opcode 0x3). Reads whatever is currently in the
    on-chip weight and activation buffers (loaded by prior load_weight/
    load_input ops) and produces INT32 results.
    acc_mode occupies bit [59] and tile_params occupies bits [11:0]
    (12 bits). weight_hold occupies target bit [56] (target[0]): when set,
    this matmul reuses the currently-loaded weight/bias bank instead of a
    freshly-DMA'd one. mat_row_base occupies
    ddr3_addr[7:0] wihch sets the out_buffer row this
    M-chunk's accumulator starts at.
  }];

  let arguments = (ins
    UI1Attr:$acc_mode,
    UI16Attr:$tile_params,
    DefaultValuedAttr<UI1Attr, "false">:$weight_hold,
    DefaultValuedAttr<UI8Attr, "0">:$mat_row_base
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Macaque_ActivateOp : Macaque_Op<"activate"> {
  let summary = "Requantize INT32 results to INT8, applying an activation function";
  let description = [{
    Corresponds to OP_ACTIVATE (opcode 0x4).
  }];

  let arguments = (ins
    UI8Attr:$act_func,
    UI32Attr:$act_scale_m,
    UI8Attr:$act_scale_shift,
    UI8Attr:$act_num_rows,
    DefaultValuedAttr<UI8Attr, "0">:$act_row_base,
    DefaultValuedAttr<UI1Attr, "false">:$act_bank_hold
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Macaque_StoreOp : Macaque_Op<"store"> {
  let summary = "Write the INT8 result tile from the on-chip output buffer back to DDR3";
  let description = [{
    Corresponds to OP_STORE (opcode 0x5).
    ddr3_addr occupies instruction bits [55:28] (28 bits) and byte_count
    occupies [27:12] (16 bits).
  }];

  let arguments = (ins
    UI32Attr:$ddr3_addr,
    UI16Attr:$byte_count
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Macaque_SyncOp : Macaque_Op<"sync"> {
  let summary = "Barrier between the DMA and compute lanes";
  let description = [{
    Corresponds to OP_SYNC (opcode 0x6).
  }];

  let assemblyFormat = "attr-dict";
}

#endif // MACAQUE_OPS
