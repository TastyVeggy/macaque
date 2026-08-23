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

#endif // MACAQUE_OPS
