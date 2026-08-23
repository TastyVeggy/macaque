#ifndef MACAQUE_DIALECT
#define MACAQUE_DIALECT

include "mlir/IR/OpBase.td"

def Macaque_Dialect : Dialect {
  let name = "macaque";
  let cppNamespace = "::macaque";
  let summary = "Macaque NPU instruction dialect";
  let description = [{
    One operation per opcode.
  }];
  let usePropertiesForAttributes = 0;
}

#endif // MACAQUE_DIALECT
