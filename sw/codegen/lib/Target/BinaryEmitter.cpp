#include "macaque/Target/BinaryEmitter.h"

#include "llvm/ADT/TypeSwitch.h"
#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/common/isa.hpp"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace mlir::macaque;
namespace isa = ::macaque::common::isa;

namespace {

FailureOr<isa::Instruction> toInstruction(Operation& op) {
  return TypeSwitch<Operation*, FailureOr<isa::Instruction>>(&op)
      .Case<LoadWeightOp>([](LoadWeightOp o) {
        return isa::Instruction{isa::Opcode::LoadWeight,
                                /*acc_mode=*/false,
                                /*target=*/0,
                                o.getDdr3Addr(),
                                o.getByteCount(),
                                /*tile_params=*/0};
      })
      .Case<LoadBiasOp>([](LoadBiasOp o) {
        return isa::Instruction{isa::Opcode::LoadBias, /*acc_mode=*/false,
                                /*target=*/0,          o.getDdr3Addr(),
                                o.getByteCount(),      /*tile_params=*/0};
      })
      .Case<LoadInputOp>([](LoadInputOp o) {
        return isa::Instruction{isa::Opcode::LoadInput, /*acc_mode=*/false,
                                /*target=*/0,           o.getDdr3Addr(),
                                o.getByteCount(),       /*tile_params=*/0};
      })
      .Case<MatmulOp>([](MatmulOp o) {
        return isa::Instruction{isa::Opcode::Matmul, o.getAccMode(),
                                /*target=*/0,        /*ddr3_addr=*/0,
                                /*byte_count=*/0,    o.getTileParams()};
      })
      .Case<ActivateOp>([](ActivateOp o) {
        return isa::Instruction{isa::Opcode::Activate, /*acc_mode=*/false,
                                o.getActFunc(),        o.getActScaleM(),
                                o.getActScaleShift(),  o.getActNumRows()};
      })
      .Case<StoreOp>([](StoreOp o) {
        return isa::Instruction{isa::Opcode::Store, /*acc_mode=*/false,
                                /*target=*/0,       o.getDdr3Addr(),
                                o.getByteCount(),   /*tile_params=*/0};
      })
      .Case<SyncOp>([](SyncOp) {
        return isa::Instruction{isa::Opcode::Sync, /*acc_mode=*/false,
                                /*target=*/0,      /*ddr3_addr=*/0,
                                /*byte_count=*/0,  /*tile_params=*/0};
      })
      .Default([](Operation* o) -> FailureOr<isa::Instruction> {
        return o->emitOpError(
            "unsupported op for binary emission - not a "
            "macaque instruction op");
      });
}

}  // namespace

namespace macaque::codegen::target {

FailureOr<std::vector<uint64_t>> emitBinary(Block& block) {
  std::vector<uint64_t> words;
  words.reserve(block.getOperations().size());
  for (Operation& op : block) {
    FailureOr<isa::Instruction> instr = toInstruction(op);
    if (failed(instr)) return failure();
    words.push_back(instr->encode());
  }
  return words;
}

}  // namespace macaque::codegen::target
