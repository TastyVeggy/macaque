#include "macaque/Target/BinaryEmitter.hpp"

#include "macaque/Dialect/MacaqueOps.hpp"
#include "macaque/common/isa.hpp"
#include "mlir/IR/Block.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::macaque;
namespace isa = ::macaque::common::isa;

namespace {

FailureOr<isa::Instruction> toInstruction(Operation &op) {
  return TypeSwitch<Operation *, FailureOr<isa::Instruction>>(&op)
      .Case<LoadWeightOp>([](LoadWeightOp o) {
        return isa::Instruction{isa::Opcode::LoadWeight,
                                /*acc_mode=*/false,
                                /*target=*/0,
                                o.getDdr3Addr(),
                                o.getByteCount(),
                                /*reserved=*/0,
                                /*tile_params=*/0};
      })
      .Case<LoadBiasOp>([](LoadBiasOp o) {
        return isa::Instruction{isa::Opcode::LoadBias, /*acc_mode=*/false,
                                /*target=*/0,          o.getDdr3Addr(),
                                o.getByteCount(),      /*reserved=*/0,
                                /*tile_params=*/0};
      })
      .Case<LoadInputOp>([](LoadInputOp o) {
        return isa::Instruction{isa::Opcode::LoadInput, /*acc_mode=*/false,
                                /*target=*/0,           o.getDdr3Addr(),
                                o.getByteCount(),       /*reserved=*/0,
                                /*tile_params=*/0};
      })
      .Case<MatmulOp>([](MatmulOp o) {
        return isa::Instruction{
            isa::Opcode::Matmul,
            o.getAccMode(),
            /*target=*/static_cast<uint8_t>(o.getWeightHold() ? 1 : 0),
            /*ddr3_addr=*/o.getMatRowBase(),
            /*byte_count=*/0,
            /*reserved=*/0,
            static_cast<uint8_t>(o.getTileParams())};
      })
      .Case<ActivateOp>([](ActivateOp o) {
        const uint16_t byteCount =
            static_cast<uint16_t>(o.getActScaleShift() & 0x1F) |
            static_cast<uint16_t>((o.getActRowBase() & 0xFF) << 5) |
            static_cast<uint16_t>((o.getActBankHold() ? 1 : 0) << 13);
        const uint32_t ddr3Addr = (o.getActScaleM() & 0x1FFFFu) << 11;
        return isa::Instruction{
            isa::Opcode::Activate,
            /*acc_mode=*/false,    o.getActFunc(),   ddr3Addr, byteCount,
            /*reserved=*/0,        o.getActNumRows()};
      })
      .Case<StoreOp>([](StoreOp o) {
        return isa::Instruction{isa::Opcode::Store, /*acc_mode=*/false,
                                /*target=*/0,       o.getDdr3Addr(),
                                o.getByteCount(),   /*reserved=*/0,
                                /*tile_params=*/0};
      })
      .Case<SyncOp>([](SyncOp) {
        return isa::Instruction{isa::Opcode::Sync, /*acc_mode=*/false,
                                /*target=*/0,      /*ddr3_addr=*/0,
                                /*byte_count=*/0,  /*reserved=*/0,
                                /*tile_params=*/0};
      })
      .Default([](Operation *o) -> FailureOr<isa::Instruction> {
        return o->emitOpError("unsupported op for binary emission - not a "
                              "macaque instruction op");
      });
}

} // namespace

namespace macaque::codegen::target {

FailureOr<std::vector<uint64_t>> emitBinary(Block &block) {
  std::vector<uint64_t> words;
  words.reserve(block.getOperations().size());
  for (Operation &op : block) {
    FailureOr<isa::Instruction> instr = toInstruction(op);
    if (failed(instr))
      return failure();
    words.push_back(instr->encode());
  }
  return words;
}

} // namespace macaque::codegen::target
