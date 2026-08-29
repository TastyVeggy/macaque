#include "macaque/Target/BinaryEmitter.hpp"

#include "macaque/Dialect/MacaqueOps.hpp"
#include "macaque/common/isa.hpp"
#include "mlir/IR/Block.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::macaque;
namespace isa = ::macaque::common::isa;

namespace {

FailureOr<uint64_t> encodeInstruction(Operation &op) {
  return TypeSwitch<Operation *, FailureOr<uint64_t>>(&op)
      .Case<LoadWeightOp>([](LoadWeightOp o) {
        return isa::Instruction{isa::Opcode::LoadWeight, o.getDdr3Addr(),
                                o.getByteCount()}
            .encode();
      })
      .Case<LoadBiasOp>([](LoadBiasOp o) {
        return isa::Instruction{isa::Opcode::LoadBias, o.getDdr3Addr(),
                                o.getByteCount()}
            .encode();
      })
      .Case<LoadInputOp>([](LoadInputOp o) {
        return isa::Instruction{isa::Opcode::LoadInput, o.getDdr3Addr(),
                                o.getByteCount(),
                                static_cast<uint8_t>(o.getValidBytesPerRow()),
                                static_cast<uint8_t>(o.getInputRows())}
            .encode();
      })
      .Case<MatmulOp>([](MatmulOp o) {
        return isa::encodeMatmul({o.getAccMode(), o.getWeightHold(),
                                  o.getMatRowBase(),
                                  static_cast<uint8_t>(o.getTileParams())});
      })
      .Case<ActivateOp>([](ActivateOp o) {
        return isa::encodeActivate({static_cast<isa::ActFunc>(o.getActFunc()),
                                    o.getActScaleM(), o.getActBankHold(),
                                    o.getActRowBase(), o.getActScaleShift(),
                                    o.getActNumRows()});
      })
      .Case<StoreOp>([](StoreOp o) {
        return isa::Instruction{isa::Opcode::Store, o.getDdr3Addr(),
                                o.getByteCount()}
            .encode();
      })
      .Case<SyncOp>([](SyncOp) { return isa::encodeOpcode(isa::Opcode::Sync); })
      .Default([](Operation *o) -> FailureOr<uint64_t> {
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
    FailureOr<uint64_t> word = encodeInstruction(op);
    if (failed(word))
      return failure();
    words.push_back(*word);
  }
  return words;
}

} // namespace macaque::codegen::target
