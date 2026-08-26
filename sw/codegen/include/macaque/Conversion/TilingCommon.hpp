#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "macaque/common/defs.hpp"
#include "macaque/common/isa.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace macaque::codegen::conversion::detail {

namespace macaque_defs = ::macaque::common;
namespace macaque_isa = ::macaque::common::isa;

inline uint32_t byteSizeOf(mlir::RankedTensorType type) {
  return static_cast<uint32_t>(type.getNumElements() *
                               (type.getElementTypeBitWidth() / 8));
}

inline uint32_t alignUp(uint32_t x) { return (x + 7u) & ~7u; }

inline constexpr int64_t kTileWidth = macaque_defs::kArraySize;

inline int64_t numTilesFor(int64_t dim) { return (dim + kTileWidth - 1) / kTileWidth; }

inline constexpr int64_t kBatchChunkRows = kTileWidth;
inline constexpr int64_t kMaxChunksPerBatch =
    macaque_defs::kMaxHoldBatchRows / kBatchChunkRows;
inline constexpr int64_t kMaxBatchRows = macaque_defs::kMaxHoldBatchRows;

// Just makes everything so much more convenient
inline constexpr int64_t kMaxFlatChunkRows = kMaxBatchRows;
static_assert(kMaxFlatChunkRows <= (1 << macaque_isa::kTileParamsBits) - 1,
             "kMaxFlatChunkRows must still fit tile_params' 8-bit encoding");

inline int64_t numFlatChunksFor(int64_t rows) {
  return (rows + kMaxFlatChunkRows - 1) / kMaxFlatChunkRows;
}

// Row count of M-chunk `chunkIdx` of a matmul with `rows` total activation
// rows
inline int64_t flatChunkRows(int64_t rows, int64_t chunkIdx) {
  return std::min(kMaxFlatChunkRows, rows - chunkIdx * kMaxFlatChunkRows);
}

inline int64_t numBatchesFor(int64_t rows) {
  return (rows + kMaxBatchRows - 1) / kMaxBatchRows;
}

// Row count of hold-batch `batchIdx` of a matmul with `rows` total
// activation rows
inline int64_t rowsPerBatch(int64_t rows, int64_t batchIdx) {
  return std::min(kMaxBatchRows, rows - batchIdx * kMaxBatchRows);
}

// Row count of chunk `chunkIdx` within one hold-batch of `batchRows` total
// rows
inline int64_t batchChunkRows(int64_t batchRows, int64_t chunkIdx) {
  return std::min(kBatchChunkRows, batchRows - chunkIdx * kBatchChunkRows);
}

inline uint32_t weightTileBytes(mlir::RankedTensorType bType) {
  return static_cast<uint32_t>(kTileWidth * kTileWidth) *
         (bType.getElementTypeBitWidth() / 8);
}

inline constexpr uint32_t kBiasTileBytes =
    static_cast<uint32_t>(kTileWidth) * static_cast<uint32_t>(sizeof(int32_t));

// One matmul optionally feeding a bias tosa.add, en route to a tosa.rescale
// - the shape RescaleToMacaque/sizeRegions both match against.
struct MatmulChain {
  mlir::tosa::MatMulOp matmul;
  mlir::tosa::AddOp biasAdd;      // null if there's no bias
  mlir::tosa::ConstOp biasConst;  // null if there's no bias
};

inline std::optional<MatmulChain> matchMatmulChain(mlir::Value rescaleInput) {
  if (auto matmul = rescaleInput.getDefiningOp<mlir::tosa::MatMulOp>())
    return MatmulChain{matmul, nullptr, nullptr};

  auto addOp = rescaleInput.getDefiningOp<mlir::tosa::AddOp>();
  if (!addOp) return std::nullopt;

  auto matmul = addOp.getInput1().getDefiningOp<mlir::tosa::MatMulOp>();
  auto biasConst = addOp.getInput2().getDefiningOp<mlir::tosa::ConstOp>();
  if (!matmul) {
    matmul = addOp.getInput2().getDefiningOp<mlir::tosa::MatMulOp>();
    biasConst = addOp.getInput1().getDefiningOp<mlir::tosa::ConstOp>();
  }
  if (!matmul || !biasConst) return std::nullopt;
  return MatmulChain{matmul, addOp, biasConst};
}

// If `rescale`'s one and only use is a tosa.clamp shaped like a quantized
// ReLU (min_val == 0, max_val >= the INT8 max), returns that clamp 
// (to fuse ReLU)
inline std::optional<mlir::tosa::ClampOp> matchReluClamp(mlir::tosa::RescaleOp rescale) {
  if (!rescale->hasOneUse()) return std::nullopt;
  auto clamp = mlir::dyn_cast<mlir::tosa::ClampOp>(*rescale->user_begin());
  if (!clamp) return std::nullopt;
  auto minAttr = mlir::dyn_cast<mlir::IntegerAttr>(clamp.getMinValAttr());
  auto maxAttr = mlir::dyn_cast<mlir::IntegerAttr>(clamp.getMaxValAttr());
  if (!minAttr || !maxAttr) return std::nullopt;
  if (minAttr.getValue().getSExtValue() != 0) return std::nullopt;
  if (maxAttr.getValue().getSExtValue() < 127) return std::nullopt;
  return clamp;
}

inline mlir::Value logicalRescaleOutput(mlir::tosa::RescaleOp rescale) {
  if (auto clamp = matchReluClamp(rescale)) return clamp->getResult();
  return rescale.getOutput();
}

// check if it's an intermediate (layer-to-layer) activation
inline bool feedsMatmul(mlir::tosa::RescaleOp rescale) {
  for (mlir::Operation* user : logicalRescaleOutput(rescale).getUsers())
    if (mlir::isa<mlir::tosa::MatMulOp>(user)) return true;
  return false;
}

inline bool feedsRescale(mlir::tosa::MatMulOp matmul) {
  for (mlir::Operation* user : matmul->getUsers()) {
    if (mlir::isa<mlir::tosa::RescaleOp>(user)) return true;
    if (mlir::isa<mlir::tosa::AddOp>(user))
      for (mlir::Operation* addUser : user->getUsers())
        if (mlir::isa<mlir::tosa::RescaleOp>(addUser)) return true;
  }
  return false;
}

}  // namespace macaque::codegen::conversion::detail
