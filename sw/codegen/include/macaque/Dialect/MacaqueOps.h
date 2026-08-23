#pragma once

#include "macaque/Dialect/MacaqueDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "macaque/Dialect/MacaqueOps.h.inc"
