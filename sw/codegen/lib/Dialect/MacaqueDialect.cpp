#include "macaque/Dialect/MacaqueDialect.h"

#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/Dialect/MacaqueOpsDialect.cpp.inc"

void macaque::MacaqueDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "macaque/Dialect/MacaqueOps.cpp.inc"
      >();
}
