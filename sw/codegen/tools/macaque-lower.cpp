#include "macaque/Conversion/TosaToMacaque.hpp"
#include "macaque/Dialect/MacaqueDialect.hpp"
#include "macaque/Target/BinaryEmitter.hpp"
#include "macaque/Target/ProgramEmitter.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional,
                  llvm::cl::desc("<input TOSA .mlir file>"),
                  llvm::cl::Required);
static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output JSON file (default: stdout)"),
                   llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "macaque TOSA-to-macaque-ISA lowering tool\n");

  MLIRContext context;
  context.getOrLoadDialect<tosa::TosaDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<mlir::macaque::MacaqueDialect>();

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input =
      openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << inputFilename << ": failed to parse as MLIR\n";
    return 1;
  }

  // Expect exactly one func.func - that function's body is what gets
  // lowered, matching every existing test's single-block-of-matmuls shape.
  func::FuncOp funcOp;
  for (auto op : module->getOps<func::FuncOp>()) {
    if (funcOp) {
      llvm::errs() << inputFilename
                   << ": expected exactly one func.func, found more than one\n";
      return 1;
    }
    funcOp = op;
  }
  if (!funcOp) {
    llvm::errs() << inputFilename << ": no func.func found in module\n";
    return 1;
  }

  Block &body = funcOp.getBody().front();

  ::macaque::codegen::conversion::CompiledProgramInfo info;
  if (failed(::macaque::codegen::conversion::lowerTosaToMacaque(body, &info))) {
    llvm::errs() << inputFilename << ": failed to lower TOSA to macaque\n";
    return 1;
  }

  if (Operation *terminator = body.getTerminator())
    terminator->erase();

  FailureOr<std::vector<uint64_t>> words =
      ::macaque::codegen::target::emitBinary(body);
  if (failed(words)) {
    llvm::errs() << inputFilename << ": failed to emit instruction words\n";
    return 1;
  }

  std::string json = ::macaque::codegen::target::emitProgramJson(*words, info);

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  output->os() << json << "\n";
  output->keep();
  return 0;
}
