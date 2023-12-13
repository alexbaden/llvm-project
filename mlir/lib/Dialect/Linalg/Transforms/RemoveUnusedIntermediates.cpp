#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGREMOVEUNUSEDINTERMEDIATES
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#if 0
static LogicalResult removeUnusedIntermediatesPrecondition(LinalgOp linalgOp) {
  // Check if the operation has exactly one region.
  if (linalgOp->getNumRegions() != 1) {
    assert(linalgOp->getNumRegions() == 0 && "op with multiple regions");
    // TOD: Otherwise it needs to be built explicitly from the region builder.
    return failure();
  }
  llvm::errs() << "Processing: " << linalgOp << "\n";
  if (!isa<GenericOp>(linalgOp))
    return success();
  return failure();
}

FailureOr<GenericOp>
mlir::linalg::removeUnusedIntermediateOp(RewriterBase &rewriter,
                                         LinalgOp linalgOp) {
  if (failed(removeUnusedIntermediatesPrecondition(linalgOp))) {
    return rewriter.notifyMatchFailure(linalgOp, "preconditions not met");
  }

  llvm::errs() << "Working...\n";
  auto genericOp = dyn_cast<GenericOp>(linalgOp.getOperation());
  assert(genericOp);
  auto genericInputs = genericOp.getInputs();
  llvm::errs() << genericInputs.size() << "\n\n";
  return rewriter.notifyMatchFailure(linalgOp, "pass not finished");
}
#endif

namespace {

LogicalResult removeUnusedIntermediatesInFunc(func::FuncOp func) {
  // OpBuilder builder(func.getBody());
  func.walk([&](linalg::LinalgOp op) {
    if (!isa<linalg::GenericOp>(op))
      return;

    auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
    assert(genericOp);
    auto genericInputs = genericOp.getInputs();

    llvm::errs() << "Processing Op: " << genericOp << "\n";
    llvm::errs() << "Inputs: " << genericInputs.size() << "\n\n";

    // TODO: this didn't work :(
    for (const auto &input : genericInputs) {
      if (input.getDefiningOp() &&
          isa<linalg::GenericOp>(input.getDefiningOp())) {
        llvm::errs() << "## Generic Input : ";
        input.print(llvm::errs());
        llvm::errs() << "\n\n";
      }
    }
  });

  return success();
}

// TODO: this should really be a func pass, but I am copying LinalgKernelCalls
// will just leave it as module for now. The inner func op can
// be lifted and made a func pass later.
class LinalgRemoveUnusedIntermediates
    : public mlir::impl::LinalgRemoveUnusedIntermediatesBase<
          LinalgRemoveUnusedIntermediates> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(removeUnusedIntermediatesInFunc(func))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createRemoveUnusedIntermediatesPass() {
  return std::make_unique<LinalgRemoveUnusedIntermediates>();
}
