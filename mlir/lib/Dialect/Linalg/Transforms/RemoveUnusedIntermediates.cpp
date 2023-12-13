#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGREMOVEUNUSEDINTERMEDIATES
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

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

namespace {

struct LinalgRemoveUnusedIntermediatesPass
    : public impl::LinalgRemoveUnusedIntermediatesBase<
          LinalgRemoveUnusedIntermediatesPass> {
  void runOnOperation() override;
};

} // namespace

void LinalgRemoveUnusedIntermediatesPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgRemoveUnusedIntermediatesPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void mlir::linalg::populateLinalgRemoveUnusedIntermediatesPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgRemoveUnusedIntermediatesPattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::createRemoveUnusedIntermediatesPass() {
  return std::make_unique<LinalgRemoveUnusedIntermediatesPass>();
}
