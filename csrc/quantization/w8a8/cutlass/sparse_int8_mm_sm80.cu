// 2:4 sparse INT8 GEMM for SM80 to SM89, built by tools/build_sparse_int8_sm80.sh.
//
// Computes D = diag(scale_m) * A * B * diag(scale_n), where A is the compressed s8
// weight (M outputs, K/2 stored per row) and B holds N column-major s8 inputs. D is
// row-major [M, N]: CUTLASS 2.x sparse GEMMs have no column-major output.
#include <cuda_runtime.h>

// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_sparse_with_visitor.h"
#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp"
// clang-format on

using namespace cute;
namespace cu = cutlass;
namespace tb = cutlass::epilogue::threadblock;

namespace {

using ElementD = cu::bfloat16_t;
using TileShape = cu::gemm::GemmShape<128, 128, 128>;
using WarpShape = cu::gemm::GemmShape<64, 64, 128>;
using InstructionShape = cu::gemm::GemmShape<16, 8, 64>;
using ThreadMap = tb::OutputTileThreadLayout<TileShape, WarpShape, float, 4, 1>;

constexpr int kStages = 3;
constexpr int kAlignmentAB = 16;

using StrideD = Stride<int64_t, Int<1>, Int<0>>;
using ScaleM = tb::VisitorColOrScalarBroadcast<ThreadMap, float,
                                               Stride<Int<1>, Int<0>, Int<0>>>;
using ScaleN = tb::VisitorRowOrScalarBroadcast<ThreadMap, float,
                                               Stride<Int<0>, Int<1>, Int<0>>>;
using Mul = tb::VisitorCompute<cu::multiplies, float, float,
                               cu::FloatRoundStyle::round_to_nearest>;
using MulD = tb::VisitorCompute<cu::multiplies, ElementD, float,
                                cu::FloatRoundStyle::round_to_nearest>;
using Evt0 = tb::Sm80EVT<Mul, ScaleN, tb::VisitorAccFetch>;
using Evt1 = tb::Sm80EVT<MulD, ScaleM, Evt0>;
using Store = tb::VisitorAuxStore<ThreadMap, ElementD,
                                  cu::FloatRoundStyle::round_to_nearest, StrideD>;
using Evt = tb::Sm80EVT<Store, Evt1>;

using Gemm = cu::gemm::device::SparseGemmWithVisitor<
    int8_t, cu::layout::RowMajor, int8_t, cu::layout::ColumnMajor, float,
    cu::layout::RowMajor, int32_t, cu::arch::OpClassTensorOp, cu::arch::Sm80,
    TileShape, WarpShape, InstructionShape, Evt,
    cu::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages, kAlignmentAB,
    kAlignmentAB, cu::arch::OpMultiplyAddSaturate>;

}  // namespace

extern "C" int vllm_sparse_int8_mm_sm80(void* stream, void* d, void const* a,
                                        void const* e, void const* b,
                                        float const* scale_m, float const* scale_n,
                                        int scale_n_is_vector, int M, int N, int K) {
  using ElementE = typename Gemm::ElementE;
  using LayoutE = typename Gemm::LayoutE;
  cu::TensorRef<int8_t const, cu::layout::RowMajor> ref_a{
      static_cast<int8_t const*>(a), K / 2};
  cu::TensorRef<int8_t const, cu::layout::ColumnMajor> ref_b{
      static_cast<int8_t const*>(b), K};
  cu::TensorRef<ElementE const, LayoutE> ref_e{
      static_cast<ElementE const*>(e), LayoutE::kInterleave * M};

  typename Store::Arguments store{static_cast<ElementD*>(d),
                                  StrideD{static_cast<int64_t>(N), {}, {}}};
  typename Evt0::Arguments evt0{{const_cast<float*>(scale_n), scale_n_is_vector != 0},
                                {},
                                {}};
  typename Evt1::Arguments evt1{{const_cast<float*>(scale_m), true}, evt0, {}};

  typename Gemm::Arguments args;
  args.problem_size = {M, N, K};
  args.ref_A = ref_a;
  args.ref_B = ref_b;
  args.ref_E = ref_e;
  args.epilogue = typename Evt::Arguments{evt1, store};

  Gemm op;
  return op(args, nullptr, static_cast<cudaStream_t>(stream)) == cu::Status::kSuccess
             ? 0
             : 2;
}
