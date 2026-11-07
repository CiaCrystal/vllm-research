// 2:4 sparse FP8 GEMM for SM120, built by tools/build_sparse_fp8_sm120.sh.
//
// Computes D = diag(scale_m) * A * B * diag(scale_n), where A is the compressed
// weight (M outputs, K/2 stored per row) and B holds N column-major inputs. D is
// column-major, which is row-major [N, M] from the caller's point of view.
#include <cuda_runtime.h>

#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"

using namespace cute;
namespace cu = cutlass;
namespace fusion = cutlass::epilogue::fusion;

namespace {

using ElementAB = cu::float_e4m3_t;
using ElementD = cu::bfloat16_t;
using LayoutA = cu::layout::RowMajor;
using LayoutB = cu::layout::ColumnMajor;
using LayoutD = cu::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 16;
constexpr int kAlignmentD = 8;

using ScaleM = fusion::Sm90ColOrScalarBroadcast<0, TileShape, float,
                                                Stride<Int<1>, Int<0>, Int<0>>>;
using ScaleN = fusion::Sm90RowOrScalarBroadcast<0, TileShape, float,
                                                Stride<Int<0>, Int<1>, Int<0>>>;
using Mul = fusion::Sm90Compute<cu::multiplies, float, float,
                                cu::FloatRoundStyle::round_to_nearest>;
using MulD = fusion::Sm90Compute<cu::multiplies, ElementD, float,
                                 cu::FloatRoundStyle::round_to_nearest>;
using Evt0 = fusion::Sm90EVT<Mul, ScaleN, fusion::Sm90AccFetch>;
using Evt = fusion::Sm90EVT<MulD, ScaleM, Evt0>;

using Epilogue = typename cu::epilogue::collective::CollectiveBuilder<
    cu::arch::Sm120, cu::arch::OpClassSparseTensorOp, TileShape, ClusterShape,
    cu::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD,
    kAlignmentD, ElementD, LayoutD, kAlignmentD,
    cu::epilogue::collective::EpilogueScheduleAuto, Evt>::CollectiveOp;

using MainloopBuilder = cu::gemm::collective::CollectiveBuilder<
    cu::arch::Sm120, cu::arch::OpClassSparseTensorOp, ElementAB, LayoutA,
    kAlignmentA, ElementAB, LayoutB, kAlignmentB, float, TileShape, ClusterShape,
    cu::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Epilogue::SharedStorage))>,
    cu::gemm::collective::KernelScheduleAuto>;
using Mainloop = typename MainloopBuilder::CollectiveOp;
using Config = typename MainloopBuilder::Sm1xxSparseConfig;
using ElementE = typename MainloopBuilder::ElementE;

using Kernel = cu::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop,
                                               Epilogue, cu::gemm::PersistentScheduler>;
using Gemm = cu::gemm::device::GemmUniversalAdapter<Kernel>;

}  // namespace

extern "C" int vllm_sparse_fp8_mm_sm120(void* stream, void* d, void const* a,
                                        void const* e, void const* b,
                                        float const* scale_m, float const* scale_n,
                                        int scale_n_is_vector, int M, int N, int K) {
  typename Gemm::Arguments args{};
  args.mode = cu::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop.ptr_A = static_cast<ElementAB const*>(a);
  args.mainloop.layout_a = Config::fill_layoutA(args.problem_shape);
  args.mainloop.ptr_B = static_cast<ElementAB const*>(b);
  args.mainloop.dB = cu::make_cute_packed_stride(typename Kernel::StrideB{},
                                                 make_shape(N, K, 1));
  args.mainloop.ptr_E = static_cast<ElementE const*>(e);
  args.mainloop.layout_e = Config::fill_layoutE(args.problem_shape);

  auto stride_d = cu::make_cute_packed_stride(typename Kernel::StrideC{},
                                              make_shape(M, N, 1));
  typename Evt0::Arguments evt0{{scale_n, scale_n_is_vector != 0}, {}, {}};
  auto* out = static_cast<ElementD*>(d);
  args.epilogue = {{{scale_m, true}, evt0, {}}, nullptr, stride_d, out, stride_d};

  Gemm op;
  if (op.can_implement(args) != cu::Status::kSuccess) {
    return 1;
  }
  return op.run(args, nullptr, static_cast<cudaStream_t>(stream)) ==
                 cu::Status::kSuccess
             ? 0
             : 2;
}
