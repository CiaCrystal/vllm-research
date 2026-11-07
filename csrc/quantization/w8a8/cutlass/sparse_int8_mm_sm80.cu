// 2:4 sparse INT8 GEMM for SM80 to SM89, built by tools/build_sparse_int8_sm80.sh.
//
// Computes D = diag(scale_m) * A * B * diag(scale_n), where A is the compressed s8
// weight (M outputs, K/2 stored per row) and B holds N column-major s8 inputs. D is
// written column-major, which is row-major [N, M] from the caller's point of view.
// CUTLASS 2.x sparse GEMMs have no such layout, so ColumnMajorStore does it.
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

// VisitorAuxStore with a store per element: a stride of M between neighbouring
// elements of a fragment rules out its wide stores.
template <class ThreadMap, class Element, class StrideMNL>
struct ColumnMajorStore {
  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };
  using Params = Arguments;
  struct SharedStorage {};

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                  Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  CUTLASS_HOST_DEVICE ColumnMajorStore() {}
  CUTLASS_HOST_DEVICE ColumnMajorStore(Params const& params, SharedStorage const&)
      : params_ptr(&params) {}
  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : tb::detail::EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(GTensor&& g, RTensor&& r, CTensor&& c, ProblemShape p)
        : tC_gAux(cute::forward<GTensor>(g)),
          tC_rAux(cute::forward<RTensor>(r)),
          tC_cAux(cute::forward<CTensor>(c)),
          problem_shape(p) {}
    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    ProblemShape problem_shape;

    CUTLASS_DEVICE void begin_step(int) { clear(tC_rAux); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(int, int, int, int frg_idx,
                              cu::Array<ElementAccumulator, FragmentSize> const&,
                              cu::Array<ElementInput, FragmentSize> const& frg_input) {
      cu::NumericArrayConverter<Element, ElementInput, FragmentSize,
                                cu::FloatRoundStyle::round_to_nearest>
          convert{};
      Tensor frg = recast<cu::Array<Element, FragmentSize>>(coalesce(tC_rAux));
      frg(frg_idx) = convert(frg_input);
      return frg_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
      auto dst_v = filter(tC_gAux(_, _, _, step_idx));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        if (elem_less(coord_v(i), problem_shape)) {
          dst_v(i) = src_v(i);
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cu::gemm::GemmCoord threadblock_tile_offset,
                                    int thread_idx, ProblemShape problem_shape) {
    Tensor mAux = make_tensor(make_gmem_ptr(params_ptr->ptr_aux), problem_shape,
                              params_ptr->dAux);
    Tensor tC_gAux = group_modes<3, 6>(
        ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset));
    Tensor tC_rAux = make_tensor<Element>(take<0, 3>(shape(tC_gAux)));
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = group_modes<3, 6>(
        ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset));
    return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux),
                     ProblemShape>(cute::move(tC_gAux), cute::move(tC_rAux),
                                   cute::move(tC_cAux), problem_shape);
  }
};

using ElementD = cu::bfloat16_t;
using TileShape = cu::gemm::GemmShape<128, 128, 128>;
using WarpShape = cu::gemm::GemmShape<64, 64, 128>;
using InstructionShape = cu::gemm::GemmShape<16, 8, 64>;
using ThreadMap = tb::OutputTileThreadLayout<TileShape, WarpShape, float, 4, 1>;

constexpr int kStages = 3;
constexpr int kAlignmentAB = 16;

using StrideD = Stride<Int<1>, int64_t, Int<0>>;
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
using Store = ColumnMajorStore<ThreadMap, ElementD, StrideD>;
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
                                  StrideD{{}, static_cast<int64_t>(M), {}}};
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
