// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"

#include <cstdlib>
#include <string>

namespace mlx::core {

namespace {

#define RETURN_IF_ERROR(cmd)          \
  if (auto ret = cmd; ret.is_bad()) { \
    return ret;                       \
  }

// In MLX a singleton dim (shape[dim] == 1) can have any stride, but in cuDNN
// whether a tensor is contiguous is determined with:
// shape[dim] == shape[dim + 1] * strides[dim + 1]
// So a contiguous array with singleton dims in MLX may be mistakenly treated
// as strided in cuDNN, and we work around it by normalizing the strides.
std::vector<int64_t> normalized_strides(const array& x) {
  std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
  if (std::all_of(
          strides.begin(), strides.end(), [](int64_t s) { return s == 0; })) {
    strides.back() = 1;
    return strides;
  }
  if (!x.flags().row_contiguous || x.ndim() < 2) {
    return strides;
  }
  for (int i = x.ndim() - 2; i >= 0; --i) {
    if (x.shape(i) == 1) {
      strides[i] = x.shape(i + 1) * strides[i + 1];
    }
  }
  return strides;
}

// Return the shape and strides after transposing from NHWC to NCHW.
inline auto nhwc_to_nchw(const array& x) {
  auto shape = convert_vector<int64_t>(x.shape());
  auto strides = normalized_strides(x);
  assert(shape.size() >= 3);
  shape.insert(shape.begin() + 1, shape.back());
  shape.erase(shape.end() - 1);
  strides.insert(strides.begin() + 1, strides.back());
  strides.erase(strides.end() - 1);
  return std::make_tuple(std::move(shape), std::move(strides));
}

// Opt-in cuDNN plan autotuning (MLX_CUDNN_AUTOTUNE=1). Default off keeps the
// upstream behavior (single heuristic-A plan). When on, we query more heuristic
// modes, build every candidate engine config, and time them once per shape
// (see DnnGraph::autotune_plans) so the fastest engine is chosen rather than
// cuDNN's first heuristic pick — the equivalent of torch's cudnn.benchmark.
bool cudnn_autotune_enabled() {
  static bool enabled = []() {
    const char* c = std::getenv("MLX_CUDNN_AUTOTUNE");
    return c != nullptr && std::string(c) != "0";
  }();
  return enabled;
}

} // namespace

// Autotuning is opt-in (MLX_CUDNN_AUTOTUNE) and additionally skipped for
// float32. On this unified-memory CUDA device (GB10) cudaMemGetInfo
// underreports the memory pool, so building/timing ALL candidate plans for the
// larger fp32 convs either OOMs cudaMallocAsync or crashes cuDNN when the plan
// set is workspace-filtered. Bounding the workspace was tried and reproducibly
// failed (loud OOM uncapped; silent cuDNN death when capped at 1 GiB and
// 256 MiB), so fp32 falls back to the single heuristic plan — the same,
// working path as autotune-off. fp16/bf16 (small workspaces) keep the win.
bool DnnGraph::autotune_enabled() const {
  return cudnn_autotune_enabled() && io_dtype_ != float32;
}

fe::error_t DnnGraph::prepare() {
  RETURN_IF_ERROR(validate());
  try {
    RETURN_IF_ERROR(build_operation_graph(handle_));
  } catch (cudnn_frontend::cudnnException& error) {
    // cuDNN bug: they did not catch all exceptions in the API.
    return {fe::error_code_t::CUDNN_BACKEND_API_FAILED, error.what()};
  }
  if (autotune_enabled()) {
    RETURN_IF_ERROR(
        create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}));
  } else {
    RETURN_IF_ERROR(create_execution_plans({fe::HeurMode_t::A}));
  }
  return {};
}

fe::error_t DnnGraph::build() {
  RETURN_IF_ERROR(check_support(handle_));
  if (autotune_enabled()) {
    // Build every candidate config; autotune_plans() picks the fastest.
    RETURN_IF_ERROR(build_plans(handle_, fe::BuildPlanPolicy_t::ALL));
  } else {
    RETURN_IF_ERROR(build_plans(handle_));
  }
  return {};
}

fe::error_t DnnGraph::autotune_plans(
    cu::CommandEncoder& encoder,
    std::unordered_map<int64_t, void*> variant_pack) {
  if (!autotune_enabled()) {
    return {};
  }
  // Time all built plans with the real inputs and select the fastest. Runs
  // eagerly on the stream (NOT inside CUDA-graph capture), so call this before
  // encode_capturing, once per shape (the result is held by the conv cache).
  // The try/catch keeps a workspace-allocation OOM from crashing: fall back to
  // the heuristic plan already selected by build_plans().
  try {
    int64_t workspace_size = get_autotune_workspace_size();
    void* workspace_ptr = allocate_workspace(encoder, workspace_size);
    cudnnSetStream(handle_, encoder.stream());
    return autotune(handle_, variant_pack, workspace_ptr);
  } catch (const std::exception&) {
    // OOM (or any failure) during autotuning: keep the heuristic plan already
    // selected by build_plans(). Correct, just not benchmarked.
    return {};
  }
}

fe::error_t DnnGraph::encode_graph(
    cu::CommandEncoder& encoder,
    std::unordered_map<int64_t, void*> variant_pack) {
  cudnnSetStream(handle_, encoder.stream());
  CudaGraph cuda_graph(encoder.device());
  RETURN_IF_ERROR(populate_cuda_graph(
      handle_, variant_pack, prepare_workspace(encoder), cuda_graph));
  encoder.add_graph_node(cuda_graph);
  return {};
}

fe::error_t DnnGraph::encode_capturing(
    cu::CommandEncoder& encoder,
    std::unordered_map<int64_t, void*> variant_pack) {
  auto* workspace_ptr = prepare_workspace(encoder);
  auto capture = encoder.capture_context();
  cudnnSetStream(handle_, encoder.stream());
  auto ret = execute(handle_, variant_pack, workspace_ptr);
  if (ret.is_bad()) {
    capture.discard = true;
  }
  return ret;
}

void* DnnGraph::prepare_workspace(cu::CommandEncoder& encoder) {
  int64_t workspace_size = 0;
  CHECK_CUDNN_FE_ERROR(get_workspace_size(workspace_size));
  return allocate_workspace(encoder, workspace_size);
}

void DnnGraph::set_tensor_attrs(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides) {
  tensor->set_uid(uid)
      .set_alignment(get_alignment(x))
      .set_data_type(dtype_to_cudnn_type(x.dtype()))
      .set_dim(shape)
      .set_stride(strides);
}

void DnnGraph::set_tensor_attrs(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x) {
  set_tensor_attrs(
      tensor,
      uid,
      x,
      convert_vector<int64_t>(x.shape()),
      normalized_strides(x));
}

void DnnGraph::set_tensor_attrs_nchw(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x) {
  auto [shape, strides] = nhwc_to_nchw(x);
  set_tensor_attrs(tensor, uid, x, shape, strides);
}

} // namespace mlx::core
