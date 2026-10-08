/*
 * lab_ops: two kernels from this folder exposed to PyTorch.
 *
 * A PyTorch C++/CUDA extension is ordinary CUDA code plus a thin wrapper
 * that checks the input tensors, allocates the output with torch::empty_like,
 * launches on PyTorch's current CUDA stream, and is registered with pybind11.
 * test_lab_ops.py compiles this file on the fly with
 * torch.utils.cpp_extension.load and compares the results with PyTorch's own
 * operators.
 *
 * Kernels (same as ml/03_rmsnorm and ml/01_softmax_online):
 *   lab_ops.rms_norm(x, weight, eps)  -> x / sqrt(mean(x^2) + eps) * weight
 *   lab_ops.softmax(x)                -> softmax over the last dimension
 * Both take contiguous float32 CUDA tensors whose last dimension is a
 * multiple of 4 and treat all leading dimensions as rows.
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cfloat>

namespace {

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float partial[THREADS / 32];
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32] : 0.0f;
  return warp_sum(v);
}

__global__ void rmsnorm_kernel(const float *in, float *out, int cols,
                               const float *weight, float eps) {
  const float4 *x = reinterpret_cast<const float4 *>(in + static_cast<size_t>(blockIdx.x) * cols);
  float4 *y = reinterpret_cast<float4 *>(out + static_cast<size_t>(blockIdx.x) * cols);
  const float4 *w = reinterpret_cast<const float4 *>(weight);
  const int cols4 = cols / 4;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  const float inv_rms = rsqrtf(block_sum(ss) / cols + eps);
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    const float4 g = w[c];
    y[c] = make_float4(v.x * inv_rms * g.x, v.y * inv_rms * g.y,
                       v.z * inv_rms * g.z, v.w * inv_rms * g.w);
  }
}

struct MaxSum {
  float m;
  float d;
};

__device__ __forceinline__ MaxSum merge(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.m, b.m);
  return {m, a.d * __expf(a.m - m) + b.d * __expf(b.m - m)};
}

__device__ __forceinline__ MaxSum warp_merge(MaxSum v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) {
    v = merge(v, {__shfl_xor_sync(FULL_MASK, v.m, o), __shfl_xor_sync(FULL_MASK, v.d, o)});
  }
  return v;
}

__device__ __forceinline__ MaxSum block_merge(MaxSum v) {
  __shared__ MaxSum partial[THREADS / 32];
  v = warp_merge(v);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32] : MaxSum{-FLT_MAX, 0.0f};
  return warp_merge(v);
}

__global__ void softmax_kernel(const float *in, float *out, int cols) {
  const float4 *x = reinterpret_cast<const float4 *>(in + static_cast<size_t>(blockIdx.x) * cols);
  float4 *y = reinterpret_cast<float4 *>(out + static_cast<size_t>(blockIdx.x) * cols);
  const int cols4 = cols / 4;
  MaxSum s = {-FLT_MAX, 0.0f};
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    // Merging a single element is the online-softmax update (m = x, d = 1).
    s = merge(s, {v.x, 1.0f});
    s = merge(s, {v.y, 1.0f});
    s = merge(s, {v.z, 1.0f});
    s = merge(s, {v.w, 1.0f});
  }
  s = block_merge(s);
  const float inv = 1.0f / s.d;
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    y[c] = make_float4(__expf(v.x - s.m) * inv, __expf(v.y - s.m) * inv,
                       __expf(v.z - s.m) * inv, __expf(v.w - s.m) * inv);
  }
}

void check_input(const torch::Tensor &x) {
  TORCH_CHECK(x.is_cuda(), "expected a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "expected float32");
  TORCH_CHECK(x.is_contiguous(), "expected a contiguous tensor");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) % 4 == 0,
              "the last dimension must be a multiple of 4");
}

}  // namespace

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  check_input(x);
  check_input(weight);
  TORCH_CHECK(weight.numel() == x.size(-1), "weight must match the last dimension");
  auto out = torch::empty_like(x);
  const int cols = static_cast<int>(x.size(-1));
  const int rows = static_cast<int>(x.numel() / cols);
  if (rows > 0) {
    rmsnorm_kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), cols,
        weight.data_ptr<float>(), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

torch::Tensor softmax(torch::Tensor x) {
  check_input(x);
  auto out = torch::empty_like(x);
  const int cols = static_cast<int>(x.size(-1));
  const int rows = static_cast<int>(x.numel() / cols);
  if (rows > 0) {
    softmax_kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm, "RMSNorm over the last dimension (CUDA, float32)",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-5);
  m.def("softmax", &softmax, "Softmax over the last dimension (CUDA, float32)",
        pybind11::arg("x"));
}
