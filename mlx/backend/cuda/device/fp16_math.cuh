// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// Binary ops for half types.
///////////////////////////////////////////////////////////////////////////////

#define MLX_DEFINE_BINARY_OP(NAME, HALF_OP)                        \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x, T y) {                 \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return HALF_OP(x, y);                                        \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return HALF_OP(x, y);                                        \
    } else {                                                       \
      return ::NAME(x, y);                                         \
    }                                                              \
  }

MLX_DEFINE_BINARY_OP(max, __hmax)
MLX_DEFINE_BINARY_OP(min, __hmin)

#undef MLX_DEFINE_BINARY_OP

///////////////////////////////////////////////////////////////////////////////
// Float-promoting math for half types.
//
// CUDA 13's CCCL has no unambiguous cuda::std overloads of these math functions
// for __half/__nv_bfloat16: the arguments convert to both float and (long)
// double, so the call is ambiguous. These templates provide an exact match for
// the half types (which beats the converting overloads), compute in float, and
// cast back; for every other type they defer to cuda::std so float/double
// precision is preserved.
///////////////////////////////////////////////////////////////////////////////

template <typename T>
constexpr bool is_half_v =
    cuda::std::is_same_v<T, __half> || cuda::std::is_same_v<T, __nv_bfloat16>;

template <typename T>
__forceinline__ __device__ bool isnan(T x) {
  if constexpr (is_half_v<T>) {
    return cuda::std::isnan(static_cast<float>(x));
  } else {
    return cuda::std::isnan(x);
  }
}

#define MLX_DEFINE_FP_UNARY(NAME)                                  \
  template <typename T>                                            \
  __forceinline__ __device__ T NAME(T x) {                        \
    if constexpr (is_half_v<T>) {                                  \
      return static_cast<T>(cuda::std::NAME(static_cast<float>(x))); \
    } else {                                                       \
      return cuda::std::NAME(x);                                   \
    }                                                              \
  }

MLX_DEFINE_FP_UNARY(trunc)
MLX_DEFINE_FP_UNARY(exp)
MLX_DEFINE_FP_UNARY(log1p)

#undef MLX_DEFINE_FP_UNARY

#define MLX_DEFINE_FP_BINARY(NAME)                                          \
  template <typename T>                                                     \
  __forceinline__ __device__ T NAME(T x, T y) {                            \
    if constexpr (is_half_v<T>) {                                           \
      return static_cast<T>(                                                \
          cuda::std::NAME(static_cast<float>(x), static_cast<float>(y)));   \
    } else {                                                                \
      return cuda::std::NAME(x, y);                                         \
    }                                                                       \
  }

MLX_DEFINE_FP_BINARY(fmod)
MLX_DEFINE_FP_BINARY(pow)
MLX_DEFINE_FP_BINARY(atan2)

#undef MLX_DEFINE_FP_BINARY

///////////////////////////////////////////////////////////////////////////////
// Additional C++ operator overrides between half types and native types.
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
constexpr bool is_integral_except =
    cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, U>;

template <typename T, typename U>
constexpr bool is_arithmetic_except =
    cuda::std::is_arithmetic_v<T> && !cuda::std::is_same_v<T, U>;

#define MLX_DEFINE_HALF_OP(HALF, HALF2FLOAT, FLOAT2HALF, OP)          \
  template <                                                          \
      typename T,                                                     \
      typename = cuda::std::enable_if_t<is_integral_except<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(HALF x, T y) {          \
    return FLOAT2HALF(HALF2FLOAT(x) OP static_cast<float>(y));        \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      typename = cuda::std::enable_if_t<is_integral_except<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(T x, HALF y) {          \
    return FLOAT2HALF(static_cast<float>(x) OP HALF2FLOAT(y));        \
  }

#define MLX_DEFINE_HALF_CMP(HALF, HALF2FLOAT, OP)                       \
  template <                                                            \
      typename T,                                                       \
      typename = cuda::std::enable_if_t<is_arithmetic_except<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(HALF x, T y) {            \
    return HALF2FLOAT(x) OP static_cast<float>(y);                      \
  }                                                                     \
  template <                                                            \
      typename T,                                                       \
      typename = cuda::std::enable_if_t<is_arithmetic_except<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(T x, HALF y) {            \
    return static_cast<float>(y) OP HALF2FLOAT(x);                      \
  }

MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, +)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, -)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, *)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, /)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, +)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, -)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, *)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, /)
MLX_DEFINE_HALF_CMP(__half, __half2float, <)
MLX_DEFINE_HALF_CMP(__half, __half2float, >)
MLX_DEFINE_HALF_CMP(__half, __half2float, <=)
MLX_DEFINE_HALF_CMP(__half, __half2float, >=)
MLX_DEFINE_HALF_CMP(__half, __half2float, ==)
MLX_DEFINE_HALF_CMP(__half, __half2float, !=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, ==)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, !=)

#undef MLX_DEFINE_HALF_OP
#undef MLX_DEFINE_HALF_CMP

} // namespace mlx::core::cu
