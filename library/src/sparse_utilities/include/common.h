/*! \file */

/*
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "sparse_utility_types.h"

#include <hip/hip_runtime.h>

#include "rocgraph_assert.hpp"

// clang-format off

#ifndef ROCGRAPH_USE_MOVE_DPP
#if defined(__GFX8__) || defined(__GFX9__)
#define ROCGRAPH_USE_MOVE_DPP 1
#else
#define ROCGRAPH_USE_MOVE_DPP 0
#endif
#endif

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) ((dir == rocgraph_direction_row) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (block_dim * block_dim * (j) + (bi) * block_dim + (bj))
#define BSR_IND_C(j, bi, bj) (block_dim * block_dim * (j) + (bi) + (bj) * block_dim)

#define GEBSR_IND(j, bi, bj, dir) ((dir == rocgraph_direction_row) ? GEBSR_IND_R(j, bi, bj) : GEBSR_IND_C(j, bi, bj))
#define GEBSR_IND_R(j, bi, bj) (row_block_dim * col_block_dim * (j) + (bi) * col_block_dim + (bj))
#define GEBSR_IND_C(j, bi, bj) (row_block_dim * col_block_dim * (j) + (bi) + (bj) * row_block_dim)



namespace rocgraph
{

// find next power of 2
__device__ __host__ __forceinline__ uint32_t fnp2(uint32_t x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

__device__ __forceinline__ int8_t ldg(const int8_t* ptr) { return __ldg(ptr); }
__device__ __forceinline__ int32_t ldg(const int32_t* ptr) { return __ldg(ptr); }
__device__ __forceinline__ int64_t ldg(const int64_t* ptr) { return __ldg(ptr); }
__device__ __forceinline__ float ldg(const float* ptr) { return __ldg(ptr); }
__device__ __forceinline__ double ldg(const double* ptr) { return __ldg(ptr); }

template <typename T>
__device__ __forceinline__ T fma(T p, T q, T r);

template <>
__device__ __forceinline__ int32_t fma(int32_t p, int32_t q, int32_t r) { return p * q + r; }

template <>
__device__ __forceinline__ int64_t fma(int64_t p, int64_t q, int64_t r) { return p * q + r; }

template <>
__device__ __forceinline__ float fma(float p, float q, float r) { return std::fma(p, q, r); }

template <>
__device__ __forceinline__ double fma(double p, double q, double r) { return std::fma(p, q, r); }

  __device__ __forceinline__ float abs(float x) { return x < 0.0f ? -x : x; }
  __device__ __forceinline__ double abs(double x) { return x < 0.0 ? -x : x; }

  __device__ __host__ __forceinline__ float ceil(float x) { return std::ceilf(x); }
  __device__ __host__ __forceinline__ double ceil(double x) { return std::ceil(x); }

  __device__ __host__ __forceinline__ float max(float x, float y) { return std::max(x, y); }
  __device__ __host__ __forceinline__ double max(double x, double y) { return std::max(x, y); }
  __device__ __host__ __forceinline__ int32_t max(int32_t x, int32_t y) { return std::max(x, y); }
  __device__ __host__ __forceinline__ int64_t max(int64_t x, int64_t y) { return std::max(x, y); }
  __device__ __host__ __forceinline__ uint32_t max(uint32_t x, uint32_t y) { return std::max(x, y); }
  __device__ __host__ __forceinline__ uint64_t max(uint64_t x, uint64_t y) { return std::max(x, y); }

  __device__ __host__ __forceinline__ float min(float x, float y) { return std::min(x, y); }
  __device__ __host__ __forceinline__ double min(double x, double y) { return std::min(x, y); }
  __device__ __host__ __forceinline__ int32_t min(int32_t x, int32_t y) { return std::min(x, y); }
  __device__ __host__ __forceinline__ int64_t min(int64_t x, int64_t y) { return std::min(x, y); }
  __device__ __host__ __forceinline__ uint32_t min(uint32_t x, uint32_t y) { return std::min(x, y); }
  __device__ __host__ __forceinline__ uint64_t min(uint64_t x, uint64_t y) { return std::min(x, y); }


  __device__ __host__ __forceinline__ float sqrt(float val) { return std::sqrt(val); }
  __device__ __host__ __forceinline__ double sqrt(double val) { return std::sqrt(val); }

  __device__ __host__ __forceinline__ float log2(float val) { return std::log2f(val); }
  __device__ __host__ __forceinline__ double log2(double val) { return std::log2(val); }
  __device__ __host__ __forceinline__ float  log2(int32_t val) { return std::log2f(val); }
  __device__ __host__ __forceinline__ double log2(int64_t val) { return std::log2(val); }
  __device__ __host__ __forceinline__ float  log2(uint32_t val) { return std::log2f(val); }
  __device__ __host__ __forceinline__ double log2(uint64_t val) { return std::log2(val); }


__device__ __forceinline__ int32_t conj(const int32_t& x) { return x; }
__device__ __forceinline__ float conj(const float& x) { return x; }
__device__ __forceinline__ double conj(const double& x) { return x; }

__device__ __forceinline__ float real(const float& x) { return x; }
__device__ __forceinline__ double real(const double& x) { return x; }

__device__ __forceinline__ float imag(const float& x) { return static_cast<float>(0); }
__device__ __forceinline__ double imag(const double& x) { return static_cast<double>(0); }

 __device__ __forceinline__ bool gt(const float& x, const float& y) { return x > y; }
 __device__ __forceinline__ bool gt(const double& x, const double& y) { return x > y; }

__device__ __forceinline__ float nontemporal_load(const float* ptr) { return __builtin_nontemporal_load(ptr); }
__device__ __forceinline__ double nontemporal_load(const double* ptr) { return __builtin_nontemporal_load(ptr); }


__device__ __forceinline__ int8_t nontemporal_load(const int8_t* ptr) { return __builtin_nontemporal_load(ptr); }
__device__ __forceinline__ int32_t nontemporal_load(const int32_t* ptr) { return __builtin_nontemporal_load(ptr); }
__device__ __forceinline__ int64_t nontemporal_load(const int64_t* ptr) { return __builtin_nontemporal_load(ptr); }

__device__ __forceinline__ void nontemporal_store(float val, float* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void nontemporal_store(double val, double* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void nontemporal_store(int8_t val, int8_t* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void nontemporal_store(int32_t val, int32_t* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void nontemporal_store(int64_t val, int64_t* ptr) { __builtin_nontemporal_store(val, ptr); }

__device__ __forceinline__ int32_t shfl(int32_t var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ int64_t shfl(int64_t var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ float shfl(float var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ double shfl(double var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }

template <typename T>
__device__ __forceinline__ T atomic_min(T * ptr, T val)
{
  return atomicMin(ptr,val);
}

template <typename T>
__device__ __forceinline__ T atomic_max(T * ptr, T val)
{
  return atomicMax(ptr,val);
}

template <typename T>
__device__ __forceinline__ T atomic_add(T * ptr, T val)
{
  return atomicAdd(ptr,val);
}

template <typename T>
__device__ __forceinline__ T atomic_cas(T * ptr, T cmp, T val)
{
  return atomicCAS(ptr, cmp, val);
}



template <>
__device__ __forceinline__ int64_t atomic_min<int64_t>(int64_t * ptr, int64_t val)
{
  return atomicMin((uint64_t*)ptr, (uint64_t)val);
}

template <>
__device__ __forceinline__ int64_t atomic_max<int64_t>(int64_t * ptr, int64_t val)
{
  return atomicMax((uint64_t*)ptr, val);
}


template <>
__device__ __forceinline__ int64_t atomic_add<int64_t>(int64_t * ptr, int64_t val)
{
  return atomicAdd((uint64_t*)ptr, val);
}

template <>
__device__ __forceinline__ int64_t atomic_cas(int64_t* ptr, int64_t cmp, int64_t val)
{
  return atomicCAS((uint64_t*)ptr, cmp, val);
}

__device__ __forceinline__ bool is_inf(float val){ return (val == std::numeric_limits<float>::infinity()); }
__device__ __forceinline__ bool is_inf(double val){ return (val == std::numeric_limits<double>::infinity()); }

__device__ __forceinline__ bool is_nan(float val){ return (val != val); }
__device__ __forceinline__ bool is_nan(double val){ return (val != val); }

template<typename T, typename I, typename J>
__device__ __forceinline__ T* load_pointer(T* p, J batch, I stride)
{
    return p + stride * batch;
}

template<typename T, typename I, typename J>
__device__ __forceinline__ const T* load_pointer(const T* p, J batch, I stride)
{
    return p + stride * batch;
}

template <typename T>
__device__ __forceinline__ T conj_val(T val, bool conj)
{
    return conj ? rocgraph::conj(val) : val;
}


// Block reduce kernel computing block sum
template <uint32_t BLOCKSIZE, typename T>
__device__ __forceinline__ void blockreduce_sum(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = data[i] + data[i + 512]; } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = data[i] + data[i + 256]; } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = data[i] + data[i + 128]; } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = data[i] + data[i +  64]; } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = data[i] + data[i +  32]; } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = data[i] + data[i +  16]; } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = data[i] + data[i +   8]; } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = data[i] + data[i +   4]; } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = data[i] + data[i +   2]; } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = data[i] + data[i +   1]; } __syncthreads(); }
}

// Block reduce kernel computing blockwide maximum entry
template <uint32_t BLOCKSIZE, typename T>
__device__ __forceinline__ void blockreduce_max(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i + 512]); } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i + 256]); } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i + 128]); } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +  64]); } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +  32]); } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +  16]); } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +   8]); } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +   4]); } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +   2]); } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = rocgraph::max(data[i], data[i +   1]); } __syncthreads(); }
}

// Block reduce kernel computing blockwide minimum entry
template <uint32_t BLOCKSIZE, typename T>
__device__ __forceinline__ void blockreduce_min(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i + 512]); } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i + 256]); } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i + 128]); } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +  64]); } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +  32]); } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +  16]); } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +   8]); } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +   4]); } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +   2]); } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = rocgraph::min(data[i], data[i +   1]); } __syncthreads(); }
}


#if ROCGRAPH_USE_MOVE_DPP

template <uint32_t WFSIZE>
static __device__ __forceinline__ void wfreduce_max(float* maximum)
{
    typedef union flt_b32
    {
        float    val;
        uint32_t b32[1];
    } flt_b32_t;

    flt_b32_t upper_max;
    flt_b32_t temp_max;
    temp_max.val = *maximum;

    if(WFSIZE > 1)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x111, 0xf, 0xf, false);
        temp_max.val     = rocgraph::max(temp_max.val, upper_max.val);
    }

    if(WFSIZE > 2)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x112, 0xf, 0xf, false);
        temp_max.val     = rocgraph::max(temp_max.val, upper_max.val);
    }

    if(WFSIZE > 4)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x114, 0xf, 0xe, false);
        temp_max.val     = rocgraph::max(temp_max.val, upper_max.val);
    }

    if(WFSIZE > 8)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x118, 0xf, 0xc, false);
        temp_max.val     = rocgraph::max(temp_max.val, upper_max.val);
    }

    if(WFSIZE > 16)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x142, 0xa, 0xf, false);
        temp_max.val     = rocgraph::max(temp_max.val, upper_max.val);
    }

    if(WFSIZE > 32)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x143, 0xc, 0xf, false);
        temp_max.val     = rocgraph::max(temp_max.val, upper_max.val);
    }

    *maximum = temp_max.val;
}

// DPP-based wavefront reduction maximum
template <uint32_t WFSIZE>
static __device__ __forceinline__ void wfreduce_max(double* maximum)
{
    typedef union i64_b32
    {
        double   i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_max;
    i64_b32_t temp_max;
    temp_max.i64 = *maximum;

    if(WFSIZE > 1)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x111, 0xf, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x111, 0xf, 0xf, false);
        temp_max.i64     = rocgraph::max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 2)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x112, 0xf, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x112, 0xf, 0xf, false);
        temp_max.i64     = rocgraph::max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 4)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x114, 0xf, 0xe, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x114, 0xf, 0xe, false);
        temp_max.i64     = rocgraph::max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 8)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x118, 0xf, 0xc, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x118, 0xf, 0xc, false);
        temp_max.i64     = rocgraph::max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 16)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x142, 0xa, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x142, 0xa, 0xf, false);
        temp_max.i64     = rocgraph::max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 32)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x143, 0xc, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x143, 0xc, 0xf, false);
        temp_max.i64     = rocgraph::max(temp_max.i64, upper_max.i64);
    }

    *maximum = temp_max.i64;
}

// DPP-based wavefront reduction maximum
template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_max(int* maximum)
{
    if(WFSIZE >  1) *maximum = rocgraph::max(*maximum, __hip_move_dpp(*maximum, 0x111, 0xf, 0xf, 0));
    if(WFSIZE >  2) *maximum = rocgraph::max(*maximum, __hip_move_dpp(*maximum, 0x112, 0xf, 0xf, 0));
    if(WFSIZE >  4) *maximum = rocgraph::max(*maximum, __hip_move_dpp(*maximum, 0x114, 0xf, 0xe, 0));
    if(WFSIZE >  8) *maximum = rocgraph::max(*maximum, __hip_move_dpp(*maximum, 0x118, 0xf, 0xc, 0));
    if(WFSIZE > 16) *maximum = rocgraph::max(*maximum, __hip_move_dpp(*maximum, 0x142, 0xa, 0xf, 0));
    if(WFSIZE > 32) *maximum = rocgraph::max(*maximum, __hip_move_dpp(*maximum, 0x143, 0xc, 0xf, 0));
}

template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_max(int64_t* maximum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_max;
    i64_b32_t temp_max;
    temp_max.i64 = *maximum;

    if(WFSIZE > 1)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x111, 0xf, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x111, 0xf, 0xf, false);
        temp_max.i64 = max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 2)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x112, 0xf, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x112, 0xf, 0xf, false);
        temp_max.i64 = max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 4)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x114, 0xf, 0xe, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x114, 0xf, 0xe, false);
        temp_max.i64 = max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 8)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x118, 0xf, 0xc, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x118, 0xf, 0xc, false);
        temp_max.i64 = max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 16)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x142, 0xa, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x142, 0xa, 0xf, false);
        temp_max.i64 = max(temp_max.i64, upper_max.i64);
    }

    if(WFSIZE > 32)
    {
        upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x143, 0xc, 0xf, false);
        upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x143, 0xc, 0xf, false);
        temp_max.i64 = max(temp_max.i64, upper_max.i64);
    }

    *maximum = temp_max.i64;
}

// DPP-based wavefront reduction minimum
template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_min(int* minimum)
{
    if(WFSIZE >  1) *minimum = rocgraph::min(*minimum, __hip_move_dpp(*minimum, 0x111, 0xf, 0xf, 0));
    if(WFSIZE >  2) *minimum = rocgraph::min(*minimum, __hip_move_dpp(*minimum, 0x112, 0xf, 0xf, 0));
    if(WFSIZE >  4) *minimum = rocgraph::min(*minimum, __hip_move_dpp(*minimum, 0x114, 0xf, 0xe, 0));
    if(WFSIZE >  8) *minimum = rocgraph::min(*minimum, __hip_move_dpp(*minimum, 0x118, 0xf, 0xc, 0));
    if(WFSIZE > 16) *minimum = rocgraph::min(*minimum, __hip_move_dpp(*minimum, 0x142, 0xa, 0xf, 0));
    if(WFSIZE > 32) *minimum = rocgraph::min(*minimum, __hip_move_dpp(*minimum, 0x143, 0xc, 0xf, 0));
}

template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_min(int64_t* minimum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_min;
    i64_b32_t temp_min;
    temp_min.i64 = *minimum;

    if(WFSIZE > 1)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x111, 0xf, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x111, 0xf, 0xf, false);
        temp_min.i64 = rocgraph::min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 2)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x112, 0xf, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x112, 0xf, 0xf, false);
        temp_min.i64 = rocgraph::min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 4)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x114, 0xf, 0xe, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x114, 0xf, 0xe, false);
        temp_min.i64 = rocgraph::min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 8)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x118, 0xf, 0xc, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x118, 0xf, 0xc, false);
        temp_min.i64 = rocgraph::min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 16)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x142, 0xa, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x142, 0xa, 0xf, false);
        temp_min.i64 = rocgraph::min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 32)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x143, 0xc, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x143, 0xc, 0xf, false);
        temp_min.i64 = rocgraph::min(temp_min.i64, upper_min.i64);
    }

    *minimum = temp_min.i64;
}

template <uint32_t WFSIZE>
__device__ __forceinline__ int32_t wfreduce_sum(int32_t sum)
{
    if(WFSIZE >  1) sum += __hip_move_dpp(sum, 0x111, 0xf, 0xf, 0);
    if(WFSIZE >  2) sum += __hip_move_dpp(sum, 0x112, 0xf, 0xf, 0);
    if(WFSIZE >  4) sum += __hip_move_dpp(sum, 0x114, 0xf, 0xe, 0);
    if(WFSIZE >  8) sum += __hip_move_dpp(sum, 0x118, 0xf, 0xc, 0);
    if(WFSIZE > 16) sum += __hip_move_dpp(sum, 0x142, 0xa, 0xf, 0);
    if(WFSIZE > 32) sum += __hip_move_dpp(sum, 0x143, 0xc, 0xf, 0);

    return sum;
}

template <uint32_t WFSIZE>
__device__ __forceinline__ int64_t wfreduce_sum(int64_t sum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_sum;
    i64_b32_t temp_sum;
    temp_sum.i64 = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    sum = temp_sum.i64;
    return sum;
}

// DPP-based float wavefront reduction sum
template <uint32_t WFSIZE>
__device__ __forceinline__ float wfreduce_sum(float sum)
{
    typedef union flt_b32
    {
        float val;
        uint32_t b32;
    } flt_b32_t;

    flt_b32_t upper_sum;
    flt_b32_t temp_sum;
    temp_sum.val = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

// DPP-based double wavefront reduction
template <uint32_t WFSIZE>
__device__ __forceinline__ double wfreduce_sum(double sum)
{
    typedef union dbl_b32
    {
        double val;
        uint32_t b32[2];
    } dbl_b32_t;

    dbl_b32_t upper_sum;
    dbl_b32_t temp_sum;
    temp_sum.val = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}
#else /* ROCGRAPH_USE_MOVE_DPP */


template <uint32_t WFSIZE>
static __device__ __forceinline__ void wfreduce_max(double* maximum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *maximum = rocgraph::max(*maximum, __shfl_xor(*maximum, i));
    }
}
template <uint32_t WFSIZE>
static __device__ __forceinline__ void wfreduce_max(float* maximum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *maximum = rocgraph::max(*maximum, __shfl_xor(*maximum, i));
    }
}

template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_max(int* maximum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *maximum = rocgraph::max(*maximum, __shfl_xor(*maximum, i));
    }
}

template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_max(int64_t* maximum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *maximum = max(*maximum, __shfl_xor(*maximum, i));
    }
}

template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_min(int* minimum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *minimum = rocgraph::min(*minimum, __shfl_xor(*minimum, i));
    }
}

template <uint32_t WFSIZE>
__device__ __forceinline__ void wfreduce_min(int64_t* minimum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *minimum = rocgraph::min(*minimum, __shfl_xor(*minimum, i));
    }
}

template <uint32_t WFSIZE>
__device__ __forceinline__ int32_t wfreduce_sum(int32_t sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }
    return sum;
}

template <uint32_t WFSIZE>
__device__ __forceinline__ int64_t wfreduce_sum(int64_t sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }
    return sum;
}

template <uint32_t WFSIZE>
__device__ __forceinline__ float wfreduce_sum(float sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

template <uint32_t WFSIZE>
__device__ __forceinline__ double wfreduce_sum(double sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}
#endif /* ROCGRAPH_USE_MOVE_DPP */

    // clang-format on

    // Perform dense matrix transposition
    template <uint32_t DIMX, uint32_t DIMY, typename I, typename T>
    __device__ void dense_transpose_device(
        I m, I n, T alpha, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
    {
        int lid = threadIdx.x & (DIMX - 1);
        int wid = threadIdx.x / DIMX;

        I row_A = blockIdx.x * DIMX + lid;
        I row_B = blockIdx.x * DIMX + wid;

        __shared__ T sdata[DIMX][DIMX];

        for(I j = 0; j < n; j += DIMX)
        {
            __syncthreads();

            I col_A = j + wid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(row_A < m && col_A + k < n)
                {
                    sdata[wid + k][lid] = A[row_A + lda * (col_A + k)];
                }
            }

            __syncthreads();

            I col_B = j + lid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(col_B < n && row_B + k < m)
                {
                    B[col_B + ldb * (row_B + k)] = alpha * sdata[lid][wid + k];
                }
            }
        }
    }

    // Perform dense matrix back transposition
    template <uint32_t DIMX, uint32_t DIMY, typename I, typename T>
    ROCGRAPH_KERNEL(DIMX* DIMY)
    void dense_transpose_back(
        I m, I n, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
    {
        int lid = hipThreadIdx_x & (DIMX - 1);
        int wid = hipThreadIdx_x / DIMX;

        I row_A = hipBlockIdx_x * DIMX + wid;
        I row_B = hipBlockIdx_x * DIMX + lid;

        __shared__ T sdata[DIMX][DIMX];

        for(I j = 0; j < n; j += DIMX)
        {
            __syncthreads();

            I col_A = j + lid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(col_A < n && row_A + k < m)
                {
                    sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
                }
            }

            __syncthreads();

            I col_B = j + wid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(row_B < m && col_B + k < n)
                {
                    B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
                }
            }
        }
    }

    // BSR gather functionality to permute the BSR values array
    template <uint32_t WFSIZE, uint32_t DIMY, uint32_t BSRDIM, typename I, typename T>
    ROCGRAPH_KERNEL(WFSIZE* DIMY)
    void bsr_gather(rocgraph_direction dir,
                    I                  nnzb,
                    const I* __restrict__ perm,
                    const T* __restrict__ bsr_val_A,
                    T* __restrict__ bsr_val_T,
                    I block_dim)
    {
        int lid = threadIdx.x & (BSRDIM - 1);
        int wid = threadIdx.x / BSRDIM;

        // Non-permuted nnz index
        I j = blockIdx.x * DIMY + threadIdx.y;

        // Do not exceed the number of elements
        if(j >= nnzb)
        {
            return;
        }

        // Load the permuted nnz index
        I p = perm[j];

        // Gather values from A and store them to T with respect to the
        // given row / column permutation
        for(I bi = lid; bi < block_dim; bi += BSRDIM)
        {
            for(I bj = wid; bj < block_dim; bj += BSRDIM)
            {
                bsr_val_T[BSR_IND(j, bi, bj, dir)] = bsr_val_A[BSR_IND(p, bj, bi, dir)];
            }
        }
    }

    // Set array to be filled with value
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void set_array_to_value(I m, T* __restrict__ array, T value)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(idx >= m)
        {
            return;
        }

        array[idx] = value;
    }

    // Scale array by value
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void scale_array(I m, T* __restrict__ array, T value)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(idx >= m)
        {
            return;
        }

        array[idx] *= value;
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void scale_array(I m, T* __restrict__ array, const T* value)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(idx >= m)
        {
            return;
        }

        if(*value != static_cast<T>(1))
        {
            array[idx] *= (*value);
        }
    }

    // Scale 2d array by value
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void scale_array_2d(
        I m, I n, int64_t ld, int64_t stride, T* __restrict__ array, T value, rocgraph_order order)
    {
        I gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        I batch = hipBlockIdx_y;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocgraph_order_column) ? gid / m : gid / n;
        I lid = (order == rocgraph_order_column) ? gid % m : gid % n;

        if(value == static_cast<T>(0))
        {
            array[lid + ld * wid + stride * batch] = static_cast<T>(0);
        }
        else
        {
            array[lid + ld * wid + stride * batch] *= value;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void scale_array_2d(I       m,
                        I       n,
                        int64_t ld,
                        int64_t stride,
                        T* __restrict__ array,
                        const T*       value,
                        rocgraph_order order)
    {
        I gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        I batch = hipBlockIdx_y;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocgraph_order_column) ? gid / m : gid / n;
        I lid = (order == rocgraph_order_column) ? gid % m : gid % n;

        if((*value) == static_cast<T>(0))
        {
            array[lid + ld * wid + stride * batch] = static_cast<T>(0);
        }
        else
        {
            array[lid + ld * wid + stride * batch] *= (*value);
        }
    }

    // conjugate values in array
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void conjugate(I m, T* __restrict__ array)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(idx >= m)
        {
            return;
        }

        array[idx] = rocgraph::conj(array[idx]);
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void csr_max_nnz_per_row(J m, const I* __restrict__ csr_row_ptr, J* __restrict__ max_nnz)
    {
        int tid = hipThreadIdx_x;
        J   gid = tid + BLOCKSIZE * hipBlockIdx_x;

        __shared__ I shared[BLOCKSIZE];

        if(gid < m)
        {
            shared[tid] = csr_row_ptr[gid + 1] - csr_row_ptr[gid];
        }
        else
        {
            shared[tid] = 0;
        }

        __syncthreads();

        rocgraph::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            rocgraph::atomic_max(max_nnz, shared[0]);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCGRAPH_KERNEL(BLOCKSIZE)
    void memset2d_kernel(I m, I n, T value, T* __restrict__ data, int64_t ld, rocgraph_order order)
    {
        I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocgraph_order_column) ? gid / m : gid / n;
        I lid = (order == rocgraph_order_column) ? gid % m : gid % n;

        data[lid + ld * wid] = value;
    }

    template <bool SLEEP>
    __device__ __forceinline__ int spin_loop(int* __restrict__ done, int scope)
    {
        int      local_done    = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
        uint32_t times_through = 0;
        while(!local_done)
        {
            if(SLEEP)
            {
                for(uint32_t i = 0; i < times_through; ++i)
                {
                    __builtin_amdgcn_s_sleep(1);
                }

                if(times_through < 3907)
                {
                    ++times_through;
                }
            }
            local_done = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
        }
        return local_done;
    }
}
