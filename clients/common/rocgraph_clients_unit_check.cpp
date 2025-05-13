/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_check.hpp"
#include "rocgraph_clients_assert.hpp"
#include "utility.hpp"

#define ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, UNIT_ASSERT_EQ)   \
    do                                                              \
    {                                                               \
        for(size_t j = 0; j < N; ++j)                               \
        {                                                           \
            for(size_t i = 0; i < M; ++i)                           \
            {                                                       \
                if(rocgraph_isnan(A[i + j * LDA]))                  \
                {                                                   \
                    ASSERT_TRUE(rocgraph_isnan(B[i + j * LDB]));    \
                }                                                   \
                else                                                \
                {                                                   \
                    UNIT_ASSERT_EQ(A[i + j * LDA], B[i + j * LDB]); \
                }                                                   \
            }                                                       \
        }                                                           \
    } while(0)

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const float* A, size_t LDA, const float* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_EQ);
}

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const double* A, size_t LDA, const double* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_DOUBLE_EQ);
}

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const int8_t* A, size_t LDA, const int8_t* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const int32_t* A, size_t LDA, const int32_t* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const uint8_t* A, size_t LDA, const uint8_t* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const uint32_t* A, size_t LDA, const uint32_t* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const int64_t* A, size_t LDA, const int64_t* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}
template <>
void rocgraph_clients_unit_check_general(
    size_t M, size_t N, const size_t* A, size_t LDA, const size_t* B, size_t LDB)
{
    ROCGRAPH_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void rocgraph_clients_unit_check_enum(const rocgraph_index_base a, const rocgraph_index_base b)
{
    ASSERT_TRUE(a == b);
}

template <>
void rocgraph_clients_unit_check_enum(const rocgraph_order a, const rocgraph_order b)
{
    ASSERT_TRUE(a == b);
}

template <>
void rocgraph_clients_unit_check_enum(const rocgraph_direction a, const rocgraph_direction b)
{
    ASSERT_TRUE(a == b);
}

template <>
void rocgraph_clients_unit_check_enum(const rocgraph_datatype a, const rocgraph_datatype b)
{
    ASSERT_TRUE(a == b);
}

template <>
void rocgraph_clients_unit_check_enum(const rocgraph_indextype a, const rocgraph_indextype b)
{
    ASSERT_TRUE(a == b);
}

void rocgraph_clients_unit_check_garray(rocgraph_indextype ind_type,
                                        size_t             size,
                                        const void*        source,
                                        const void*        target)
{
    void* s;
    CHECK_HIP_SUCCESS(rocgraph_hipHostMalloc(&s, rocgraph_indextype_sizeof(ind_type) * size));
    CHECK_HIP_SUCCESS(
        hipMemcpy(s, source, rocgraph_indextype_sizeof(ind_type) * size, hipMemcpyDeviceToHost));
    void* t;
    CHECK_HIP_SUCCESS(rocgraph_hipHostMalloc(&t, rocgraph_indextype_sizeof(ind_type) * size));
    CHECK_HIP_SUCCESS(
        hipMemcpy(t, target, rocgraph_indextype_sizeof(ind_type) * size, hipMemcpyDeviceToHost));
    switch(ind_type)
    {
    case rocgraph_indextype_i32:
    {
        rocgraph_clients_unit_check_segments<int32_t>(size, (const int32_t*)s, (const int32_t*)t);
        break;
    }
    case rocgraph_indextype_i64:
    {
        rocgraph_clients_unit_check_segments<int64_t>(size, (const int64_t*)s, (const int64_t*)t);
        break;
    }
    case rocgraph_indextype_u16:
    {
        break;
    }
    }
    CHECK_HIP_SUCCESS(rocgraph_hipFree(s));
    CHECK_HIP_SUCCESS(rocgraph_hipFree(t));
}

void rocgraph_clients_unit_check_garray(rocgraph_datatype val_type,
                                        size_t            size,
                                        const void*       source,
                                        const void*       target)
{
    void* s;
    CHECK_HIP_SUCCESS(rocgraph_hipHostMalloc(&s, rocgraph_datatype_sizeof(val_type) * size));
    CHECK_HIP_SUCCESS(
        hipMemcpy(s, source, rocgraph_datatype_sizeof(val_type) * size, hipMemcpyDeviceToHost));
    void* t;
    CHECK_HIP_SUCCESS(rocgraph_hipHostMalloc(&t, rocgraph_datatype_sizeof(val_type) * size));
    CHECK_HIP_SUCCESS(
        hipMemcpy(t, target, rocgraph_datatype_sizeof(val_type) * size, hipMemcpyDeviceToHost));
    switch(val_type)
    {
    case rocgraph_datatype_f32_r:
    {
        rocgraph_clients_unit_check_segments<float>(size, (const float*)s, (const float*)t);
        break;
    }
    case rocgraph_datatype_f64_r:
    {
        rocgraph_clients_unit_check_segments<double>(size, (const double*)s, (const double*)t);
        break;
    }
    case rocgraph_datatype_i32_r:
    {
        rocgraph_clients_unit_check_segments<int32_t>(size, (const int32_t*)s, (const int32_t*)t);
        break;
    }
    case rocgraph_datatype_u32_r:
    {
        //      rocgraph_clients_unit_check_segments<uint32_t>(size,(const uint32_t*) source, (const uint32_t*) t);
        break;
    }
    case rocgraph_datatype_i8_r:
    {
        rocgraph_clients_unit_check_segments<int8_t>(size, (const int8_t*)s, (const int8_t*)t);
        break;
    }
    case rocgraph_datatype_u8_r:
    {
        rocgraph_clients_unit_check_segments<uint8_t>(
            size, (const uint8_t*)source, (const uint8_t*)target);
        break;
    }
    }
    CHECK_HIP_SUCCESS(rocgraph_hipFree(s));
    CHECK_HIP_SUCCESS(rocgraph_hipFree(t));
}

template <typename T>
void rocgraph_clients_expect_array_lt_scalar(size_t size, const T* a, T s)
{
    size_t count = 0;
    for(size_t i = 0; i < size; ++i)
    {
        if(a[i] >= s)
        {
            ++count;
        }
    }
    if(count > 0)
    {
        ASSERT_EQ(count, 0);
    }
}

template void rocgraph_clients_expect_array_lt_scalar(size_t size, const int32_t* a, int32_t s);
template void rocgraph_clients_expect_array_lt_scalar(size_t size, const int64_t* a, int64_t s);

template <typename T, typename I>
void rocgraph_clients_unit_check_array_indirect(
    size_t M, const T* a, size_t a_inc, const I* a_perm, const T* b, size_t b_inc, const I* b_perm)
{
    for(size_t i = 0; i < M; ++i)
    {
        const T a_val = (a_perm != nullptr) ? a[a_perm[i] * a_inc] : a[i * a_inc];
        const T b_val = (b_perm != nullptr) ? b[b_perm[i] * b_inc] : b[i * b_inc];

#ifdef GOOGLE_TEST
        if(rocgraph_isnan(a_val))
        {
            ASSERT_TRUE(rocgraph_isnan(b_val));
        }
        else if(rocgraph_isinf(a_val))
        {
            ASSERT_TRUE(rocgraph_isinf(b_val));
        }
        else
        {
            ASSERT_EQ(a_val, b_val);
        }
#else
        if(a_val != b_val)
        {
            std::cerr.precision(12);
            std::cerr << "ASSERT_EQ(" << a_val << ", " << b_val << ") failed" << std::endl;
            exit(EXIT_FAILURE);
        }
#endif
    }
}

#define INSTANTIATE(T, I)                                     \
    template void rocgraph_clients_unit_check_array_indirect( \
        size_t, const T*, size_t, const I*, const T*, size_t, const I*)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);

INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int64_t);

#undef INSTANTIATE
