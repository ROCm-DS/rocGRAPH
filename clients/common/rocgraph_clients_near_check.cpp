/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_check.hpp"
#include "rocgraph_clients_assert.hpp"
#include "utility.hpp"

#define MAX_TOL_MULTIPLIER 4

template <typename T>
void near_check_general_template(size_t             M,
                                 size_t             N,
                                 const T*           A,
                                 size_t             LDA,
                                 const T*           B,
                                 size_t             LDB,
                                 floating_data_t<T> tol
                                 = rocgraph_clients_default_tolerance<T>::value)
{
    int tolm = 1;
    for(size_t j = 0; j < N; ++j)
    {
        for(size_t i = 0; i < M; ++i)
        {
            T compare_val
                = std::max(std::abs(A[i + j * LDA] * tol), 10 * std::numeric_limits<T>::epsilon());
#ifdef GOOGLE_TEST
            if(rocgraph_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocgraph_isnan(B[i + j * LDB]));
            }
            else if(rocgraph_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocgraph_isinf(B[i + j * LDB]));
            }
            else
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(std::abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    ASSERT_NEAR(A[i + j * LDA], B[i + j * LDB], compare_val);
                }
                tolm = std::max(tolm, k);
            }
#else

            int k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                          << ") failed: " << std::abs(A[i + j * LDA] - B[i + j * LDB])
                          << " exceeds permissive range [" << compare_val << ","
                          << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                exit(EXIT_FAILURE);
            }
            tolm = std::max(tolm, k);
#endif
        }
    }

    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <typename T, typename I>
void rocgraph_clients_near_check_array_indirect(size_t             M,
                                                const T*           a,
                                                size_t             a_inc,
                                                const I*           a_perm,
                                                const T*           b,
                                                size_t             b_inc,
                                                const I*           b_perm,
                                                floating_data_t<T> tol)
{
    int tolm = 1;
    for(size_t i = 0; i < M; ++i)
    {
        const T a_val = (a_perm != nullptr) ? a[a_perm[i] * a_inc] : a[i * a_inc];
        const T b_val = (b_perm != nullptr) ? b[b_perm[i] * b_inc] : b[i * b_inc];

        T compare_val = std::max(std::abs(a_val * tol), 10 * std::numeric_limits<T>::epsilon());
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
            int k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(a_val - b_val) <= compare_val * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                ASSERT_NEAR(a_val, b_val, compare_val);
            }
            tolm = std::max(tolm, k);
        }
#else

        int k;
        for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
        {
            if(std::abs(a_val - b_val) <= compare_val * k)
            {
                break;
            }
        }

        if(k > MAX_TOL_MULTIPLIER)
        {
            std::cerr.precision(12);
            std::cerr << "ASSERT_NEAR(" << a_val << ", " << b_val
                      << ") failed: " << std::abs(a_val - b_val) << " exceeds permissive range ["
                      << compare_val << "," << compare_val * MAX_TOL_MULTIPLIER << " ]"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        tolm = std::max(tolm, k);
#endif
    }
    if(tolm > 1)
    {
        std::cerr
            << "WARNING near_check_array has been permissive with a tolerance multiplier equal to "
            << tolm << std::endl;
    }
}

template <typename T>
void rocgraph_clients_near_check_array(
    size_t M, const T* a, size_t a_inc, const T* b, size_t b_inc, floating_data_t<T> tol)
{
    rocgraph_clients_near_check_array_indirect<T, int32_t>(
        M, a, a_inc, nullptr, b, b_inc, nullptr, tol);
}

template <typename T>
void rocgraph_clients_near_check_general(
    size_t M, size_t N, const T* A, size_t LDA, const T* B, size_t LDB, floating_data_t<T> tol)
{
    near_check_general_template(M, N, A, LDA, B, LDB, tol);
}

//
// Instantiate rocgraph_clients_near_check_general
//
#define INSTANTIATE(T)                                                        \
    template void rocgraph_clients_near_check_general(size_t             M,   \
                                                      size_t             N,   \
                                                      const T*           A,   \
                                                      size_t             LDA, \
                                                      const T*           B,   \
                                                      size_t             LDB, \
                                                      floating_data_t<T> tol)

INSTANTIATE(int32_t);
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE

//
// Instantiate rocgraph_clients_near_check_array
//
#define INSTANTIATE(T)                               \
    template void rocgraph_clients_near_check_array( \
        size_t, const T*, size_t, const T*, size_t, floating_data_t<T>)

INSTANTIATE(int32_t);
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE

//
// Instantiate rocgraph_clients_near_check_array_indirect
//
#define INSTANTIATE(T, I)                                     \
    template void rocgraph_clients_near_check_array_indirect( \
        size_t, const T*, size_t, const I*, const T*, size_t, const I*, floating_data_t<T>)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);

INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int64_t);

#undef INSTANTIATE
