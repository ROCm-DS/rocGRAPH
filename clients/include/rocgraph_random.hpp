/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_RANDOM_HPP
#define ROCGRAPH_RANDOM_HPP

#include "rocgraph_math.hpp"

#include <random>
#include <type_traits>

/* ==================================================================================== */
// Random number generator

using rocgraph_rng_t = std::mt19937;

void rocgraph_rng_set(rocgraph_rng_t a);

void rocgraph_seed_set(rocgraph_rng_t a);

void rocgraph_rng_nan_set(rocgraph_rng_t a);

rocgraph_rng_t& rocgraph_rng_get();

rocgraph_rng_t& rocgraph_seed_get();

rocgraph_rng_t& rocgraph_rng_nan_get();

// extern  rocgraph_rng_t rocgraph_rng, rocgraph_seed, rocgraph_rng_nan;

extern int rocgraph_rand_uniform_float_idx;
extern int rocgraph_rand_uniform_double_idx;
extern int rocgraph_rand_normal_double_idx;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocgraph_seedrand()
{
    rocgraph_rng_set(rocgraph_seed_get());
    rocgraph_rng_nan_set(rocgraph_seed_get());

    rocgraph_rand_uniform_float_idx  = 0;
    rocgraph_rand_uniform_double_idx = 0;
    rocgraph_rand_normal_double_idx  = 0;
}

int    rocgraph_uniform_int(int a, int b);
float  rocgraph_uniform_float(float a, float b);
double rocgraph_uniform_double(double a, double b);
double rocgraph_normal_double();

/* ==================================================================================== */
/*! \brief  Random number generator which generates NaN values */
class rocgraph_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union u_t
        {
            u_t() {}
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(rocgraph_rng_nan_get());
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(rocgraph_rng_nan_get());
    }

    // Random int8_t
    explicit operator int8_t()
    {
        return (int8_t)std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                                          std::numeric_limits<int8_t>::max())(
            rocgraph_rng_nan_get());
    }

    // Random char
    explicit operator char()
    {
        return (char)std::uniform_int_distribution<int>(std::numeric_limits<char>::min(),
                                                        std::numeric_limits<char>::max())(
            rocgraph_rng_nan_get());
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }
};

/* ==================================================================================== */
/* generate random number :*/

/*! \brief  generate a random number in range [a,b] using integer numbers*/
template <typename T>
inline T random_generator_exact(int a = 1, int b = 10)
{
    return std::uniform_int_distribution<int>(a, b)(rocgraph_rng_get());
}

/*! \brief  generate a random number in range [a,b]*/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T random_generator(T a = static_cast<T>(1), T b = static_cast<T>(10))
{
    return random_generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline T random_generator(T a = static_cast<T>(0), T b = static_cast<T>(1))
{
    return std::uniform_real_distribution<T>(a, b)(rocgraph_rng_get());
}

/*! \brief  generate a random number in range [a,b] from a predetermined finite cache using integer numbers*/
template <typename T>
inline T random_cached_generator_exact(int a = 1, int b = 10)
{
    return rocgraph_uniform_int(a, b);
}

template <>
inline float random_cached_generator_exact(int a, int b)
{
    return static_cast<float>(rocgraph_uniform_int(a, b));
}

template <>
inline double random_cached_generator_exact(int a, int b)
{
    return static_cast<double>(rocgraph_uniform_int(a, b));
}

/*! \brief  generate a random number in range [a,b] from a predetermined finite cache*/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T random_cached_generator(T a = static_cast<T>(1), T b = static_cast<T>(10))
{
    return random_cached_generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline T random_cached_generator(T a = static_cast<T>(0), T b = static_cast<T>(1))
{
    return static_cast<T>(rocgraph_uniform_float(a, b));
}

template <>
inline double random_cached_generator(double a, double b)
{
    return rocgraph_uniform_double(a, b);
}

/*! \brief generate a random normally distributed number around 0 with stddev 1 from a predetermined finite cache */
template <typename T>
inline T random_cached_generator_normal()
{
    return static_cast<T>(rocgraph_normal_double());
}

#endif // ROCGRAPH_RANDOM_HPP
