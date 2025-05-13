/*! \file */

// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_random.hpp"

#include <iostream>

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
rocgraph_rng_t rocgraph_rng(69069);
rocgraph_rng_t rocgraph_rng_nan(69069);
rocgraph_rng_t rocgraph_seed(rocgraph_rng);

void rocgraph_rng_set(rocgraph_rng_t a)
{
    rocgraph_rng = a;
}

void rocgraph_seed_set(rocgraph_rng_t a)
{
    rocgraph_seed = a;
}

void rocgraph_rng_nan_set(rocgraph_rng_t a)
{
    rocgraph_rng_nan = a;
}

rocgraph_rng_t& rocgraph_rng_get()
{
    return rocgraph_rng;
}

rocgraph_rng_t& rocgraph_seed_get()
{
    return rocgraph_seed;
}

rocgraph_rng_t& rocgraph_rng_nan_get()
{
    return rocgraph_rng_nan;
}

#define RANDOM_CACHE_SIZE 1024

int rocgraph_rand_uniform_float_idx;
int rocgraph_rand_uniform_double_idx;
int rocgraph_rand_normal_double_idx;

static int s_rand_uniform_float_init  = 0;
static int s_rand_uniform_double_init = 0;
static int s_rand_normal_double_init  = 0;

// float random uniform numbers between 0.0f - 1.0f
static float s_rand_uniform_float_array[RANDOM_CACHE_SIZE];
// double random uniform numbers between 0.0 - 1.0
static double s_rand_uniform_double_array[RANDOM_CACHE_SIZE];
// double random normal numbers between 0.0 - 1.0
static double s_rand_normal_double_array[RANDOM_CACHE_SIZE];

float rocgraph_uniform_float(float a, float b)
{
    if(!s_rand_uniform_float_init)
    {
        for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
        {
            s_rand_uniform_float_array[i]
                = std::uniform_real_distribution<float>(0.0f, 1.0f)(rocgraph_rng_get());
        }
        s_rand_uniform_float_init = 1;
    }

    rocgraph_rand_uniform_float_idx
        = (rocgraph_rand_uniform_float_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + s_rand_uniform_float_array[rocgraph_rand_uniform_float_idx] * (b - a);
}

double rocgraph_uniform_double(double a, double b)
{
    if(!s_rand_uniform_double_init)
    {
        for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
        {
            s_rand_uniform_double_array[i]
                = std::uniform_real_distribution<double>(0.0, 1.0)(rocgraph_rng_get());
        }
        s_rand_uniform_double_init = 1;
    }

    rocgraph_rand_uniform_double_idx
        = (rocgraph_rand_uniform_double_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + s_rand_uniform_double_array[rocgraph_rand_uniform_double_idx] * (b - a);
}

int rocgraph_uniform_int(int a, int b)
{
    return rocgraph_uniform_float(static_cast<float>(a), static_cast<float>(b));
}

double rocgraph_normal_double()
{
    if(!s_rand_normal_double_init)
    {
        for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
        {
            s_rand_normal_double_array[i]
                = std::normal_distribution<double>(0.0, 1.0)(rocgraph_rng_get());
        }
        s_rand_normal_double_init = 1;
    }

    rocgraph_rand_normal_double_idx
        = (rocgraph_rand_normal_double_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return s_rand_normal_double_array[rocgraph_rand_normal_double_idx];
}
