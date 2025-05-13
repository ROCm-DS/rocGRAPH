/*! \file */

// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_vector_utils.hpp"

#include <algorithm>

/*! \brief Perform affine transform of bound [-1, 1] on an array of real values. */
template <typename T,
          typename std::enable_if<(std::is_same<T, float>::value || std::is_same<T, double>::value),
                                  int>::type
          = 0>
static void normalize_array(T* v, size_t v_size)
{
    if(v_size > 0)
    {
        auto max_val = v[0];
        auto min_val = v[0];

        for(size_t i = 1; i < v_size; i++)
        {
            max_val = std::max(v[i], max_val);
            min_val = std::min(v[i], min_val);
        }

        // y = (2x - max - min) / (max - min)
        auto denom = static_cast<T>(1) / (max_val - min_val);

        for(size_t i = 0; i < v_size; i++)
        {
            v[i] = (2 * v[i] - max_val - min_val) * denom;
        }
    }
}

template <typename T>
void rocgraph_vector_utils<T>::normalize(host_dense_vector<T>& v)
{
    normalize_array<T>(v.data(), v.size());
}

#define INSTANTIATE(TYPE) template struct rocgraph_vector_utils<TYPE>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE
