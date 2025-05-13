/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_EXPORTER_HPP
#define ROCGRAPH_EXPORTER_HPP

#include "rocgraph_check.hpp"
#include "rocgraph_traits.hpp"

#include "rocgraph_matrix_coo.hpp"
#include "rocgraph_matrix_csx.hpp"
#include "rocgraph_matrix_dense.hpp"
#include "rocgraph_vector.hpp"

template <typename IMPL>
class rocgraph_exporter
{
protected:
    rocgraph_exporter()  = default;
    ~rocgraph_exporter() = default;

public:
    template <typename T, typename I = rocgraph_int, typename J = rocgraph_int>
    rocgraph_status write_graph_csx(rocgraph_direction dir,
                                    J                  m,
                                    J                  n,
                                    I                  nnz,
                                    const I* __restrict__ ptr,
                                    const J* __restrict__ ind,
                                    const T* __restrict__ val,
                                    rocgraph_index_base base)

    {
        return static_cast<IMPL&>(*this).write_graph_csx(dir, m, n, nnz, ptr, ind, val, base);
    }
    template <typename T, typename I = rocgraph_int>
    rocgraph_status write_dense_vector(I size, const T* __restrict__ x, I incx)
    {
        return static_cast<IMPL&>(*this).write_dense_vector(size, x, incx);
    }

    template <typename T, typename I = rocgraph_int>
    rocgraph_status
        write_dense_matrix(rocgraph_order order, I m, I n, const T* __restrict__ x, I ld)
    {
        return static_cast<IMPL&>(*this).write_dense_matrix(order, m, n, x, ld);
    }

    template <typename T, typename I = rocgraph_int>
    rocgraph_status write_graph_coo(I m,
                                    I n,
                                    I nnz,
                                    const I* __restrict__ row_ind,
                                    const I* __restrict__ col_ind,
                                    const T* __restrict__ val,
                                    rocgraph_index_base base)
    {
        return static_cast<IMPL&>(*this).write_graph_coo(m, n, nnz, row_ind, col_ind, val, base);
    }

    template <rocgraph_direction DIRECTION, typename T, typename I, typename J>
    rocgraph_status write(const host_csx_matrix<DIRECTION, T, I, J>& that_)
    {
        return this->write_graph_csx<T, I, J>(
            that_.dir, that_.m, that_.n, that_.nnz, that_.ptr, that_.ind, that_.val, that_.base);
    }

    template <typename T, typename I>
    rocgraph_status write(const host_coo_matrix<T, I>& that_)
    {
        return this->write_graph_coo<T, I>(
            that_.m, that_.n, that_.nnz, that_.row_ind, that_.col_ind, that_.val, that_.base);
    }

    template <typename T, typename I>
    rocgraph_status write(const host_dense_matrix<T, I>& that_)
    {
        return this->write_dense_matrix<T, I>(
            that_.order, that_.m, that_.n, that_.data(), that_.ld);
    }

    template <typename T>
    rocgraph_status write(const host_dense_vector<T>& that_)
    {
        static constexpr size_t one = static_cast<size_t>(1);
        return this->write_dense_vector<T, size_t>(that_.size(), that_.val, one);
    }
};

#endif
