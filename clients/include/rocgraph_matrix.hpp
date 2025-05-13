/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_HPP
#define ROCGRAPH_MATRIX_HPP

#include "rocgraph_check.hpp"
#include "rocgraph_matrix_dense.hpp"

template <typename T>
struct device_scalar;
template <typename T>
struct host_scalar;
template <typename T>
struct managed_scalar;

template <typename T>
struct host_scalar : public host_dense_matrix<T>
{
    host_scalar()
        : host_dense_matrix<T>(1, 1) {};
    explicit host_scalar(const T& value)
        : host_dense_matrix<T>(1, 1)
    {
        T* p = *this;
        p[0] = value;
    };

    explicit host_scalar(const device_scalar<T>& that)
        : host_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }

    explicit host_scalar(const managed_scalar<T>& that)
        : host_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }

    inline host_scalar<T>& operator=(const T& that)
    {
        T* p = *this;
        p[0] = that;
        return *this;
    };
};

template <typename T>
struct device_scalar : public device_dense_matrix<T>
{
    device_scalar()
        : device_dense_matrix<T>(1, 1) {};
    explicit device_scalar(const host_scalar<T>& that)
        : device_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }
    explicit device_scalar(const managed_scalar<T>& that)
        : device_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }
};

template <typename T>
struct managed_scalar : public managed_dense_matrix<T>
{
    managed_scalar()
        : managed_dense_matrix<T>(1, 1) {};

    explicit managed_scalar(const host_scalar<T>& that)
        : managed_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }

    explicit managed_scalar(const device_scalar<T>& that)
        : managed_dense_matrix<T>(1, 1)
    {
        this->transfer_from(that);
    }
};

#include "rocgraph_matrix_coo.hpp"
#include "rocgraph_matrix_csx.hpp"

#endif // ROCGRAPH_MATRIX_HPP.
