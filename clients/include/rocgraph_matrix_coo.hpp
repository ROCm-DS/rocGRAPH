/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_COO_HPP
#define ROCGRAPH_MATRIX_COO_HPP

#include "rocgraph_vector.hpp"

template <memory_mode::value_t MODE, typename T, typename I = rocgraph_int>
struct coo_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    I                     m{};
    I                     n{};
    int64_t               nnz{};
    rocgraph_index_base   base{};
    rocgraph_storage_mode storage_mode{rocgraph_storage_mode_sorted};
    array_t<I>            row_ind{};
    array_t<I>            col_ind{};
    array_t<T>            val{};

    coo_matrix() {};
    ~coo_matrix() {};

    coo_matrix(I m_, I n_, int64_t nnz_, rocgraph_index_base base_)
        : m(m_)
        , n(n_)
        , nnz(nnz_)
        , base(base_)
        , row_ind(nnz_)
        , col_ind(nnz_)
        , val(nnz_) {};

    explicit coo_matrix(const coo_matrix<MODE, T, I>& that_, bool transfer = true)
        : coo_matrix<MODE, T, I>(that_.m, that_.n, that_.nnz, that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit coo_matrix(const coo_matrix<THAT_MODE, T, I>& that_, bool transfer = true)
        : coo_matrix<MODE, T, I>(that_.m, that_.n, that_.nnz, that_.base)
    {
        if(transfer)
        {
            this->transfer_from(that_);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const coo_matrix<THAT_MODE, T, I>& that)
    {
        CHECK_HIP_THROW_ERROR((this->m == that.m && this->n == that.n && this->nnz == that.nnz
                               && this->base == that.base)
                                  ? hipSuccess
                                  : hipErrorInvalidValue);

        this->row_ind.transfer_from(that.row_ind);
        this->col_ind.transfer_from(that.col_ind);
        this->val.transfer_from(that.val);
    };

    void define(I m_, I n_, int64_t nnz_, rocgraph_index_base base_)
    {
        if(m_ != this->m)
        {
            this->m = m_;
        }

        if(n_ != this->n)
        {
            this->n = n_;
        }

        if(nnz_ != this->nnz)
        {
            this->nnz = nnz_;
            this->row_ind.resize(this->nnz);
            this->col_ind.resize(this->nnz);
            this->val.resize(this->nnz);
        }

        if(base_ != this->base)
        {
            this->base = base_;
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const coo_matrix<THAT_MODE, T, I>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            coo_matrix<memory_mode::host, T, I> on_host(*this);
            on_host.unit_check(that_);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                rocgraph_clients_unit_check_scalar(this->m, that_.m);
                rocgraph_clients_unit_check_scalar(this->n, that_.n);
                rocgraph_clients_unit_check_scalar(this->nnz, that_.nnz);
                rocgraph_clients_unit_check_enum(this->base, that_.base);

                this->row_ind.unit_check(that_.row_ind);
                this->col_ind.unit_check(that_.col_ind);
                this->val.unit_check(that_.val);

                break;
            }
            case memory_mode::device:
            {
                coo_matrix<memory_mode::host, T, I> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const coo_matrix<THAT_MODE, T, I>& that_,
                    floating_data_t<T> tol = rocgraph_clients_default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            coo_matrix<memory_mode::host, T, I> on_host(*this);
            on_host.near_check(that_, tol);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                rocgraph_clients_unit_check_scalar(this->m, that_.m);
                rocgraph_clients_unit_check_scalar(this->n, that_.n);
                rocgraph_clients_unit_check_scalar(this->nnz, that_.nnz);
                rocgraph_clients_unit_check_enum(this->base, that_.base);

                this->row_ind.unit_check(that_.row_ind);
                this->col_ind.unit_check(that_.col_ind);
                this->val.near_check(that_.val, tol);
                break;
            }
            case memory_mode::device:
            {
                coo_matrix<memory_mode::host, T, I> that(that_);
                this->near_check(that, tol);
                break;
            }
            }
            break;
        }
        }
    }

    void info() const
    {
        std::cout << "INFO COO " << std::endl;
        std::cout << " m    : " << this->m << std::endl;
        std::cout << " n    : " << this->n << std::endl;
        std::cout << " nnz  : " << this->nnz << std::endl;
        std::cout << " base : " << this->base << std::endl;
    }

    void print() const
    {
        switch(MODE)
        {
        case memory_mode::host:
        case memory_mode::managed:
        {
            const I* pi = (const I*)this->row_ind;
            const I* pj = (const I*)this->col_ind;
            const T* v  = (const T*)val;

            std::cout << "COO MATRIX" << std::endl;
            std::cout << "M:" << this->m << std::endl;
            std::cout << "N:" << this->n << std::endl;
            std::cout << "NNZ:" << this->nnz << std::endl;
            std::cout << "BASE:" << this->base << std::endl;
            for(int64_t k = 0; k < this->nnz; ++k)
            {
                I i = pi[k] - this->base;
                I j = pj[k] - this->base;
                std::cout << "( " << i << ", " << j << ", " << v[k] << " )" << std::endl;
            }
            break;
        }
        case memory_mode::device:
        {
            coo_matrix<memory_mode::host, T, I> on_host(*this);
            on_host.print();
            break;
        }
        }
    }
};

template <typename T, typename I = rocgraph_int>
using host_coo_matrix = coo_matrix<memory_mode::host, T, I>;
template <typename T, typename I = rocgraph_int>
using device_coo_matrix = coo_matrix<memory_mode::device, T, I>;
template <typename T, typename I = rocgraph_int>
using managed_coo_matrix = coo_matrix<memory_mode::managed, T, I>;

#endif // ROCGRAPH_MATRIX_COO_HPP
