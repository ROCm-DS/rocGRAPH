/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_MATRIX_DENSE_HPP
#define ROCGRAPH_MATRIX_DENSE_HPP

#include "rocgraph_vector.hpp"

template <memory_mode::value_t MODE, typename T, typename I = rocgraph_int>
struct dense_matrix_view;

template <typename T, typename I = rocgraph_int>
using host_dense_matrix_view = dense_matrix_view<memory_mode::host, T, I>;

template <typename T, typename I = rocgraph_int>
using device_dense_matrix_view = dense_matrix_view<memory_mode::device, T, I>;

template <typename T, typename I = rocgraph_int>
using managed_dense_matrix_view = dense_matrix_view<memory_mode::managed, T, I>;

template <typename T>
using host_dense_matrix32_t = host_dense_matrix_view<T, int32_t>;
template <typename T>
using host_dense_matrix64_t = host_dense_matrix_view<T, int64_t>;

template <typename T>
using device_dense_matrix32_t = device_dense_matrix_view<T, int32_t>;
template <typename T>
using device_dense_matrix64_t = device_dense_matrix_view<T, int64_t>;

template <typename T>
using managed_dense_matrix32_t = managed_dense_matrix_view<T, int32_t>;
template <typename T>
using managed_dense_matrix64_t = managed_dense_matrix_view<T, int64_t>;

template <memory_mode::value_t MODE, typename T, typename I = rocgraph_int>
struct dense_matrix;

template <typename T, typename I = rocgraph_int>
using host_dense_matrix = dense_matrix<memory_mode::host, T, I>;
template <typename T, typename I = rocgraph_int>
using device_dense_matrix = dense_matrix<memory_mode::device, T, I>;
template <typename T, typename I = rocgraph_int>
using managed_dense_matrix = dense_matrix<memory_mode::managed, T, I>;

template <typename T>
using host_dense_matrix32 = host_dense_matrix<T, int32_t>;
template <typename T>
using host_dense_matrix64 = host_dense_matrix<T, int64_t>;

template <typename T>
using device_dense_matrix32 = device_dense_matrix<T, int32_t>;
template <typename T>
using device_dense_matrix64 = device_dense_matrix<T, int64_t>;

template <typename T>
using managed_dense_matrix32 = managed_dense_matrix<T, int32_t>;
template <typename T>
using managed_dense_matrix64 = managed_dense_matrix<T, int64_t>;

/* ============================================================================================ */
/*! \brief Meta-data of a dense matrix on host, device and managed memory.
 *  \details
 *  \p This structure does not create any memory, therefore copy constructors are deleted explicitly.
 *  \p Assignment operators are allowed and are triggering transfer between memories.
 *  \p Cast operators are set to pointer of \ref T.
 *  \p For convenience, all other routines are transfering data to host if needed.
 *  \p The transfer method takes care of the leading dimension being greater than its allowed minimum.
 */
template <memory_mode::value_t MODE, typename T, typename I>
struct dense_matrix_view
{
private:
    T* val{};

public:
    I              m{};
    I              n{};
    int64_t        ld{1};
    rocgraph_order order{rocgraph_order_column};
    dense_matrix_view() {};
    ~dense_matrix_view() {};

    template <memory_mode::value_t THAT_MODE>
    dense_matrix_view<MODE, T, I>(const dense_matrix_view<THAT_MODE, T, I>& that) = delete;
    dense_matrix_view<MODE, T, I>(const dense_matrix_view<MODE, T, I>& that)      = delete;

    template <memory_mode::value_t THAT_MODE>
    inline dense_matrix_view& operator=(const dense_matrix_view<THAT_MODE, T, I>& that)
    {
        this->transfer_from(that);
        return *this;
    }

    inline dense_matrix_view& operator=(const dense_matrix_view& that)
    {
        this->transfer_from(that);
        return *this;
    }

public:
    T* data()
    {
        return this->val;
    }
    const T* data() const
    {
        return this->val;
    }

    dense_matrix_view&
        operator()(I m_, I n_, T* val_, int64_t ld_, rocgraph_order order_ = rocgraph_order_column)
    {
        m     = m_;
        n     = n_;
        val   = val_;
        ld    = ld_;
        order = order_;
        return *this;
    }
    dense_matrix_view(
        I m_, I n_, T* val_, int64_t ld_, rocgraph_order order_ = rocgraph_order_column)
        : val(val_)
        , m(m_)
        , n(n_)
        , ld(ld_)
        , order(order_)
    {

        switch(order_)
        {
        case rocgraph_order_row:
        {
            if(ld_ < n_)
            {
                std::cerr << "dense_matrix constructor, row order 'ld' is invalid:"
                          << " (ld = " << ld_ << ")"
                          << " < "
                          << " (n = " << n_ << ")"
                          << ")" << std::endl;
                exit(1);
            }
            break;
        }
        case rocgraph_order_column:
        {
            if(ld_ < m_)
            {
                std::cerr << "dense_matrix constructor, row order 'ld' is invalid:"
                          << " (ld = " << ld_ << ")"
                          << " < "
                          << " (m = " << m_ << ")"
                          << ")" << std::endl;
                exit(1);
            }
            break;
        }
        }
    };

    operator T*()
    {
        return this->data();
    }

    operator const T*() const
    {
        return this->data();
    }

    void info() const
    {
        std::cout << "m:    " << this->m << std::endl;
        std::cout << "n:    " << this->n << std::endl;
        std::cout << "ld:   " << this->ld << std::endl;
        std::cout << "ptr:  " << (const T*)this->val << std::endl;
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const dense_matrix_view<THAT_MODE, T, I>& that_)
    {
        CHECK_HIP_THROW_ERROR((this->m == that_.m && this->n == that_.n) ? hipSuccess
                                                                         : hipErrorInvalidValue);
        CHECK_HIP_THROW_ERROR((this->order == that_.order) ? hipSuccess : hipErrorInvalidValue);
        if(((this->order == rocgraph_order_column) && (that_.m == that_.ld)
            && (this->m == this->ld))
           || ((this->order == rocgraph_order_row) && (that_.n == that_.ld)
               && (this->n == this->ld)))
        {
            CHECK_HIP_THROW_ERROR(hipMemcpy(((T*)(*this)),
                                            ((const T*)that_),
                                            sizeof(T) * this->m * this->n,
                                            memory_mode::get_hipMemcpyKind(MODE, THAT_MODE)));
        }
        else
        {
            const I num_sequences = (this->order == rocgraph_order_column) ? this->n : this->m;
            const I size_sequence = (this->order == rocgraph_order_column) ? this->m : this->n;
            CHECK_HIP_THROW_ERROR((this->ld >= size_sequence && that_.ld >= size_sequence)
                                      ? hipSuccess
                                      : hipErrorInvalidValue);
            for(I j = 0; j < num_sequences; ++j)
            {
                CHECK_HIP_THROW_ERROR(hipMemcpy(((T*)*this) + j * this->ld,
                                                ((const T*)that_) + j * that_.ld,
                                                sizeof(T) * size_sequence,
                                                memory_mode::get_hipMemcpyKind(MODE, THAT_MODE)));
            }
        }
    }

    void print() const;
    template <memory_mode::value_t THAT_MODE>
    void unit_check(const dense_matrix_view<THAT_MODE, T, I>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T, I> on_host(*this);
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
                rocgraph_clients_unit_check_scalar<I>(this->m, that_.m);
                rocgraph_clients_unit_check_scalar<I>(this->n, that_.n);
                rocgraph_clients_unit_check_enum(this->order, that_.order);

                switch(this->order)
                {
                case rocgraph_order_column:
                {
                    rocgraph_clients_unit_check_general<T>(
                        this->m, this->n, *this, this->ld, that_, that_.ld);
                    break;
                }

                case rocgraph_order_row:
                {
                    rocgraph_clients_unit_check_general<T>(
                        this->n, this->m, *this, this->ld, that_, that_.ld);
                    break;
                }
                }
                break;
            }
            case memory_mode::device:
            {
                dense_matrix<memory_mode::host, T, I> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const dense_matrix_view<THAT_MODE, T, I>& that_,
                    floating_data_t<T> tol = rocgraph_clients_default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T, I> on_host(*this);
            on_host.near_check(that_, tol);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::host:
            case memory_mode::managed:
            {
                rocgraph_clients_unit_check_scalar<I>(this->m, that_.m);
                rocgraph_clients_unit_check_scalar<I>(this->n, that_.n);
                rocgraph_clients_unit_check_enum(this->order, that_.order);

                switch(this->order)
                {
                case rocgraph_order_column:
                {
                    rocgraph_clients_near_check_general<T>(
                        this->m, this->n, *this, this->ld, that_, that_.ld, tol);
                    break;
                }

                case rocgraph_order_row:
                {
                    //
                    // Little trick
                    // If this poses a problem, we need to refactor unit_check_general.
                    //
                    rocgraph_clients_near_check_general<T>(
                        this->n, this->m, *this, this->ld, that_, that_.ld, tol);
                    break;
                }
                }

                break;
            }
            case memory_mode::device:
            {
                dense_matrix<memory_mode::host, T, I> that(that_);
                this->near_check(that, tol);
                break;
            }
            }
            break;
        }
        }
    }
};

/* ============================================================================================ */
/*! \brief Implementation of a dense matrix responsible of the memory allocation.
 */
template <memory_mode::value_t MODE, typename T, typename I>
struct dense_matrix : public dense_matrix_view<MODE, T, I>
{
private:
    using allocator = rocgraph_allocator<MODE, T>;

public:
    dense_matrix() {};
    ~dense_matrix()
    {
        if(this->data() != nullptr)
        {
#ifdef GOOGLE_TEST
            allocator::check_guards(this->data(), size_t(this->m) * size_t(this->n));
#endif
            allocator::free(this->data());
        }
    };

    template <memory_mode::value_t THAT_MODE>
    inline dense_matrix& operator=(const dense_matrix_view<THAT_MODE, T, I>& that)
    {
        this->transfer_from(that);
        return *this;
    }

    /*! \brief Copy constructor */
    dense_matrix(const dense_matrix<MODE, T, I>& that, bool transfer = true)
        : dense_matrix_view<MODE, T, I>(that.m,
                                        that.n,
                                        allocator::malloc(size_t(that.m) * size_t(that.n)),
                                        (that.order == rocgraph_order_column) ? that.m : that.n,
                                        that.order)
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    dense_matrix(I m_, I n_, rocgraph_order order_ = rocgraph_order_column)
        : dense_matrix_view<MODE, T, I>(m_,
                                        n_,
                                        allocator::malloc(size_t(m_) * size_t(n_)),
                                        (order_ == rocgraph_order_column) ? m_ : n_,
                                        order_) {};

    /*! \brief Copy constructor from a dense_matrix_view with the same memory mode. */
    explicit dense_matrix(const dense_matrix_view<MODE, T, I>& that, bool transfer = true)
        : dense_matrix_view<MODE, T, I>(that.m,
                                        that.n,
                                        allocator::malloc(size_t(that.m) * size_t(that.n)),
                                        (that.order == rocgraph_order_column) ? that.m : that.n,
                                        that.order)
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    /*! \brief Copy constructor from a dense_matrix_view with a different memory mode*/
    template <memory_mode::value_t THAT_MODE>
    explicit dense_matrix(const dense_matrix_view<THAT_MODE, T, I>& that, bool transfer = true)
        : dense_matrix_view<MODE, T, I>(that.m,
                                        that.n,
                                        allocator::malloc(size_t(that.m) * size_t(that.n)),
                                        (that.order == rocgraph_order_column) ? that.m : that.n,
                                        that.order)
    {

        if(transfer)
        {
            this->transfer_from(that);
        }
    }
};

template <memory_mode::value_t MODE, typename T, typename I>
void dense_matrix_view<MODE, T, I>::print() const
{
    switch(MODE)
    {
    case memory_mode::host:
    case memory_mode::managed:
    {
        switch(this->order)
        {
        case rocgraph_order_column:
        {
            for(I i = 0; i < this->m; ++i)
            {
                for(I j = 0; j < this->n; ++j)
                {
                    std::cout << " " << this->val[j * this->ld + i];
                }
                std::cout << std::endl;
            }
            break;
        }
        case rocgraph_order_row:
        {
            for(I i = 0; i < this->m; ++i)
            {
                for(I j = 0; j < this->n; ++j)
                {
                    std::cout << " " << this->val[i * this->ld + j];
                }
                std::cout << std::endl;
            }
            break;
        }
        }
        break;
    }
    case memory_mode::device:
    {
        dense_matrix<memory_mode::host, T, I> on_host(*this);
        on_host.print();
        break;
    }
    }
};

#endif // ROCGRAPH_MATRIX_DENSE_HPP
