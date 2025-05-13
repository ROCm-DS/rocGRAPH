/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef ROCGRAPH_VECTOR_HPP
#define ROCGRAPH_VECTOR_HPP

#include "rocgraph_allocator.hpp"
template <memory_mode::value_t MODE, typename T>
struct dense_vector;

template <typename T>
using host_dense_vector = dense_vector<memory_mode::host, T>;

template <typename T>
using device_dense_vector = dense_vector<memory_mode::device, T>;

template <typename T>
using managed_dense_vector = dense_vector<memory_mode::managed, T>;

#include "rocgraph_check.hpp"
#include "rocgraph_init.hpp"
#include "rocgraph_traits.hpp"

template <memory_mode::value_t MODE, typename T>
struct dense_vector_t
{
protected:
    size_t m_size;
    T*     m_val;

public:
    using value_type = T;
    dense_vector_t(size_t size, T* val);
    dense_vector_t& operator()(size_t size, T* val)
    {
        m_size = size;
        m_val  = val;
        return *this;
    }
    size_t size() const;
    operator T*();
    operator const T*() const;
    T*       data();
    const T* data() const;
    ~dense_vector_t();
    const T* begin() const
    {
        return this->data();
    };
    const T* end() const
    {
        return this->data() + size();
    };
    T* begin()
    {
        return this->data();
    };
    T* end()
    {
        return this->data() + size();
    };
    // Disallow copying or assigning
    dense_vector_t<MODE, T>(const dense_vector_t<MODE, T>&) = delete;
    template <memory_mode::value_t THAT_MODE>
    dense_vector_t<MODE, T>(const dense_vector_t<THAT_MODE, T>&) = delete;

    dense_vector_t<MODE, T>& operator=(const dense_vector_t<MODE, T>& that_)
    {
        this->transfer_from(that_);
        return *this;
    }

    template <memory_mode::value_t THAT_MODE>
    dense_vector_t<MODE, T>& operator=(const dense_vector_t<THAT_MODE, T>& that_)
    {
        this->transfer_from(that_);
    }

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const dense_vector_t<THAT_MODE, T>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            host_dense_vector<T> on_host(*this);
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
                rocgraph_clients_unit_check_scalar<size_t>(this->size(), that_.size());
                rocgraph_clients_unit_check_segments<T>(this->size(), this->data(), that_.data());
                break;
            }
            case memory_mode::device:
            {
                host_dense_vector<T> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }
    template <memory_mode::value_t THAT_MODE>
    void near_check(const dense_vector_t<THAT_MODE, T>& that_,
                    floating_data_t<T> tol_ = rocgraph_clients_default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            host_dense_vector<T> on_host(*this, true);
            on_host.near_check(that_, tol_);
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
                rocgraph_clients_unit_check_scalar<size_t>(this->size(), that_.size());
                rocgraph_clients_near_check_segments<T>(
                    this->size(), this->data(), that_.data(), tol_);
                break;
            }
            case memory_mode::device:
            {
                host_dense_vector<T> that(that_);
                this->near_check(that, tol_);
                break;
            }
            }
            break;
        }
        }
    }
    void print() const;

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const dense_vector_t<THAT_MODE, T>& that)
    {
        //        CHECK_HIP_THROW_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
        auto err = hipMemcpy(this->data(),
                             that.data(),
                             sizeof(T) * that.size(),
                             memory_mode::get_hipMemcpyKind(MODE, THAT_MODE));
        CHECK_HIP_THROW_ERROR(err);
    }

    void transfer_to(std::vector<T>& that) const;
};

template <memory_mode::value_t MODE, typename T>
struct dense_vector : dense_vector_t<MODE, T>
{
private:
    using allocator = rocgraph_allocator<MODE, T>;

public:
    dense_vector()
        : dense_vector_t<MODE, T>(0, nullptr) {};
    ~dense_vector()
    {
#ifdef GOOGLE_TEST
        allocator::check_guards(this->data(), this->size());
#endif
        allocator::free(this->data());
    };
    dense_vector<MODE, T>& operator=(const dense_vector_t<MODE, T>& that_)
    {
        this->resize(that_.size());
        this->transfer_from(that_);
        return *this;
    }

    template <memory_mode::value_t THAT_MODE>
    dense_vector<MODE, T>& operator=(const dense_vector_t<THAT_MODE, T>& that_)
    {
        this->resize(that_.size());
        this->transfer_from(that_);
        return *this;
    }

    hipError_t memcheck() const
    {
        return ((this->m_size == 0) && (this->data() == nullptr))
                   ? hipSuccess
                   : (((this->m_size > 0) && (this->data() != nullptr)) ? hipSuccess
                                                                        : hipErrorOutOfMemory);
    }

    // Tell whether malloc failed

    void resize(size_t s)
    {
        if(s != this->m_size)
        {
            T* val = allocator::malloc(s);
            if(this->m_val)
            {
                memcpy(val, this->m_val, std::min(s, this->m_size));
                allocator::free(this->m_val);
            }
            this->m_val  = val;
            this->m_size = s;
        }
    }

    void assign(size_t s, T val)
    {
        switch(MODE)
        {

        case memory_mode::managed:
        case memory_mode::device:
        {
            if(val != static_cast<T>(0))
            {
                std::cerr << "here error " << val << std::endl;
                exit(1);
            }
            hipMemset(this->m_val, 0, sizeof(T) * s);
            break;
        }

        case memory_mode::host:
        {
            if(val != static_cast<T>(0))
            {
                for(size_t i = 0; i < s; ++i)
                    this->m_val[i] = val;
            }
            else
            {
                memset(this->m_val, 0, sizeof(T) * s);
            }
            break;
        }
        }
    }

    void resize(size_t s, T val)
    {
        if(s != this->m_size)
        {
            T* val = allocator::malloc(s);
            if(this->m_val)
            {
                allocator::free(this->m_val);
            }
            this->m_val  = val;
            this->m_size = s;
        }
        this->assign(s, val);
    }

    explicit dense_vector(size_t size, T init_val)
        : dense_vector_t<MODE, T>(size, allocator::malloc(size))
    {
        this->assign(size, init_val);
    }

    explicit dense_vector(size_t s)
        : dense_vector_t<MODE, T>(s, allocator::malloc(s))
    {
    }

    explicit dense_vector(const dense_vector_t<MODE, T>& that, bool transfer = true)
        : dense_vector_t<MODE, T>(that.size(), allocator::malloc(that.size()))
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit dense_vector(const dense_vector_t<THAT_MODE, T>& that, bool transfer = true)
        : dense_vector_t<MODE, T>(that.size(), allocator::malloc(that.size()))
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }
};

template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::dense_vector_t(size_t size, T* val)
    : m_size(size)
    , m_val(val)
{
}

template <memory_mode::value_t MODE, typename T>
size_t dense_vector_t<MODE, T>::size() const
{
    return this->m_size;
}

template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::operator T*()
{
    return this->m_val;
}

template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::operator const T*() const
{
    return this->m_val;
}

template <memory_mode::value_t MODE, typename T>
T* dense_vector_t<MODE, T>::data()
{
    return this->m_val;
}
template <memory_mode::value_t MODE, typename T>
const T* dense_vector_t<MODE, T>::data() const
{
    return this->m_val;
}
template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::~dense_vector_t()
{
}

template <memory_mode::value_t MODE, typename T>
void dense_vector_t<MODE, T>::print() const
{
    switch(MODE)
    {
    case memory_mode::host:
    case memory_mode::managed:
    {
        size_t   N = this->size();
        const T* x = this->data();
        for(size_t i = 0; i < N; ++i)
        {
            std::cout << " " << x[i] << std::endl;
        }
        break;
    }
    case memory_mode::device:
    {
        dense_vector<memory_mode::host, T> on_host(*this, true);
        on_host.print();
        break;
    }
    }
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses host memory */

template <memory_mode::value_t mode_>
struct memory_traits;

template <>
struct memory_traits<memory_mode::device>
{
    template <typename S>
    using array_t = device_dense_vector<S>;
};

template <>
struct memory_traits<memory_mode::managed>
{
    template <typename S>
    using array_t = managed_dense_vector<S>;
};

template <>
struct memory_traits<memory_mode::host>
{
    template <typename S>
    using array_t = host_dense_vector<S>;
};

#endif // ROCGRAPH_VECTOR_HPP
