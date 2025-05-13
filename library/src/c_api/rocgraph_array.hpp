// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "internal/types/rocgraph_data_type_id.h"
#include "internal/types/rocgraph_type_erased_device_array_t.h"
#include "internal/types/rocgraph_type_erased_device_array_view_t.h"
#include "internal/types/rocgraph_type_erased_host_array_t.h"
#include "internal/types/rocgraph_type_erased_host_array_view_t.h"

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_type_erased_device_array_view_t
        {
            void*                 data_;
            size_t                size_;
            size_t                num_bytes_;
            rocgraph_data_type_id type_;

            template <typename T>
            T* as_type()
            {
                return reinterpret_cast<T*>(data_);
            }

            template <typename T>
            T const* as_type() const
            {
                return reinterpret_cast<T const*>(data_);
            }

            size_t num_bytes() const
            {
                return num_bytes_;
            }
        };

        struct rocgraph_type_erased_device_array_t
        {
            // NOTE: size must be first here because the device buffer is released
            size_t                size_;
            rmm::device_buffer    data_;
            rocgraph_data_type_id type_;

            rocgraph_type_erased_device_array_t(size_t                       size,
                                                size_t                       num_bytes,
                                                rocgraph_data_type_id        type,
                                                rmm::cuda_stream_view const& stream_view)
                : size_(size)
                , data_(num_bytes, stream_view)
                , type_(type)
            {
            }

            template <typename T>
            rocgraph_type_erased_device_array_t(rmm::device_uvector<T>& vec,
                                                rocgraph_data_type_id   type)
                : size_(vec.size())
                , data_(vec.release())
                , type_(type)
            {
            }

            template <typename T>
            T* as_type()
            {
                return reinterpret_cast<T*>(data_.data());
            }

            template <typename T>
            T const* as_type() const
            {
                return reinterpret_cast<T const*>(data_.data());
            }

            auto view()
            {
                return new rocgraph_type_erased_device_array_view_t{
                    data_.data(), size_, data_.size(), type_};
            }
        };

        struct rocgraph_type_erased_host_array_view_t
        {
            std::byte*            data_;
            size_t                size_;
            size_t                num_bytes_;
            rocgraph_data_type_id type_;

            template <typename T>
            T* as_type()
            {
                return reinterpret_cast<T*>(data_);
            }

            template <typename T>
            T const* as_type() const
            {
                return reinterpret_cast<T const*>(data_);
            }

            size_t num_bytes() const
            {
                return num_bytes_;
            }
        };

        struct rocgraph_type_erased_host_array_t
        {
            std::unique_ptr<std::byte[]> data_{nullptr};
            size_t                       size_{0};
            size_t                       num_bytes_{0};
            rocgraph_data_type_id        type_;

            rocgraph_type_erased_host_array_t(size_t                size,
                                              size_t                num_bytes,
                                              rocgraph_data_type_id type)
                : data_(std::make_unique<std::byte[]>(num_bytes))
                , size_(size)
                , num_bytes_(num_bytes)
                , type_(type)
            {
            }

            template <typename T>
            rocgraph_type_erased_host_array_t(std::vector<T>& vec, rocgraph_data_type_id type)
                : size_(vec.size())
                , num_bytes_(vec.size() * sizeof(T))
                , type_(type)
            {
                data_ = std::make_unique<std::byte[]>(num_bytes_);
                std::copy(vec.begin(), vec.end(), reinterpret_cast<T*>(data_.get()));
            }

            auto view()
            {
                return new rocgraph_type_erased_host_array_view_t{
                    data_.get(), size_, num_bytes_, type_};
            }
        };

    } // namespace c_api
} // namespace rocgraph
