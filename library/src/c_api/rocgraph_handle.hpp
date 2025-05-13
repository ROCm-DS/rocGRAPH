// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "sparse_utility_types.h"
#include <raft/core/handle.hpp>

struct _rocgraph_handle
{
private:
    raft::handle_t*       m_raft_handle{};
    bool                  m_is_raft_handle_allocated{};
    hipStream_t           m_stream{};
    int                   m_device{};
    hipDeviceProp_t       m_properties{};
    int                   m_wavefront_size{};
    int                   m_asic_rev{};
    rocgraph_pointer_mode m_pointer_mode{rocgraph_pointer_mode_host};
    size_t                m_buffer_size{};
    void*                 m_buffer{};
    float*                m_sone{};
    double*               m_done{};

public:
    _rocgraph_handle(void* raft_handle = nullptr);
    ~_rocgraph_handle();

    const raft::handle_t* get_raft_handle() const;
    raft::handle_t*       get_raft_handle();

    int32_t get_wavefront_size() const;

    hipStream_t       get_stream();
    const hipStream_t get_stream() const;
    rocgraph_status   set_stream(hipStream_t that);

    void*       get_buffer();
    const void* get_buffer() const;

    size_t get_buffer_size() const;

    rocgraph_status       set_pointer_mode(rocgraph_pointer_mode that);
    rocgraph_pointer_mode get_pointer_mode() const;

    const hipDeviceProp_t* get_properties() const;
    hipDeviceProp_t*       get_properties();
};
