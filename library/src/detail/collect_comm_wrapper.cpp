// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_capi_helper.hpp"
#include "structure/detail/structure_utils.cuh"
#include "utilities/collect_comm.cuh"

#include "detail/shuffle_wrappers.hpp"
#include "utilities/misc_utils_device.hpp"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

namespace rocgraph
{
    namespace detail
    {

        template <typename T>
        rmm::device_uvector<T> device_allgatherv(raft::handle_t const&       handle,
                                                 raft::comms::comms_t const& comm,
                                                 raft::device_span<T const>  d_input)
        {
            auto gathered_v = rocgraph::device_allgatherv(handle, comm, d_input);

            return gathered_v;
        }

        template rmm::device_uvector<int32_t>
            device_allgatherv(raft::handle_t const&            handle,
                              raft::comms::comms_t const&      comm,
                              raft::device_span<int32_t const> d_input);

        template rmm::device_uvector<int64_t>
            device_allgatherv(raft::handle_t const&            handle,
                              raft::comms::comms_t const&      comm,
                              raft::device_span<int64_t const> d_input);

        template rmm::device_uvector<float>
            device_allgatherv(raft::handle_t const&          handle,
                              raft::comms::comms_t const&    comm,
                              raft::device_span<float const> d_input);

        template rmm::device_uvector<double>
            device_allgatherv(raft::handle_t const&           handle,
                              raft::comms::comms_t const&     comm,
                              raft::device_span<double const> d_input);

    } // namespace detail
} // namespace rocgraph
