// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cudart_utils.hpp>

namespace rocgraph
{
    namespace detail
    {

        /** helper method to get multi-processor count parameter */
        inline int getMultiProcessorCount()
        {
            int devId;
            RAFT_CUDA_TRY(hipGetDevice(&devId));
            int mpCount;
            RAFT_CUDA_TRY(
                hipDeviceGetAttribute(&mpCount, hipDeviceAttributeMultiprocessorCount, devId));
            return mpCount;
        }

    } // namespace detail
} // namespace rocgraph
