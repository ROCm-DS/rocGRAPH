// Copyright (c) 2020-2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * This file is deprecated and will be removed in a future release.
 */
#include "raft/core/device_mdarray.hpp"
#include "raft/core/device_mdspan.hpp"
#include "raft/core/device_span.hpp"
#include "raft/core/handle.hpp"

#include <string>

namespace raft
{

    /* Function for testing RAFT include
 *
 * @return message indicating RAFT has been included successfully*/
    inline std::string test_raft()
    {
        std::string status = "RAFT Setup successfully";
        return status;
    }

} // namespace raft
