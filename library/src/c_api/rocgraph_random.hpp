// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/random/rng_state.hpp>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_rng_state_t
        {
            raft::random::RngState rng_state_;
        };

    } // namespace c_api
} // namespace rocgraph
