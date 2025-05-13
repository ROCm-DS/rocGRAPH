// Copyright (c) 2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
namespace raft
{
#ifndef RAFT_DISABLE_CUDA
    auto constexpr static const CUDA_ENABLED = true;
#else
    auto constexpr static const CUDA_ENABLED = false;
#endif
} // namespace raft
