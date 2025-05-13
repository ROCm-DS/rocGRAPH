// Copyright (c) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifndef RAFT_DISABLE_CUDA
#pragma message(__FILE__ " should only be used in CUDA-disabled RAFT builds." \
                         " Please use equivalent .cuh header instead.")
#else
// It is safe to include this cuh file in an hpp header because all CUDA code
// is ifdef'd out for CUDA-disabled builds.
#include <raft/core/mdbuffer.cuh>
#endif
