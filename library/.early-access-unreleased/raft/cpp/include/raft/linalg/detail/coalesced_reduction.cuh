// Copyright (c) 2022-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

// Always include inline definitions of coalesced reduction, because we do not
// force explicit instantion.
#include "coalesced_reduction-inl.cuh"

// Do include the extern template instantiations when possible.
#ifdef RAFT_COMPILED
#include "coalesced_reduction-ext.cuh"
#endif
