# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set(ROCGRAPH_C_SOURCE
    src/c_api/amd_detail/algorithms.c
    src/c_api/amd_detail/array.c
    src/c_api/amd_detail/centrality_algorithms.c
    src/c_api/amd_detail/community_algorithms.c
    src/c_api/amd_detail/core_algorithms.c
    src/c_api/amd_detail/error.c
    src/c_api/amd_detail/graph_functions.c
    src/c_api/amd_detail/graph_generators.c
    src/c_api/amd_detail/graph.c
    src/c_api/amd_detail/labeling_algorithms.c
    src/c_api/amd_detail/random.c
    src/c_api/amd_detail/resource_handle.c
    src/c_api/amd_detail/sampling_algorithms.c
    src/c_api/amd_detail/similarity_algorithms.c
    src/c_api/amd_detail/traversal_algorithms.c)
