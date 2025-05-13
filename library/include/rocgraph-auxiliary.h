// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file*/
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCGRAPH_AUXILIARY_H
#define ROCGRAPH_AUXILIARY_H

#include "rocgraph-types.h"

#include "internal/aux/rocgraph_centrality_result_aux.h"
#include "internal/aux/rocgraph_clustering_result_aux.h"
#include "internal/aux/rocgraph_coo_aux.h"
#include "internal/aux/rocgraph_coo_list_aux.h"
#include "internal/aux/rocgraph_core_result_aux.h"
#include "internal/aux/rocgraph_data_mask_aux.h"
#include "internal/aux/rocgraph_degrees_result_aux.h"
#include "internal/aux/rocgraph_edge_centrality_result_aux.h"
#include "internal/aux/rocgraph_error_aux.h"
#include "internal/aux/rocgraph_extract_paths_result_aux.h"
#include "internal/aux/rocgraph_graph_aux.h"
#include "internal/aux/rocgraph_handle_aux.h"
#include "internal/aux/rocgraph_hierarchical_clustering_result_aux.h"
#include "internal/aux/rocgraph_hits_result_aux.h"
#include "internal/aux/rocgraph_induced_subgraph_result_aux.h"
#include "internal/aux/rocgraph_k_core_result_aux.h"
#include "internal/aux/rocgraph_labeling_result_aux.h"
#include "internal/aux/rocgraph_paths_result_aux.h"
#include "internal/aux/rocgraph_random_walk_result_aux.h"
#include "internal/aux/rocgraph_rng_state_aux.h"
#include "internal/aux/rocgraph_sample_result_aux.h"
#include "internal/aux/rocgraph_sampling_options_aux.h"
#include "internal/aux/rocgraph_similarity_result_aux.h"
#include "internal/aux/rocgraph_triangle_count_result_aux.h"
#include "internal/aux/rocgraph_type_erased_device_array_aux.h"
#include "internal/aux/rocgraph_type_erased_device_array_view_aux.h"
#include "internal/aux/rocgraph_type_erased_host_array_aux.h"
#include "internal/aux/rocgraph_type_erased_host_array_view_aux.h"
#include "internal/aux/rocgraph_vertex_pairs_aux.h"

#include "internal/rocgraph-debug.h"
#include "internal/rocgraph-memstat.h"

#endif /* ROCGRAPH_AUXILIARY_H */
