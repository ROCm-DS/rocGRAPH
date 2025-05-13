/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <rocgraph/rocgraph.h>

template <typename weight_t, typename vertex_t>
void rocgraph_clients_create_test_graph(const rocgraph_handle_t* p_handle,
                                        const vertex_t*          h_src,
                                        const vertex_t*          h_dst,
                                        const weight_t*          h_wgt,
                                        size_t                   num_edges,
                                        rocgraph_bool            store_transposed,
                                        rocgraph_bool            renumber,
                                        rocgraph_bool            is_symmetric,
                                        rocgraph_graph_t**       p_graph,
                                        rocgraph_error_t**       ret_error);

void rocgraph_clients_create_sg_test_graph(const rocgraph_handle_t* p_handle,
                                           rocgraph_data_type_id    vertex_tid,
                                           rocgraph_data_type_id    edge_tid,
                                           void*                    h_src,
                                           void*                    h_dst,
                                           rocgraph_data_type_id    weight_tid,
                                           void*                    h_wgt,
                                           rocgraph_data_type_id    edge_type_tid,
                                           void*                    h_edge_type,
                                           rocgraph_data_type_id    edge_id_tid,
                                           void*                    h_edge_id,
                                           size_t                   num_edges,
                                           rocgraph_bool            store_transposed,
                                           rocgraph_bool            renumber,
                                           rocgraph_bool            is_symmetric,
                                           rocgraph_bool            is_multigraph,
                                           rocgraph_graph_t**       graph,
                                           rocgraph_error_t**       ret_error);
