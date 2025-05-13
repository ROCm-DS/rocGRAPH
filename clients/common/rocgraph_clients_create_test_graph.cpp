/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_clients_data_type_id_get.hpp"
#include "rocgraph_test.hpp"

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
                                        rocgraph_error_t**       ret_error)
{

    rocgraph_graph_properties_t properties;

    properties.is_symmetric  = is_symmetric;
    properties.is_multigraph = rocgraph_bool_false;

    const rocgraph_data_type_id vertex_tid = rocgraph_clients_data_type_id_get<vertex_t>();
    const rocgraph_data_type_id weight_tid = rocgraph_clients_data_type_id_get<weight_t>();

    rocgraph_type_erased_device_array_t*      src;
    rocgraph_type_erased_device_array_t*      dst;
    rocgraph_type_erased_device_array_t*      wgt;
    rocgraph_type_erased_device_array_view_t* src_view;
    rocgraph_type_erased_device_array_view_t* dst_view;
    rocgraph_type_erased_device_array_view_t* wgt_view;

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error),
        *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error),
        *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_create(p_handle, num_edges, weight_tid, &wgt, ret_error),
        *ret_error);

    src_view = rocgraph_type_erased_device_array_view(src);
    dst_view = rocgraph_type_erased_device_array_view(dst);
    wgt_view = rocgraph_type_erased_device_array_view(wgt);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                     p_handle, src_view, (rocgraph_byte_t*)h_src, ret_error),
                                 *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                     p_handle, dst_view, (rocgraph_byte_t*)h_dst, ret_error),
                                 *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                     p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, ret_error),
                                 *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sg_graph_create(p_handle,
                                                          &properties,
                                                          src_view,
                                                          dst_view,
                                                          wgt_view,
                                                          nullptr,
                                                          nullptr,
                                                          store_transposed,
                                                          renumber,
                                                          rocgraph_bool_false,
                                                          p_graph,
                                                          ret_error),
                                 *ret_error);

    rocgraph_type_erased_device_array_view_free(wgt_view);
    rocgraph_type_erased_device_array_view_free(dst_view);
    rocgraph_type_erased_device_array_view_free(src_view);
    rocgraph_type_erased_device_array_free(wgt);
    rocgraph_type_erased_device_array_free(dst);
    rocgraph_type_erased_device_array_free(src);
}

#define INSTANTIATE(weight_t, vertex_t)                                   \
    template void rocgraph_clients_create_test_graph<weight_t, vertex_t>( \
        const rocgraph_handle_t* p_handle,                                \
        const vertex_t*          h_src,                                   \
        const vertex_t*          h_dst,                                   \
        const weight_t*          h_wgt,                                   \
        size_t                   num_edges,                               \
        rocgraph_bool            store_transposed,                        \
        rocgraph_bool            renumber,                                \
        rocgraph_bool            is_symmetric,                            \
        rocgraph_graph_t**       p_graph,                                 \
        rocgraph_error_t**       ret_error)

INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int64_t);

#undef INSTANTIATE

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
                                           rocgraph_error_t**       ret_error)
{
    rocgraph_graph_properties_t properties;

    properties.is_symmetric  = is_symmetric;
    properties.is_multigraph = is_multigraph;

    rocgraph_type_erased_device_array_t*      src            = nullptr;
    rocgraph_type_erased_device_array_t*      dst            = nullptr;
    rocgraph_type_erased_device_array_t*      wgt            = nullptr;
    rocgraph_type_erased_device_array_t*      edge_type      = nullptr;
    rocgraph_type_erased_device_array_t*      edge_id        = nullptr;
    rocgraph_type_erased_device_array_view_t* src_view       = nullptr;
    rocgraph_type_erased_device_array_view_t* dst_view       = nullptr;
    rocgraph_type_erased_device_array_view_t* wgt_view       = nullptr;
    rocgraph_type_erased_device_array_view_t* edge_type_view = nullptr;
    rocgraph_type_erased_device_array_view_t* edge_id_view   = nullptr;

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error),
        *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(
        rocgraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error),
        *ret_error);

    src_view = rocgraph_type_erased_device_array_view(src);
    dst_view = rocgraph_type_erased_device_array_view(dst);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                     p_handle, src_view, (rocgraph_byte_t*)h_src, ret_error),
                                 *ret_error);

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                     p_handle, dst_view, (rocgraph_byte_t*)h_dst, ret_error),
                                 *ret_error);

    if(h_wgt != nullptr)
    {
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, weight_tid, &wgt, ret_error),
                                     *ret_error);

        wgt_view = rocgraph_type_erased_device_array_view(wgt);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_view_copy_from_host(
                                         p_handle, wgt_view, (rocgraph_byte_t*)h_wgt, ret_error),
                                     *ret_error);
    }

    if(h_edge_type != nullptr)
    {
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, edge_type_tid, &edge_type, ret_error),
                                     *ret_error);
        edge_type_view = rocgraph_type_erased_device_array_view(edge_type);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, edge_type_view, (rocgraph_byte_t*)h_edge_type, ret_error),
            *ret_error);
    }

    if(h_edge_id != nullptr)
    {
        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_type_erased_device_array_create(
                                         p_handle, num_edges, edge_id_tid, &edge_id, ret_error),
                                     *ret_error);

        edge_id_view = rocgraph_type_erased_device_array_view(edge_id);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_from_host(
                p_handle, edge_id_view, (rocgraph_byte_t*)h_edge_id, ret_error),
            *ret_error);
    }

    CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_sg_graph_create(p_handle,
                                                          &properties,
                                                          src_view,
                                                          dst_view,
                                                          wgt_view,
                                                          edge_id_view,
                                                          edge_type_view,
                                                          store_transposed,
                                                          renumber,
                                                          rocgraph_bool_false,
                                                          graph,
                                                          ret_error),
                                 *ret_error);

    if(edge_id != nullptr)
    {
        rocgraph_type_erased_device_array_view_free(edge_id_view);
        rocgraph_type_erased_device_array_free(edge_id);
    }

    if(edge_type != nullptr)
    {
        rocgraph_type_erased_device_array_view_free(edge_type_view);
        rocgraph_type_erased_device_array_free(edge_type);
    }

    if(wgt != nullptr)
    {
        rocgraph_type_erased_device_array_view_free(wgt_view);
        rocgraph_type_erased_device_array_free(wgt);
    }

    rocgraph_type_erased_device_array_view_free(dst_view);
    rocgraph_type_erased_device_array_view_free(src_view);
    rocgraph_type_erased_device_array_free(dst);
    rocgraph_type_erased_device_array_free(src);
}
