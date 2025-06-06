// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "prims/property_op_utils.cuh"
#include "prims/transform_reduce_e.cuh"

#include "graph_view.hpp"

#include <raft/core/handle.hpp>

#include <cstdint>

namespace rocgraph
{

    /**
 * @brief Count the number of edges that satisfies the given predicate.
 *
 * This function is inspired by thrust::count_if().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either rocgraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or rocgraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to
 * fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * rocgraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * rocgraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either rocgraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or rocgraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns true if this edge should be included in the returned count.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return GraphViewType::edge_type Number of times @p e_op returned true.
 */
    template <typename GraphViewType,
              typename EdgeSrcValueInputWrapper,
              typename EdgeDstValueInputWrapper,
              typename EdgeValueInputWrapper,
              typename EdgeOp>
    typename GraphViewType::edge_type count_if_e(raft::handle_t const&    handle,
                                                 GraphViewType const&     graph_view,
                                                 EdgeSrcValueInputWrapper edge_src_value_input,
                                                 EdgeDstValueInputWrapper edge_dst_value_input,
                                                 EdgeValueInputWrapper    edge_value_input,
                                                 EdgeOp                   e_op,
                                                 bool do_expensive_check = false)
    {
        using vertex_t = typename GraphViewType::vertex_type;
        using edge_t   = typename GraphViewType::edge_type;

        if(do_expensive_check)
        {
            // currently, nothing to do
        }

        return transform_reduce_e(handle,
                                  graph_view,
                                  edge_src_value_input,
                                  edge_dst_value_input,
                                  edge_value_input,
                                  cast_edge_op_bool_to_integer<GraphViewType,
                                                               vertex_t,
                                                               EdgeSrcValueInputWrapper,
                                                               EdgeDstValueInputWrapper,
                                                               EdgeValueInputWrapper,
                                                               EdgeOp,
                                                               edge_t>{e_op},
                                  edge_t{0});
    }

} // namespace rocgraph
