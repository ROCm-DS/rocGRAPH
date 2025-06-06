// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "prims/detail/extract_transform_v_frontier_e.cuh"
#include "prims/property_op_utils.cuh"

#include "edge_src_dst_property.hpp"
#include "graph_view.hpp"
#include "utilities/dataframe_buffer.hpp"
#include "utilities/error.hpp"

#include <raft/core/handle.hpp>

#include <cstdint>
#include <numeric>

namespace rocgraph
{

    /**
 * @brief Iterate over outgoing_edges from the current vertex frontier and extract the valid edge
 * functor outputs.
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
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
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
 * property values for the destination, and property values for the edge and returns thrust::nullopt
 * (if the return value is to be discarded) or a valid @p e_op output to be extracted and
 * accumulated.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Dataframe buffer object storing extracted and accumulated valid @p e_op return values.
 */

    template <typename GraphViewType,
              typename VertexFrontierBucketType,
              typename EdgeSrcValueInputWrapper,
              typename EdgeDstValueInputWrapper,
              typename EdgeValueInputWrapper,
              typename EdgeOp>
    decltype(allocate_dataframe_buffer<
             typename detail::edge_op_result_type<typename VertexFrontierBucketType::key_type,
                                                  typename GraphViewType::vertex_type,
                                                  typename EdgeSrcValueInputWrapper::value_type,
                                                  typename EdgeDstValueInputWrapper::value_type,
                                                  typename EdgeValueInputWrapper::value_type,
                                                  EdgeOp>::type::value_type>(
        std::size_t{0}, rmm::cuda_stream_view{}))
        extract_transform_v_frontier_outgoing_e(raft::handle_t const&           handle,
                                                GraphViewType const&            graph_view,
                                                VertexFrontierBucketType const& frontier,
                                                EdgeSrcValueInputWrapper edge_src_value_input,
                                                EdgeDstValueInputWrapper edge_dst_value_input,
                                                EdgeValueInputWrapper    edge_value_input,
                                                EdgeOp                   e_op,
                                                bool                     do_expensive_check = false)
    {
        static_assert(!GraphViewType::is_storage_transposed);

        using e_op_result_t =
            typename detail::edge_op_result_type<typename VertexFrontierBucketType::key_type,
                                                 typename GraphViewType::vertex_type,
                                                 typename EdgeSrcValueInputWrapper::value_type,
                                                 typename EdgeDstValueInputWrapper::value_type,
                                                 typename EdgeValueInputWrapper::value_type,
                                                 EdgeOp>::type;
        static_assert(!std::is_same_v<e_op_result_t, void>);
        using payload_t = typename e_op_result_t::value_type;

        auto value_buffer = allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
        std::tie(std::ignore, value_buffer)
            = detail::extract_transform_v_frontier_e<false, void, payload_t>(handle,
                                                                             graph_view,
                                                                             frontier,
                                                                             edge_src_value_input,
                                                                             edge_dst_value_input,
                                                                             edge_value_input,
                                                                             e_op,
                                                                             do_expensive_check);

        return value_buffer;
    }

} // namespace rocgraph
