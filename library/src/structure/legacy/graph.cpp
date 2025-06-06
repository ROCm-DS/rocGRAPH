// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "utilities/graph_utils.cuh"

#include "legacy/graph.hpp"
#include "utilities/error.hpp"

#include <raft/util/device_atomics.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace
{

    template <typename vertex_t, typename edge_t>
    void degree_from_offsets(vertex_t              number_of_vertices,
                             edge_t const*         offsets,
                             edge_t*               degree,
                             rmm::cuda_stream_view stream_view)
    {
        // Computes out-degree for x = 0 and x = 2
        thrust::for_each(
            rmm::exec_policy(stream_view),
            thrust::make_counting_iterator<vertex_t>(0),
            thrust::make_counting_iterator<vertex_t>(number_of_vertices),
            [offsets, degree] __device__(vertex_t v) { degree[v] = offsets[v + 1] - offsets[v]; });
    }

    template <typename vertex_t, typename edge_t>
    void degree_from_vertex_ids(const raft::handle_t* handle,
                                vertex_t              number_of_vertices,
                                edge_t                number_of_edges,
                                vertex_t const*       indices,
                                edge_t*               degree,
                                rmm::cuda_stream_view stream_view)
    {
        thrust::for_each(
            rmm::exec_policy(stream_view),
            thrust::make_counting_iterator<edge_t>(0),
            thrust::make_counting_iterator<edge_t>(number_of_edges),
            [indices, degree] __device__(edge_t e) { atomicAdd(degree + indices[e], edge_t{1}); });
        if((handle != nullptr) && (handle->comms_initialized()))
        {
            auto& comm = handle->get_comms();
            comm.allreduce(
                degree, degree, number_of_vertices, raft::comms::op_t::SUM, stream_view.value());
        }
    }

} // namespace

namespace rocgraph
{
    namespace legacy
    {

        template <typename VT, typename ET, typename WT>
        void GraphViewBase<VT, ET, WT>::get_vertex_identifiers(VT* identifiers) const
        {
            rocgraph::detail::sequence<VT>(number_of_vertices, identifiers);
        }

        template <typename VT, typename ET, typename WT>
        void GraphCompressedSparseBaseView<VT, ET, WT>::get_source_indices(VT* src_indices) const
        {
            ROCGRAPH_EXPECTS(offsets != nullptr, "No graph specified");
            rocgraph::detail::offsets_to_indices<ET, VT>(
                offsets, GraphViewBase<VT, ET, WT>::number_of_vertices, src_indices);
        }

        template <typename VT, typename ET, typename WT>
        void GraphCOOView<VT, ET, WT>::degree(ET* degree, DegreeDirection direction) const
        {
            //
            // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
            //        in then x should be modified to reflect the expected direction.
            //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
            //        the offsets/indices and request an out-degree (x=2))
            //
            hipStream_t stream{nullptr};

            if(direction != DegreeDirection::IN)
            {
                if((GraphViewBase<VT, ET, WT>::handle != nullptr)
                   && (GraphViewBase<VT, ET, WT>::handle
                           ->comms_initialized())) // FIXME retrieve global source
                // indexing for the allreduce work
                {
                    ROCGRAPH_FAIL("MG degree not implemented for OUT degree");
                }
                degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle,
                                       GraphViewBase<VT, ET, WT>::number_of_vertices,
                                       GraphViewBase<VT, ET, WT>::number_of_edges,
                                       src_indices,
                                       degree,
                                       stream);
            }

            if(direction != DegreeDirection::OUT)
            {
                degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle,
                                       GraphViewBase<VT, ET, WT>::number_of_vertices,
                                       GraphViewBase<VT, ET, WT>::number_of_edges,
                                       dst_indices,
                                       degree,
                                       stream);
            }
        }

        template <typename VT, typename ET, typename WT>
        void GraphCompressedSparseBaseView<VT, ET, WT>::degree(ET*             degree,
                                                               DegreeDirection direction) const
        {
            //
            // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
            //        in then x should be modified to reflect the expected direction.
            //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
            //        the offsets/indices and request an out-degree (x=2))
            //
            rmm::cuda_stream_view stream_view;

            if(direction != DegreeDirection::IN)
            {
                if((GraphViewBase<VT, ET, WT>::handle != nullptr)
                   && (GraphViewBase<VT, ET, WT>::handle->comms_initialized()))
                {
                    ROCGRAPH_FAIL(
                        "MG degree not implemented for OUT degree"); // FIXME retrieve global
                    // source indexing for
                    // the allreduce to work
                }
                degree_from_offsets(
                    GraphViewBase<VT, ET, WT>::number_of_vertices, offsets, degree, stream_view);
            }

            if(direction != DegreeDirection::OUT)
            {
                degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle,
                                       GraphViewBase<VT, ET, WT>::number_of_vertices,
                                       GraphViewBase<VT, ET, WT>::number_of_edges,
                                       indices,
                                       degree,
                                       stream_view);
            }
        }

    } // namespace legacy
} // namespace rocgraph

#include "utilities/eidir_graph_utils.hpp"

#include "eidir_graph.hpp"
