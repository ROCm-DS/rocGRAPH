// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"

#include "edge_src_dst_property.hpp"
#include "graph.hpp"
#include "graph_view.hpp"
#include "utilities/thrust_tuple_utils.hpp"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/optional.h>
#include <thrust/tuple.h>

namespace rocgraph
{
    namespace detail
    {

        struct return_edges_with_properties_e_op
        {
            template <typename key_t, typename vertex_t, typename EdgeProperties>
            auto __host__ __device__ operator()(key_t    optionally_tagged_src,
                                                vertex_t dst,
                                                thrust::nullopt_t,
                                                thrust::nullopt_t,
                                                EdgeProperties edge_properties) const
            {
                static_assert(std::is_same_v<key_t, vertex_t>
                              || std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>);

                // FIXME: A solution using thrust_tuple_cat would be more flexible here
                if constexpr(std::is_same_v<key_t, vertex_t>)
                {
                    vertex_t src{optionally_tagged_src};

                    if constexpr(std::is_same_v<EdgeProperties, thrust::nullopt_t>)
                    {
                        return thrust::make_optional(thrust::make_tuple(src, dst));
                    }
                    else if constexpr(std::is_arithmetic<EdgeProperties>::value)
                    {
                        return thrust::make_optional(thrust::make_tuple(src, dst, edge_properties));
                    }
                    else if constexpr(rocgraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value
                                      && (thrust::tuple_size<EdgeProperties>::value == 2))
                    {
                        return thrust::make_optional(
                            thrust::make_tuple(src,
                                               dst,
                                               thrust::get<0>(edge_properties),
                                               thrust::get<1>(edge_properties)));
                    }
                    else if constexpr(rocgraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value
                                      && (thrust::tuple_size<EdgeProperties>::value == 3))
                    {
                        return thrust::make_optional(
                            thrust::make_tuple(src,
                                               dst,
                                               thrust::get<0>(edge_properties),
                                               thrust::get<1>(edge_properties),
                                               thrust::get<2>(edge_properties)));
                    }
                }
                else if constexpr(std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>)
                {
                    vertex_t src{thrust::get<0>(optionally_tagged_src)};
                    int32_t  label{thrust::get<1>(optionally_tagged_src)};

                    src = thrust::get<0>(optionally_tagged_src);
                    if constexpr(std::is_same_v<EdgeProperties, thrust::nullopt_t>)
                    {
                        return thrust::make_optional(thrust::make_tuple(src, dst, label));
                    }
                    else if constexpr(std::is_arithmetic<EdgeProperties>::value)
                    {
                        return thrust::make_optional(
                            thrust::make_tuple(src, dst, edge_properties, label));
                    }
                    else if constexpr(rocgraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value
                                      && (thrust::tuple_size<EdgeProperties>::value == 2))
                    {
                        return thrust::make_optional(
                            thrust::make_tuple(src,
                                               dst,
                                               thrust::get<0>(edge_properties),
                                               thrust::get<1>(edge_properties),
                                               label));
                    }
                    else if constexpr(rocgraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value
                                      && (thrust::tuple_size<EdgeProperties>::value == 3))
                    {
                        return thrust::make_optional(
                            thrust::make_tuple(src,
                                               dst,
                                               thrust::get<0>(edge_properties),
                                               thrust::get<1>(edge_properties),
                                               thrust::get<2>(edge_properties),
                                               label));
                    }
                }
            }
        };

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_t,
                  typename label_t,
                  typename tag_t,
                  bool multi_gpu>
        std::tuple<rmm::device_uvector<vertex_t>,
                   rmm::device_uvector<vertex_t>,
                   std::optional<rmm::device_uvector<weight_t>>,
                   std::optional<rmm::device_uvector<edge_t>>,
                   std::optional<rmm::device_uvector<edge_type_t>>,
                   std::optional<rmm::device_uvector<label_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                           handle,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&         graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>>    edge_weight_view,
                std::optional<edge_property_view_t<edge_t, edge_t const*>>      edge_id_view,
                std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
                rocgraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> const&
                     vertex_frontier,
                bool do_expensive_check)
        {
            rmm::device_uvector<vertex_t>                   majors(0, handle.get_stream());
            rmm::device_uvector<vertex_t>                   minors(0, handle.get_stream());
            std::optional<rmm::device_uvector<edge_t>>      edge_ids{std::nullopt};
            std::optional<rmm::device_uvector<weight_t>>    edge_weights{std::nullopt};
            std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
            std::optional<rmm::device_uvector<label_t>>     labels{std::nullopt};

            if(edge_weight_view)
            {
                if(edge_id_view)
                {
                    if(edge_type_view)
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_weights, edge_ids, edge_types, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_weight_view, *edge_id_view, *edge_type_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_weights, edge_ids, edge_types)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_weight_view, *edge_id_view, *edge_type_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                    else
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_weights, edge_ids, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_weight_view, *edge_id_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_weights, edge_ids)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_weight_view, *edge_id_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                }
                else
                {
                    if(edge_type_view)
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_weights, edge_types, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_weight_view, *edge_type_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_weights, edge_types)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_weight_view, *edge_type_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                    else
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_weights, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    *edge_weight_view,
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_weights)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    *edge_weight_view,
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                }
            }
            else
            {
                if(edge_id_view)
                {
                    if(edge_type_view)
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_ids, edge_types, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_id_view, *edge_type_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_ids, edge_types)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    view_concat(*edge_id_view, *edge_type_view),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                    else
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_ids, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    *edge_id_view,
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_ids)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    *edge_id_view,
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                }
                else
                {
                    if(edge_type_view)
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, edge_types, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    *edge_type_view,
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors, edge_types)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    *edge_type_view,
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                    else
                    {
                        if constexpr(std::is_same_v<tag_t, int32_t>)
                        {
                            std::tie(majors, minors, labels)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    edge_dummy_property_t{}.view(),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                        else
                        {
                            std::tie(majors, minors)
                                = rocgraph::extract_transform_v_frontier_outgoing_e(
                                    handle,
                                    graph_view,
                                    vertex_frontier.bucket(0),
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    edge_dummy_property_t{}.view(),
                                    return_edges_with_properties_e_op{},
                                    do_expensive_check);
                        }
                    }
                }
            }

            return std::make_tuple(std::move(majors),
                                   std::move(minors),
                                   std::move(edge_weights),
                                   std::move(edge_ids),
                                   std::move(edge_types),
                                   std::move(labels));
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_t,
                  typename label_t,
                  bool multi_gpu>
        std::tuple<rmm::device_uvector<vertex_t>,
                   rmm::device_uvector<vertex_t>,
                   std::optional<rmm::device_uvector<weight_t>>,
                   std::optional<rmm::device_uvector<edge_t>>,
                   std::optional<rmm::device_uvector<edge_type_t>>,
                   std::optional<rmm::device_uvector<label_t>>>
            gather_one_hop_edgelist(
                raft::handle_t const&                                           handle,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&         graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>>    edge_weight_view,
                std::optional<edge_property_view_t<edge_t, edge_t const*>>      edge_id_view,
                std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
                raft::device_span<vertex_t const>                               active_majors,
                std::optional<raft::device_span<label_t const>>                 active_major_labels,
                bool                                                            do_expensive_check)
        {
            if(active_major_labels)
            {
                rocgraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false>
                    vertex_label_frontier(handle, 1);
                vertex_label_frontier.bucket(0).insert(
                    thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
                    thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

                return gather_one_hop_edgelist<vertex_t,
                                               edge_t,
                                               weight_t,
                                               edge_type_t,
                                               label_t,
                                               label_t,
                                               multi_gpu>(handle,
                                                          graph_view,
                                                          edge_weight_view,
                                                          edge_id_view,
                                                          edge_type_view,
                                                          vertex_label_frontier,
                                                          do_expensive_check);
            }
            else
            {
                rocgraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_frontier(
                    handle, 1);
                vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

                return gather_one_hop_edgelist<vertex_t,
                                               edge_t,
                                               weight_t,
                                               edge_type_t,
                                               label_t,
                                               void,
                                               multi_gpu>(handle,
                                                          graph_view,
                                                          edge_weight_view,
                                                          edge_id_view,
                                                          edge_type_view,
                                                          vertex_frontier,
                                                          do_expensive_check);
            }
        }

    } // namespace detail
} // namespace rocgraph
