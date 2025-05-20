// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "dendrogram.hpp"
#include "edge_property.hpp"
#include "edge_src_dst_property.hpp"
#include "graph.hpp"
#include "graph_view.hpp"
#ifdef TIMING
#include "utilities/high_res_timer.hpp"
#endif

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{
    namespace detail
    {

#ifdef TIMING
        // Some timing functions
        template <bool multi_gpu>
        void timer_start(raft::handle_t const& handle,
                         HighResTimer&         hr_timer,
                         std::string const&    region)
        {
            if constexpr(multi_gpu)
            {
                if(handle.get_comms().get_rank() == 0)
                    hr_timer.start(region);
            }
            else
            {
                hr_timer.start(region);
            }
        }

        template <bool multi_gpu>
        void timer_stop(raft::handle_t const& handle, HighResTimer& hr_timer)
        {
            if constexpr(multi_gpu)
            {
                if(handle.get_comms().get_rank() == 0)
                {
                    handle.get_stream().synchronize();
                    hr_timer.stop();
                }
            }
            else
            {
                handle.get_stream().synchronize();
                hr_timer.stop();
            }
        }

        template <bool multi_gpu>
        void timer_display_and_clear(raft::handle_t const& handle,
                                     HighResTimer const&   hr_timer,
                                     std::ostream&         os)
        {
            if(multi_gpu)
            {
                if(handle.get_comms().get_rank() == 0)
                    hr_timer.display_and_clear(os);
            }
            else
            {
                hr_timer.display_and_clear(os);
            }
        }
#endif

        template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
        weight_t compute_modularity(
            raft::handle_t const&                                        handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const&      graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, vertex_t> const&
                src_clusters_cache,
            edge_dst_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, vertex_t> const&
                                                 dst_clusters_cache,
            rmm::device_uvector<vertex_t> const& next_clusters,
            rmm::device_uvector<weight_t> const& cluster_weights,
            weight_t                             total_edge_weight,
            weight_t                             resolution);

        template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
        std::tuple<graph_t<vertex_t, edge_t, false, multi_gpu>,
                   std::optional<
                       edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>>>
            graph_contraction(
                raft::handle_t const&                                        handle,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&      graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weights,
                raft::device_span<vertex_t>                                  labels);

        template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
        rmm::device_uvector<vertex_t> update_clustering_by_delta_modularity(
            raft::handle_t const&                                        handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const&      graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            weight_t                                                     total_edge_weight,
            weight_t                                                     resolution,
            rmm::device_uvector<weight_t> const&                         vertex_weights_v,
            rmm::device_uvector<vertex_t>&&                              cluster_keys_v,
            rmm::device_uvector<weight_t>&&                              cluster_weights_v,
            rmm::device_uvector<vertex_t>&&                              next_clusters_v,
            edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t> const&
                src_vertex_weights_cache,
            edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, vertex_t> const&
                src_clusters_cache,
            edge_dst_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, vertex_t> const&
                 dst_clusters_cache,
            bool up_down);

        template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
        std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
            compute_cluster_keys_and_values(
                raft::handle_t const&                                        handle,
                graph_view_t<vertex_t, edge_t, false, multi_gpu> const&      graph_view,
                std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                rmm::device_uvector<vertex_t> const&                         next_clusters_v,
                edge_src_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                    vertex_t> const&                         src_clusters_cache);

    } // namespace detail
} // namespace rocgraph
