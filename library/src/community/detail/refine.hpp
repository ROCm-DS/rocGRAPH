// Copyright (C) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "dendrogram.hpp"
#include "edge_src_dst_property.hpp"
#include "graph.hpp"

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{
    namespace detail
    {

        template <typename graph_view_t, typename weight_t>
        std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
                   std::pair<rmm::device_uvector<typename graph_view_t::vertex_type>,
                             rmm::device_uvector<typename graph_view_t::vertex_type>>>
            refine_clustering(
                raft::handle_t const&                                     handle,
                raft::random::RngState&                                   rng_state,
                graph_view_t const&                                       graph_view,
                std::optional<edge_property_view_t<typename graph_view_t::edge_type,
                                                   weight_t const*>>      edge_weight_view,
                weight_t                                                  total_edge_weight,
                weight_t                                                  resolution,
                weight_t                                                  theta,
                rmm::device_uvector<weight_t> const&                      vertex_weights_v,
                rmm::device_uvector<typename graph_view_t::vertex_type>&& cluster_keys_v,
                rmm::device_uvector<weight_t>&&                           cluster_weights_v,
                rmm::device_uvector<typename graph_view_t::vertex_type>&& next_clusters_v,
                edge_src_property_t<graph_view_t, weight_t> const&        src_vertex_weights_cache,
                edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const&
                    src_clusters_cache,
                edge_dst_property_t<graph_view_t, typename graph_view_t::vertex_type> const&
                     dst_clusters_cache,
                bool up_down);

    }
} // namespace rocgraph
