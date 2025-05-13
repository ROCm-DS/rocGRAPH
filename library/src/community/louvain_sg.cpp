// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "community/louvain_impl.cuh"

namespace rocgraph
{

    // Explicit template instantations

    template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int32_t, false, false> const&,
                std::optional<edge_property_view_t<int32_t, float const*>>,
                size_t,
                float,
                float);
    template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, float const*>>,
                size_t,
                float,
                float);
    template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int64_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, float const*>>,
                size_t,
                float,
                float);
    template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int32_t, false, false> const&,
                std::optional<edge_property_view_t<int32_t, double const*>>,
                size_t,
                double,
                double);
    template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, double const*>>,
                size_t,
                double,
                double);
    template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int64_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, double const*>>,
                size_t,
                double,
                double);

    template std::pair<size_t, float>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int32_t, false, false> const&,
                std::optional<edge_property_view_t<int32_t, float const*>>,
                int32_t*,
                size_t,
                float,
                float);
    template std::pair<size_t, double>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int32_t, false, false> const&,
                std::optional<edge_property_view_t<int32_t, double const*>>,
                int32_t*,
                size_t,
                double,
                double);
    template std::pair<size_t, float>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, float const*>>,
                int32_t*,
                size_t,
                float,
                float);
    template std::pair<size_t, double>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int32_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, double const*>>,
                int32_t*,
                size_t,
                double,
                double);
    template std::pair<size_t, float>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int64_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, float const*>>,
                int64_t*,
                size_t,
                float,
                float);
    template std::pair<size_t, double>
        louvain(raft::handle_t const&,
                std::optional<std::reference_wrapper<raft::random::RngState>>,
                graph_view_t<int64_t, int64_t, false, false> const&,
                std::optional<edge_property_view_t<int64_t, double const*>>,
                int64_t*,
                size_t,
                double,
                double);

} // namespace rocgraph
