// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "edge_property.hpp"
#include "graph.hpp"
#include "graph_view.hpp"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace rocgraph
{

    template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
    struct renumber_meta_t;

    template <typename vertex_t, typename edge_t, bool multi_gpu>
    struct renumber_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>>
    {
        vertex_t              number_of_vertices{};
        edge_t                number_of_edges{};
        partition_t<vertex_t> partition{};
        std::vector<vertex_t> edge_partition_segment_offsets{};
    };

    template <typename vertex_t, typename edge_t, bool multi_gpu>
    struct renumber_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>>
    {
        std::vector<vertex_t> segment_offsets{};
    };

    /**
 * @brief renumber edgelist (multi-GPU)
 *
 * This function assumes that vertices are pre-shuffled to their target processes and edges are
 * pre-shuffled to their target processess and edge partitions using compute_gpu_id_from_vertex_t
 * and compute_gpu_id_from_ext_edge_endpoints_t & compute_partition_id_from_ext_edge_endpoints_t
 * functors, respectively.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param local_vertices If valid, part of the entire set of vertices in the graph to be renumbered.
 * This parameter can be used to include isolated vertices. Applying the
 * compute_gpu_id_from_vertex_t to every vertex should return the local GPU ID for this function to
 * work (vertices should be pre-shuffled).
 * @param edgelist_srcs Pointers (one pointer per local edge partition assigned to this process) to
 * edge source vertex IDs. Source IDs are updated in-place ([INOUT] parameter). Applying the
 * compute_gpu_id_from_ext_edge_endpoints_t functor to every (destination ID, source ID) pair (if
 * store_transposed = true) or (source ID, destination ID) pair (if store_transposed = false) should
 * return the local GPU ID for this function to work (edges should be pre-shuffled). Applying the
 * compute_partition_id_from_ext_edge_endpoints_t to every (destination ID, source ID) pair (if
 * store_transposed = true) or (source ID, destination ID) pair (if store_transposed = false) should
 * also return the corresponding edge partition ID. The best way to enforce this is to use
 * shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning &
 * groupby_and_count_edgelist_by_local_partition_id.
 * @param edgelist_dsts Pointers (one pointer per local edge partition assigned to this process) to
 * edge destination vertex IDs. Destination IDs are updated in-place ([INOUT] parameter).
 * @param edgelist_edge_counts Edge counts (one count per local edge partition assigned to this
 * process).
 * @param edgelist_intra_partition_segment_offsets If valid, store segment offsets within a local
 * edge partition; a local edge partition can be further segmented by applying the
 * compute_gpu_id_from_vertex_t function to edge minor vertex IDs. This optinoal information is used
 * for further memory footprint optimization if provided.
 * @param store_transposed Should be true if renumbered edges will be used to create a graph with
 * store_transposed = true. Should be false if the edges will be used to create a graph with
 * store_transposed = false.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>
 * Tuple of labels (vertex IDs before renumbering) for the entire set of vertices (assigned to this
 * process in multi-GPU) and meta-data collected while renumbering. The meta-data includes total
 * number of vertices, total number of edges, partition_t object storing graph partitioning
 * information, vertex partition segment offsets (a vertex partition is partitioned to multiple
 * segments based on vertex degrees), and the number of local unique edge major & minor vertex IDs.
 * This meta-data is expected to be used in graph construction & graph primitives.
 */
    template <typename vertex_t, typename edge_t, bool multi_gpu>
    std::enable_if_t<
        multi_gpu,
        std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
        renumber_edgelist(raft::handle_t const&                          handle,
                          std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
                          std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
                          std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
                          std::vector<edge_t> const&    edgelist_edge_counts,
                          std::optional<std::vector<std::vector<edge_t>>> const&
                               edgelist_intra_partition_segment_offsets,
                          bool store_transposed,
                          bool do_expensive_check = false);

    /**
 * @brief renumber edgelist (single-GPU)
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices If valid, vertices in the graph to be renumbered. This parameter can be used to
 * include isolated vertices.
 * @param edgelist_srcs A pointer to edge source vertex IDs. Source IDs are updated in-place
 * ([INOUT] parameter).
 * @param edgelist_dsts A pointer to edge destination vertex IDs. Destination IDs are updated
 * in-place ([INOUT] parameter).
 * @param num_edgelist_edges Number of edges in the edgelist.
 * @param store_transposed Should be true if renumbered edges will be used to create a graph with
 * store_transposed = true. Should be false if the edges will be used to create a graph with
 * store_transposed = false.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>
 * Tuple of labels (vertex IDs before renumbering) for the entire set of vertices and meta-data
 * collected while renumbering. The meta-data includes vertex partition segment offsets (a vertex
 * partition is partitioned to multiple segments based on vertex degrees). This meta-data is
 * expected to be used in graph construction & graph primitives.
 */
    template <typename vertex_t, typename edge_t, bool multi_gpu>
    std::enable_if_t<
        !multi_gpu,
        std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
        renumber_edgelist(raft::handle_t const&                          handle,
                          std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                          vertex_t* edgelist_srcs /* [INOUT] */,
                          vertex_t* edgelist_dsts /* [INOUT] */,
                          edge_t    num_edgelist_edges,
                          bool      store_transposed,
                          bool      do_expensive_check = false);

    /**
 * @brief Renumber external vertices to internal vertices based on the provided @p
 * renumber_map_labels.
 *
 * Note rocgraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the vertices to be renumbered. The input external vertices are
 * renumbered to internal vertices in-place.
 * @param num_vertices Number of vertices to be renumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
    template <typename vertex_t, bool multi_gpu>
    void renumber_ext_vertices(raft::handle_t const& handle,
                               vertex_t*             vertices /* [INOUT] */,
                               size_t                num_vertices,
                               vertex_t const*       renumber_map_labels,
                               vertex_t              local_int_vertex_first,
                               vertex_t              local_int_vertex_last,
                               bool                  do_expensive_check = false);

    /**
 * @brief Unrenumber local internal vertices to external vertices based on the providied @p
 * renumber_map_labels.
 *
 * Note rocgraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the local internal vertices to be unrenumbered. Each input element
 * should be in [@p local_int_vertex_first, @p local_int_vertex_last). The input internal vertices
 * are renumbered to external vertices in-place.
 * @param num_vertices Number of vertices to be unrenumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
    template <typename vertex_t>
    void unrenumber_local_int_vertices(
        raft::handle_t const& handle,
        vertex_t*             vertices /* [INOUT] */,
        size_t                num_vertices,
        vertex_t const*
                 renumber_map_labels /* size = local_int_vertex_last - local_int_vertex_first */,
        vertex_t local_int_vertex_first,
        vertex_t local_int_vertex_last,
        bool     do_expensive_check = false);

    /**
 * @brief Unrenumber (possibly non-local) internal vertices to external vertices based on the
 * providied @p renumber_map_labels.
 *
 * Note rocgraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the internal vertices to be unrenumbered. The input internal vertices
 * are renumbered to external vertices in-place.
 * @param num_vertices Number of vertices to be unrenumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param vertex_partition_range_lasts Last local internal vertices (exclusive, assigned to each
 * process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
    template <typename vertex_t, bool multi_gpu>
    void unrenumber_int_vertices(raft::handle_t const&        handle,
                                 vertex_t*                    vertices /* [INOUT] */,
                                 size_t                       num_vertices,
                                 vertex_t const*              renumber_map_labels,
                                 std::vector<vertex_t> const& vertex_partition_range_lasts,
                                 bool                         do_expensive_check = false);

    /**
 * @brief Unrenumber local edges' internal source & destination IDs to external IDs based on the
 * provided @p renumber_map_labels (multi-GPU).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @params edgelist_srcs Pointers (one pointer per local edge partition assigned to this process) to
 * the local internal source vertex IDs to be unrenumbered. The input  source vertex IDs are
 * renumbered to external IDs in-place.
 * @params edgelist_dsts Pointers (one pointer per local edge partition assigned to this process) to
 * the local internal destination vertex IDs to be unrenumbered. The input destination vertex IDs
 * are renumbered to external IDs in-place.
 * @param edgelist_edge_counts Edge counts (one count per local edge partition assigned to this
 * process).
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range assigned to this process.
 * @param vertex_partition_range_lasts Last local internal vertices (exclusive, assigned to each
 * process in multi-GPU).
 * @param edgelist_intra_partition_segment_offsets If valid, store segment offsets within a local
 * edge partition; a local edge partition can be further segmented by applying the
 * compute_gpu_id_from_vertex_t function to edge minor vertex IDs. This optinoal information is used
 * for further memory footprint optimization if provided.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
    template <typename vertex_t, bool store_transposed, bool multi_gpu>
    std::enable_if_t<multi_gpu, void>
        unrenumber_local_int_edges(raft::handle_t const&         handle,
                                   std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
                                   std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
                                   std::vector<size_t> const&    edgelist_edge_counts,
                                   vertex_t const*               renumber_map_labels,
                                   std::vector<vertex_t> const&  vertex_partition_range_lasts,
                                   std::optional<std::vector<std::vector<size_t>>> const&
                                        edgelist_intra_partition_segment_offsets,
                                   bool do_expensive_check = false);

    /**
 * @brief Unrenumber local edges' internal source & destination IDs to external IDs based on the
 * provided @p renumber_map_labels (single-GPU).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @params edgelist_srcs Pointer to the local internal source vertex IDs to be unrenumbered. The
 * input source vertex IDs are renumbered to external IDs in-place.
 * @params edgelist_dsts Pointer to the local internal destination vertex IDs to be unrenumbered.
 * The input destination vertex IDs are renumbered to external IDs in-place.
 * @param num_edgelist_edges Number of edges in the edge list.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices.
 * @param num_vertices Number of vertices to be unrenumbered.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
    template <typename vertex_t, bool store_transposed, bool multi_gpu>
    std::enable_if_t<!multi_gpu, void>
        unrenumber_local_int_edges(raft::handle_t const& handle,
                                   vertex_t*             edgelist_srcs /* [INOUT] */,
                                   vertex_t*             edgelist_dsts /* [INOUT] */,
                                   size_t                num_edgelist_edges,
                                   vertex_t const*       renumber_map_labels,
                                   vertex_t              num_vertices,
                                   bool                  do_expensive_check = false);

    /**
 * @brief Renumber local external vertices to internal vertices based on the provided @p
 * renumber_map_labels.
 *
 * Note rocgraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the vertices to be renumbered. The input external vertices are
 * renumbered to internal vertices in-place.
 * @param num_vertices Number of vertices to be renumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
    template <typename vertex_t, bool multi_gpu>
    void renumber_local_ext_vertices(raft::handle_t const& handle,
                                     vertex_t*             vertices /* [INOUT] */,
                                     size_t                num_vertices,
                                     vertex_t const*       renumber_map_labels,
                                     vertex_t              local_int_vertex_first,
                                     vertex_t              local_int_vertex_last,
                                     bool                  do_expensive_check = false);
    /**
 * @brief Construct the edge list from the graph view object.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the graph to be decompressed.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param renumber_map If valid, return the renumbered edge list based on the provided @p
 * renumber_map
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of edge sources, destinations, (optional) edge weights (if
 * @p edge_weight_view.has_value() is true) and (optional) edge ids (if
 * @p edge_id_view.has_value() is true).
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              typename edge_type_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>,
               std::optional<rmm::device_uvector<edge_t>>,
               std::optional<rmm::device_uvector<edge_type_t>>>
        decompress_to_edgelist(
            raft::handle_t const&                                              handle,
            graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>>       edge_weight_view,
            std::optional<edge_property_view_t<edge_t, edge_t const*>>         edge_id_view,
            std::optional<edge_property_view_t<edge_t, edge_type_t const*>>    edge_type_view,
            std::optional<raft::device_span<vertex_t const>>                   renumber_map,
            bool do_expensive_check = false);

    /**
 * @brief Symmetrize edgelist.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs Vector of edge source vertex IDs. If multi-GPU, applying the
 * compute_gpu_id_from_ext_edge_endpoints_t to every edge should return the local GPU ID for this
 * function to work (edges should be pre-shuffled).
 * @param edgelist_dsts Vector of edge destination vertex IDs.
 * @param edgelist_weights Vector of edge weights.
 * @param reciprocal Flag indicating whether to keep (if set to `false`) or discard (if set to
 * `true`) edges that appear only in one direction.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>,
 * std::optional<rmm::device_uvector<weight_t>>> Tuple of symmetrized sources, destinations, and
 * optional weights (if @p edgelist_weights is valid).
 */
    template <typename vertex_t, typename weight_t, bool store_transposed, bool multi_gpu>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>>
        symmetrize_edgelist(raft::handle_t const&                          handle,
                            rmm::device_uvector<vertex_t>&&                edgelist_srcs,
                            rmm::device_uvector<vertex_t>&&                edgelist_dsts,
                            std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                            bool                                           reciprocal);

    /**
 * @brief Symmetrize the input graph.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to be symmetrized.
 * @param edge_weights Optional owning object holding edge weights for @p graph.
 * @param renumber_map Renumber map to recover the original vertex IDs from the renumbered vertex
 * IDs. This should be valid if multi-GPU.
 * @param reciprocal If true, an edge is kept only when the reversed edge also exists. If false,
 * keep (and symmetrize) all the edges that appear only in one direction.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Return a symmetrized graph, an owning object holding edge weights (if @p
 * edge_weights.has_value() is true) and a new renumber map (to recover the original vertex IDs, if
 * @p renumber_map.has_value() is true).
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<
        graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        std::optional<
            edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
        std::optional<rmm::device_uvector<vertex_t>>>
        symmetrize_graph(
            raft::handle_t const&                                    handle,
            graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
            std::optional<
                edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                weight_t>>&&               edge_weights,
            std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
            bool                                           reciprocal         = false,
            bool                                           do_expensive_check = false);

    /**
 * @brief Transpose the input graph.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to be transposed.
 * @param edge_weights Optional owning object holding edge weights for @p graph.
 * @param renumber_map Renumber map to recover the original vertex IDs from the renumbered vertex
 * IDs. This should be valid if multi-GPU.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Return a transposed graph, an owning object holding edge weights (if @p
 * edge_weights.has_value() is true) and a new renumber map (to recover the original vertex IDs, if
 * @p renumber_map.has_value() is true).
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<
        graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        std::optional<
            edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
        std::optional<rmm::device_uvector<vertex_t>>>
        transpose_graph(
            raft::handle_t const&                                    handle,
            graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
            std::optional<
                edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                weight_t>>&&               edge_weights,
            std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
            bool                                           do_expensive_check = false);

    /**
 * @brief Transpose the storage format (no change in an actual graph topology).
 *
 * In SG, convert between CSR and CSC. In multi-GPU, currently convert between CSR + DCSR hybrid
 * and CSC + DCSC hybrid (but the internal representation in multi-GPU is subject to change).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to transpose its storage format.
 * @param edge_weights Optional owning object holding edge weights for @p graph.
 * @param renumber_map Renumber map to recover the original vertex IDs from the renumbered vertex
 * IDs. This should be valid if multi-GPU.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<graph_t<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu>,
 * @return Return a storage transposed graph, an owning object holding edge weights (if @p
 * edge_weights.has_value() is true) and a new renumber map (to recover the original vertex IDs, if
 * @p renumber_map.has_value() is true).
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<
        graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
        std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
                                      weight_t>>,
        std::optional<rmm::device_uvector<vertex_t>>>
        transpose_graph_storage(
            raft::handle_t const&                                    handle,
            graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
            std::optional<
                edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                weight_t>>&&               edge_weights,
            std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
            bool                                           do_expensive_check = false);

    /**
 * @brief Compute the coarsened graph.
 *
 * Aggregates the vertices with the same label to a new vertex in the output coarsened graph.
 * Multi-edges in the coarsened graph are collapsed to a single edge with its weight equal to the
 * sum of multi-edge weights.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to be coarsened.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param labels Vertex labels (assigned to this process in multi-GPU) to be used in coarsening.
 * @param renumber Flag indicating whether to renumber vertices or not (must be true if @p multi_gpu
 * is true). Setting @p renumber to false is highly discouraged except for testing as this
 * negatively affects the performance and memory footprint. If @p renumber is set to true, @p labels
 * should have only non-negative integers and the number of vertices is assumed to be the maximum
 * element in @p labels (reduced over the entire set of GPUs in multi-GPU) + 1. This may produce
 * many isolated vertices if the number of unique elements (over the entire set of GPUs in
 * multi-GPU) in @p labels is much smaller than the assumed number of vertices.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of the coarsened graph, coarsened graph edge weights (if @p
 * edge_weight_view.has_value() is true) and the renumber map (if @p renumber is true).
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<
        graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        std::optional<
            edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
        std::optional<rmm::device_uvector<vertex_t>>>
        coarsen_graph(raft::handle_t const&                                              handle,
                      graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
                      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                      vertex_t const*                                              labels,
                      bool                                                         renumber,
                      bool do_expensive_check = false);

    /**
 * @brief Relabel old labels to new labels.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param old_new_label_pairs Pairs of an old label and the corresponding new label (each process
 * holds only part of the entire old labels and the corresponding new labels; partitioning can be
 * arbitrary).
 * @param num_label_pairs Number of (old, new) label pairs.
 * @param labels Labels to be relabeled. This initially holds old labels. Old labels are updated to
 * new labels in-place ([INOUT] parameter).
 * @param num_labels Number of labels to be relabeled.
 * @param skip_missing_labels Flag dictating the behavior on missing labels (@p labels contains old
 * labels missing in @p old_new_label_pairs). If set to true, missing elements are skipped (not
 * relabeled). If set to false, undefined behavior (if @p do_expensive_check is set to true, this
 * function will throw an exception).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return rmm::device_uvector<vertex_t> New labels corresponding to the @p old_labels.
 */
    template <typename vertex_t, bool multi_gpu>
    void relabel(raft::handle_t const&                        handle,
                 std::tuple<vertex_t const*, vertex_t const*> old_new_label_pairs,
                 vertex_t                                     num_label_pairs,
                 vertex_t*                                    labels /* [INOUT] */,
                 vertex_t                                     num_labels,
                 bool                                         skip_missing_labels,
                 bool                                         do_expensive_check = false);

    // FIXME: the first two elements of the returned tuple should be source & destination instead of
    // major & minor. Major & minor shouldn't be used in the non-detail public API.
    /**
 * @brief extract induced subgraph(s).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object, we extract induced subgraphs from @p graph_view.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param subgraph_offsets Span pointing to subgraph vertex offsets
 * @param subgraph_vertices Span pointing to subgraph vertices.
 *        @p subgraph_offsets and @p subgraph_vertices provide vertex sets (or local vertex sets in
 * multi-GPU) for @p subgraph_offsets.size() - 1 subgraphs to extract.  For the i'th subgraph to
 * extract, one can extract the (local-)vertex set by accessing a subset of @p subgraph_vertices,
 * where the range of the subset is [@p subgraph_offsetes[i], @p subgraph_offsets[i + 1]). In
 * multi-GPU, the vertex set for each subgraph is distributed in multiple-GPUs and each GPU holds
 *        only the vertices that are local to the GPU.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Quadraplet of edge major (destination if @p store_transposed is true, source otherwise)
 * vertices, edge minor (source if @p store_transposed  is true, destination otherwise) vertices,
 * edge weights (if @p edge_weight_view.has_value() is true), and edge offsets for each induced
 * subgraphs (size == num_subgraphs + 1). The sizes of the edge major & minor vertices are
 * edge_offsets[num_subgraphs].
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>,
               rmm::device_uvector<size_t>>
        extract_induced_subgraphs(
            raft::handle_t const&                                              handle,
            graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>>       edge_weight_view,
            raft::device_span<size_t const>                                    subgraph_offsets,
            raft::device_span<vertex_t const>                                  subgraph_vertices,
            bool do_expensive_check = false);

    // FIXME: this code should be re-factored (there should be a header file for this function including
    // implementation) to support different types (arithmetic types or thrust tuple of arithmetic types)
    // of edge properties.
    /**
 * @brief create a graph from (the optional vertex list and) the given edge list (with optional edge
 * IDs and types).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  If valid, part of the entire set of vertices in the graph to be renumbered.
 * This parameter can be used to include isolated vertices. If @p renumber is false and @p vertices
 * is valid, @p vertices elements should be consecutive integers starting from 0. If multi-GPU,
 * applying the compute_gpu_id_from_vertex_t to every vertex should return the local GPU ID for this
 * function to work (vertices should be pre-shuffled).
 * @param edgelist_srcs Vector of edge source vertex IDs. If multi-GPU, applying the
 * compute_gpu_id_from_ext_edge_endpoints_t to every edge should return the local GPU ID for this
 * function to work (edges should be pre-shuffled).
 * @param edgelist_dsts Vector of edge destination vertex IDs.
 * @param edgelist_weights Vector of weight values for edges
 * @param edgelist_edge_ids Vector of edge_id values for edges
 * @param edgelist_edge_types Vector of edge_type values for edges
 * @param graph_properties Properties of the graph represented by the input (optional vertex list
 * and) edge list.
 * @param renumber Flag indicating whether to renumber vertices or not (must be true if @p multi_gpu
 * is true).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of the generated graph and optional edge_property_t objects storing the provided
 * edge properties and a renumber map (if @p renumber is true).
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              typename edge_id_t,
              typename edge_type_t,
              bool store_transposed,
              bool multi_gpu>
    std::tuple<
        graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        std::optional<
            edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
        std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                      edge_id_t>>,
        std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                      edge_type_t>>,
        std::optional<rmm::device_uvector<vertex_t>>>
        create_graph_from_edgelist(
            raft::handle_t const&                             handle,
            std::optional<rmm::device_uvector<vertex_t>>&&    vertices,
            rmm::device_uvector<vertex_t>&&                   edgelist_srcs,
            rmm::device_uvector<vertex_t>&&                   edgelist_dsts,
            std::optional<rmm::device_uvector<weight_t>>&&    edgelist_weights,
            std::optional<rmm::device_uvector<edge_id_t>>&&   edgelist_edge_ids,
            std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
            graph_properties_t                                graph_properties,
            bool                                              renumber,
            bool                                              do_expensive_check = false);

    /**
 * @brief      Find all 2-hop neighbors in the graph
 *
 * Find pairs of vertices in the input graph such that each pair is connected by
 * a path that is two hops in length.
 *
 * @throws     rocgraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param  graph The input graph object
 * @param  start_vertices Optional list of starting vertices to discover two-hop neighbors of
 * @return tuple containing pairs of vertices that are 2-hops apart.
 */
    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> get_two_hop_neighbors(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        std::optional<raft::device_span<vertex_t const>>                   start_vertices);

    /**
 * @brief Compute per-vertex incoming edge weight sums.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to compute per-vertex incoming edge weight
 * sums.
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @return Incoming edge weight sums for each vertex.
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    rmm::device_uvector<weight_t> compute_in_weight_sums(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        edge_property_view_t<edge_t, weight_t const*>                      edge_weight_view);

    /**
 * @brief Compute per-vertex outgoing edge weight sums.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to compute per-vertex outgoing edge weight
 * sums.
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @return Outgoing edge weight sums for each vertex.
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    rmm::device_uvector<weight_t> compute_out_weight_sums(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        edge_property_view_t<edge_t, weight_t const*>                      edge_weight_view);

    /**
 * @brief Compute maximum per-vertex incoming edge weight sums.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to compute the maximum per-vertex incoming
 * edge weight sums.
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @return Maximum per-vertex incoming edge weight sums.
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    weight_t compute_max_in_weight_sum(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        edge_property_view_t<edge_t, weight_t const*>                      edge_weight_view);

    /**
 * @brief Compute maximum per-vertex outgoing edge weight sums.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to compute the maximum per-vertex outgoing
 * edge weight sums.
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @return Maximum per-vertex outgoing edge weight sums.
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    weight_t compute_max_out_weight_sum(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        edge_property_view_t<edge_t, weight_t const*>                      edge_weight_view);

    /**
 * @brief Sum the weights of the entire set of edges.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to sum the edge weights.
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @return Sum of the weights of the entire set of edges.
 */
    template <typename vertex_t,
              typename edge_t,
              typename weight_t,
              bool store_transposed,
              bool multi_gpu>
    weight_t compute_total_edge_weight(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        edge_property_view_t<edge_t, weight_t const*>                      edge_weight_view);

    /**
 * @brief Select random vertices
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to select random vertices from.
 * @param given_set Distributed set to sample from. If @p given_set is not specified, sample from
 *  the entire vertex range provided by @p graph_view.
 * @param  rng_state The RngState instance holding pseudo-random number generator state.
 * @param  select_count The number of vertices to select from the graph
 * @param  with_replacement If true, select with replacement, if false select without replacement
 * @param  sort_vertices If true, return the sorted vertices (in the ascending order).
 * @return Device vector of selected vertices.
 */
    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    rmm::device_uvector<vertex_t> select_random_vertices(
        raft::handle_t const&                                              handle,
        graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
        std::optional<raft::device_span<vertex_t const>>                   given_set,
        raft::random::RngState&                                            rng_state,
        size_t                                                             select_count,
        bool                                                               with_replacement,
        bool                                                               sort_vertices,
        bool do_expensive_check = false);

    /**
 * @brief renumber sampling output
 *
 * @deprecated This API will be deprecated and will be replaced by the
 * renumber_and_compress_sampled_edgelist and renumber_and_sort_sampled_edgelist functions in
 * sampling_functions.hpp.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) outputs satisfying the
 * following requirements.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, flag=src, hop) triplets for each
 * vertex ID in @p edgelist_srcs and (vertex ID, flag=dst, hop) triplets for each vertex ID in @p
 * edgelist_dsts. From these triplets, we can find the minimum (hop, flag) pairs for every unique
 * vertex ID (hop is the primary key and flag is the secondary key, flag=src is considered smaller
 * than flag=dst if hop numbers are same). Vertex IDs with smaller (hop, flag) pairs precede vertex
 * IDs with larger (hop, flag) pairs in renumbering. Ordering can be arbitrary among the vertices
 * with the same (hop, flag) pairs.
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in @p edgelist_srcs precede vertex IDs that
 * appear only in @p edgelist_dsts.
 * 3. If label_offsets.has_value() is ture, edge lists for different labels will be renumbered
 * separately.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam label_t Type of labels. Needs to be an integral type.
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing original edgelist source vertices.
 * @param edgelist_dsts A vector storing original edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_hops An optional pointer to the array storing hops for each edge list (source,
 * destination) pairs (size = @p edgelist_srcs.size() if valid).
 * @param label_offsets An optional tuple of unique labels and the input edge list (@p
 * edgelist_srcs, @p edgelist_hops, and @p edgelist_dsts) offsets for the labels (siez = # unique
 * labels + 1).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing renumbered edge sources (size = @p edgelist_srcs.size()) ,
 * renumbered edge destinations (size = @p edgelist_dsts.size()), renumber_map to query original
 * verties (size = # unique vertices or aggregate # unique vertices for every label), and
 * renumber_map offsets (size = std::get<0>(*label_offsets).size() + 1, valid only if @p
 * label_offsets.has_value() is true).
 */
    template <typename vertex_t, typename label_t>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<size_t>>>
        renumber_sampled_edgelist(
            raft::handle_t const&                                      handle,
            rmm::device_uvector<vertex_t>&&                            edgelist_srcs,
            rmm::device_uvector<vertex_t>&&                            edgelist_dsts,
            std::optional<raft::device_span<int32_t const>>            edgelist_hops,
            std::optional<std::tuple<raft::device_span<label_t const>,
                                     raft::device_span<size_t const>>> label_offsets,
            bool                                                       do_expensive_check = false);

    /**
 * @brief Remove self loops from an edge list
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t      Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t    Type of edge weight. Currently float and double are supported.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs  List of source vertex ids
 * @param edgelist_dsts  List of destination vertex ids
 * @param edgelist_weights  Optional list of edge weights
 * @param edgelist_edge_ids  Optional list of edge ids
 * @param edgelist_edge_types  Optional list of edge types
 * @return Tuple of vectors storing edge sources, destinations, optional weights,
 *    optional edge ids, optional edge types.
 */
    template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>,
               std::optional<rmm::device_uvector<edge_t>>,
               std::optional<rmm::device_uvector<edge_type_t>>>
        remove_self_loops(raft::handle_t const&                             handle,
                          rmm::device_uvector<vertex_t>&&                   edgelist_srcs,
                          rmm::device_uvector<vertex_t>&&                   edgelist_dsts,
                          std::optional<rmm::device_uvector<weight_t>>&&    edgelist_weights,
                          std::optional<rmm::device_uvector<edge_t>>&&      edgelist_edge_ids,
                          std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types);

    /**
 * @brief Remove all but one edge when a multi-edge exists.
 *
 * When a multi-edge exists, one of the edges will remain. If @p keep_min_value_edge is false, an
 * arbitrary edge will be selected among the edges in the multi-edge. If @p keep_min_value_edge is
 * true, the edge with the minimum value will be selected. The edge weights will be first compared
 * (if @p edgelist_weights.has_value() is true); edge IDs will be compared next (if @p
 * edgelist_edge_ids.has_value() is true); and edge types (if @p edgelist_edge_types.has_value() is
 * true) will compared last.
 *
 * In an MG context it is assumed that edges have been shuffled to the proper GPU,
 * in which case any multi-edges will be on the same GPU.
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t      Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t    Type of edge weight. Currently float and double are supported.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs  List of source vertex ids
 * @param edgelist_dsts  List of destination vertex ids
 * @param edgelist_weights  Optional list of edge weights
 * @param edgelist_edge_ids  Optional list of edge ids
 * @param edgelist_edge_types  Optional list of edge types
 * @param keep_min_value_edge Flag indicating whether to keep an arbitrary edge (false) or the
 * minimum value edge (true) among the edges in a multi-edge. Relevant only if @p
 * edgelist_weights.has_value() | @p edgelist_edge_ids.has_value() | @p
 * edgelist_edge_types.has_value() is true. Setting this to true incurs performance overhead as this
 * requires more comparisons.
 * @return Tuple of vectors storing edge sources, destinations, optional weights,
 *    optional edge ids, optional edge types.
 */
    template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>,
               std::optional<rmm::device_uvector<edge_t>>,
               std::optional<rmm::device_uvector<edge_type_t>>>
        remove_multi_edges(raft::handle_t const&                             handle,
                           rmm::device_uvector<vertex_t>&&                   edgelist_srcs,
                           rmm::device_uvector<vertex_t>&&                   edgelist_dsts,
                           std::optional<rmm::device_uvector<weight_t>>&&    edgelist_weights,
                           std::optional<rmm::device_uvector<edge_t>>&&      edgelist_edge_ids,
                           std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                           bool keep_min_value_edge = false);

    /**
 * @brief Shuffle external vertex ids to the proper GPU.
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  List of vertex ids
 * @return Vector of vertex ids mapped to this GPU.
 */
    template <typename vertex_t>
    rmm::device_uvector<vertex_t>
        shuffle_external_vertices(raft::handle_t const&           handle,
                                  rmm::device_uvector<vertex_t>&& vertices);

    /**
 * @brief Shuffle external vertex ids and values to the proper GPU.
 *
 * @tparam vertex_t   Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t    Type of values. currently supported types are int32_t,
 * int64_t, size_t, float and double.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  List of vertex ids
 * @param values List of values
 * @return Tuple of vectors storing vertex ids and values mapped to this GPU.
 */
    template <typename vertex_t, typename value_t>
    std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
        shuffle_external_vertex_value_pairs(raft::handle_t const&           handle,
                                            rmm::device_uvector<vertex_t>&& vertices,
                                            rmm::device_uvector<value_t>&&  values);

    /**
 * @brief Shuffle external edges to the proper GPU.
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t      Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t    Type of edge weight. Currently float and double are supported.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  List of source vertex ids
 * @param edge_dsts  List of destination vertex ids
 * @param edge_weights  Optional list of edge weights
 * @param edge_ids  Optional list of edge ids
 * @param edge_types Optional list of edge types
 * @return Tuple of vectors storing edge sources, destinations, optional weights,
 *          optional edge ids, optional edge types mapped to this GPU.
 */
    template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
    std::tuple<rmm::device_uvector<vertex_t>,
               rmm::device_uvector<vertex_t>,
               std::optional<rmm::device_uvector<weight_t>>,
               std::optional<rmm::device_uvector<edge_t>>,
               std::optional<rmm::device_uvector<edge_type_t>>>
        shuffle_external_edges(raft::handle_t const&                             handle,
                               rmm::device_uvector<vertex_t>&&                   edge_srcs,
                               rmm::device_uvector<vertex_t>&&                   edge_dsts,
                               std::optional<rmm::device_uvector<weight_t>>&&    edge_weights,
                               std::optional<rmm::device_uvector<edge_t>>&&      edge_ids,
                               std::optional<rmm::device_uvector<edge_type_t>>&& edge_types);

} // namespace rocgraph
