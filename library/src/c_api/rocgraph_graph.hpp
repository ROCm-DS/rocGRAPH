// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "c_api/rocgraph_array.hpp"
#include "c_api/rocgraph_error.hpp"

#include "internal/aux/rocgraph_graph_aux.h"

#include "graph.hpp"
#include "graph_functions.hpp"

#include <memory>

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_graph_t
        {
            rocgraph_data_type_id vertex_type_;
            rocgraph_data_type_id edge_type_;
            rocgraph_data_type_id weight_type_;
            rocgraph_data_type_id edge_type_id_type_;
            bool                  store_transposed_;
            bool                  multi_gpu_;

            void* graph_; // graph_t<...>*
            void* number_map_; // rmm::device_uvector<vertex_t>*
            void* edge_weights_; // edge_property_t<
                //    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                //    weight_t>*
            void* edge_ids_; // edge_property_t<
                //    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                //    edge_t>*
            void* edge_types_; // edge_property_t<
                //    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                //    edge_type_id_t>*
        };

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  bool store_transposed,
                  bool multi_gpu>
        rocgraph_status transpose_storage(raft::handle_t const& handle,
                                          rocgraph_graph_t*     graph,
                                          rocgraph_error_t*     error)
        {
            if(store_transposed == graph->store_transposed_)
            {
                if((graph->edge_ids_ != nullptr) || (graph->edge_types_ != nullptr))
                {
                    error->error_message_ = "transpose failed, transposing a graph with edge ID, "
                                            "type pairs unimplemented.";
                    return rocgraph_status_not_implemented;
                }

                auto p_graph = reinterpret_cast<
                    rocgraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
                    graph->graph_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph->number_map_);

                auto optional_edge_weights = std::optional<
                    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                    weight_t>>(std::nullopt);

                if(graph->edge_weights_ != nullptr)
                {
                    auto edge_weights = reinterpret_cast<
                        edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                        weight_t>*>(graph->edge_weights_);
                    optional_edge_weights = std::make_optional(std::move(*edge_weights));
                    delete edge_weights;
                }

                auto graph_transposed
                    = new rocgraph::graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>(handle);

                std::optional<rmm::device_uvector<vertex_t>> new_number_map{std::nullopt};

                auto new_optional_edge_weights = std::optional<
                    edge_property_t<graph_view_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
                                    weight_t>>(std::nullopt);

                std::tie(*graph_transposed, new_optional_edge_weights, new_number_map)
                    = rocgraph::transpose_graph_storage(
                        handle,
                        std::move(*p_graph),
                        std::move(optional_edge_weights),
                        std::make_optional<rmm::device_uvector<vertex_t>>(std::move(*number_map)));

                *number_map = std::move(new_number_map.value());

                delete p_graph;

                if(new_optional_edge_weights)
                {
                    auto new_edge_weights = new rocgraph::edge_property_t<
                        rocgraph::graph_view_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
                        weight_t>(handle);

                    *new_edge_weights    = std::move(new_optional_edge_weights.value());
                    graph->edge_weights_ = new_edge_weights;
                }

                graph->graph_            = graph_transposed;
                graph->store_transposed_ = !store_transposed;

                return rocgraph_status_success;
            }
            else
            {
                error->error_message_ = "transpose failed, value of transpose does not match graph";
                return rocgraph_status_invalid_input;
            }
        }

    } // namespace c_api
} // namespace rocgraph
