// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_generic_cascaded_dispatch.hpp"
#include "c_api/rocgraph_graph.hpp"

namespace rocgraph
{
    namespace c_api
    {

        template <typename functor_t, typename result_t>
        rocgraph_status run_algorithm(::rocgraph_graph_t const* graph,
                                      functor_t&                functor,
                                      result_t*                 result,
                                      ::rocgraph_error_t**      error)
        {
            *result = result_t{};
            *error  = nullptr;

            try
            {
                auto p_graph = reinterpret_cast<rocgraph::c_api::rocgraph_graph_t const*>(graph);

                rocgraph::c_api::vertex_dispatcher(p_graph->vertex_type_,
                                                   p_graph->edge_type_,
                                                   p_graph->weight_type_,
                                                   p_graph->edge_type_id_type_,
                                                   p_graph->store_transposed_,
                                                   p_graph->multi_gpu_,
                                                   functor);

                if(functor.status_ != rocgraph_status_success)
                {
                    *error = reinterpret_cast<::rocgraph_error_t*>(functor.error_.release());
                    return functor.status_;
                }

                if constexpr(std::is_same_v<result_t, decltype(functor.result_)>)
                {
                    *result = functor.result_;
                }
                else
                {
                    *result = reinterpret_cast<result_t>(functor.result_);
                }
            }
            catch(std::exception const& ex)
            {
                *error = reinterpret_cast<::rocgraph_error_t*>(
                    new rocgraph::c_api::rocgraph_error_t{ex.what()});
                return rocgraph_status_unknown_error;
            }

            return rocgraph_status_success;
        }

    } // namespace c_api
} // namespace rocgraph
