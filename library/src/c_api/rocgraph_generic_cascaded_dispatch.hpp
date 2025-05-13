// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "internal/types/rocgraph_handle_t.h"

#include <sstream>

namespace rocgraph
{
    namespace c_api
    {

#if 0
template <rocgraph_data_type_id>
struct translate_data_type;

template <>
struct translate_data_type<rocgraph_data_type_id_int32> {
  using type = int32_t;
};

template <>
struct translate_data_type<rocgraph_data_type_id_int64> {
  using type = int64_t;
};

template <>
struct translate_data_type<rocgraph_data_type_id_float32> {
  using type = float;
};

template <>
struct translate_data_type<rocgraph_data_type_id_float64> {
  using type = double;
};

template <>
struct translate_data_type<rocgraph_data_type_id_size_t> {
  using type = size_t;
};
#endif

// MULTI-GPU: Disabled.
#if 0
// multi_gpu bool dispatcher:
// resolves bool `multi_gpu`
// and using template arguments vertex_t, edge_t, weight_t, store_transpose
// Calls functor
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          typename functor_t>
constexpr decltype(auto) multi_gpu_dispatcher(bool multi_gpu, functor_t& functor)
{
  if (multi_gpu) {
    return functor
      .template operator()<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, true>();
  } else {
    return functor
      .template operator()<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, false>();
  }
}
#endif

        // transpose bool dispatcher:
        // resolves bool `store_transpose`
        // and using template arguments vertex_t, edge_t, weight_t
        // cascades into next level
        // multi_gpu_dispatcher()
        //
        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_t,
                  typename functor_t>
        constexpr decltype(auto)
            transpose_dispatcher(bool store_transposed, bool multi_gpu, functor_t& functor)
        {
            // MULTI-GPU: Removed level of the dispatch, inlining multi_gpu_dispatcher.
            constexpr bool multi_gpu_const = false;
            if(store_transposed)
            {
                constexpr bool          store_transposed_const = true;
                return functor.template operator()<vertex_t,
                                                   edge_t,
                                                   weight_t,
                                                   edge_type_t,
                                                   store_transposed_const,
                                                   multi_gpu_const>();
                //return multi_gpu_dispatcher<vertex_t, edge_t, weight_t, edge_type_t, true>(multi_gpu, functor);
            }
            else
            {
                constexpr bool          store_transposed_const = false;
                return functor.template operator()<vertex_t,
                                                   edge_t,
                                                   weight_t,
                                                   edge_type_t,
                                                   store_transposed_const,
                                                   multi_gpu_const>();
                //return multi_gpu_dispatcher<vertex_t, edge_t, weight_t, edge_type_t, false>(multi_gpu, functor);
            }
        }

        // edge_type_type type dispatcher:
        // resolves weight_t from weight_type enum
        // and using template arguments vertex_t, edge_t
        // cascades into next level
        // transpose_dispatcher()
        //
        template <typename vertex_t, typename edge_t, typename weight_t, typename functor_t>
        constexpr decltype(auto) edge_type_type_dispatcher(rocgraph_data_type_id edge_type_type,
                                                           bool                  store_transposed,
                                                           bool                  multi_gpu,
                                                           functor_t&            functor)
        {
            switch(edge_type_type)
            {
            case rocgraph_data_type_id_int32:
            {
                using edge_type_t = int32_t;
                return transpose_dispatcher<vertex_t, edge_t, weight_t, edge_type_t>(
                    store_transposed, multi_gpu, functor);
            }
            case rocgraph_data_type_id_int64:
            {
                throw std::runtime_error(
                    "ERROR: Data type INT64 not allowed for edge type (valid types: INT32).");
                break;
            }
            case rocgraph_data_type_id_float32:
            {
                throw std::runtime_error(
                    "ERROR: Data type FLOAT32 not allowed for edge type (valid types: INT32).");
                break;
            }
            case rocgraph_data_type_id_float64:
            {
                throw std::runtime_error(
                    "ERROR: Data type FLOAT64 not allowed for edge type (valid types: INT32).");
                break;
            }

            default:
            {
                // std::stringstream ss;
                // ss << "ERROR: Unknown type enum:" << static_cast<int>(edge_type_type);
                // throw std::runtime_error(ss.str());
                // TODO: need c++23 to have std::stringstream inside a constexpr function
                throw std::runtime_error("ERROR: Unknown type enum");
            }
            }
        }

        // weight type dispatcher:
        // resolves weight_t from weight_type enum
        // and using template arguments vertex_t, edge_t
        // cascades into next level
        // edge_type_type_dispatcher()
        //
        template <typename vertex_t, typename edge_t, typename functor_t>
        constexpr decltype(auto) weight_dispatcher(rocgraph_data_type_id weight_type,
                                                   rocgraph_data_type_id edge_type_type,
                                                   bool                  store_transposed,
                                                   bool                  multi_gpu,
                                                   functor_t&            functor)
        {
            switch(weight_type)
            {
            case rocgraph_data_type_id_int32:
            {
                using weight_t = int32_t;
                return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
                    edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_int64:
            {
                using weight_t = int64_t;
                return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
                    edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_float32:
            {
                using weight_t = float;
                return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
                    edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_float64:
            {
                using weight_t = double;
                return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
                    edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            default:
            {
                // std::stringstream ss;
                // ss << "ERROR: Unknown type enum:" << static_cast<int>(weight_type);
                // throw std::runtime_error(ss.str());
                // TODO: need c++23 to have std::stringstream inside a constexpr function
                throw std::runtime_error("ERROR: Unknown type enum");
            }
            }
        }

        // edge type dispatcher:
        // resolves edge_t from edge_type enum
        // and using template argument vertex_t
        // cascades into the next level
        // weight_dispatcher();
        //
        template <typename vertex_t, typename functor_t>
        constexpr decltype(auto) edge_dispatcher(rocgraph_data_type_id edge_type,
                                                 rocgraph_data_type_id weight_type,
                                                 rocgraph_data_type_id edge_type_type,
                                                 bool                  store_transposed,
                                                 bool                  multi_gpu,
                                                 functor_t&            functor)
        {
            switch(edge_type)
            {
            case rocgraph_data_type_id_int32:
            {
                using edge_t = int32_t;
                return weight_dispatcher<vertex_t, edge_t>(
                    weight_type, edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_int64:
            {
                using edge_t = int64_t;
                return weight_dispatcher<vertex_t, edge_t>(
                    weight_type, edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_float32:
            {
                throw std::runtime_error("ERROR: FLOAT32 not supported for a vertex type");
            }
            break;
            case rocgraph_data_type_id_float64:
            {
                throw std::runtime_error("ERROR: FLOAT64 not supported for a vertex type");
            }
            break;
            default:
            {
                // std::stringstream ss;
                // ss << "ERROR: Unknown type enum:" << static_cast<int>(edge_type);
                // throw std::runtime_error(ss.str());
                // TODO: need c++23 to have std::stringstream inside a constexpr function
                throw std::runtime_error("ERROR: Unknown type enum");
            }
            }
        }

        // vertex type dispatcher:
        // entry point,
        // resolves vertex_t from vertex_type enum
        // and  cascades into the next level
        // edge_dispatcher();
        //
        template <typename functor_t>
        inline decltype(auto) vertex_dispatcher(rocgraph_data_type_id vertex_type,
                                                rocgraph_data_type_id edge_type,
                                                rocgraph_data_type_id weight_type,
                                                rocgraph_data_type_id edge_type_type,
                                                bool                  store_transposed,
                                                bool                  multi_gpu,
                                                functor_t&            functor)
        {
            switch(vertex_type)
            {
            case rocgraph_data_type_id_int32:
            {
                using vertex_t = int32_t;
                return edge_dispatcher<vertex_t>(
                    edge_type, weight_type, edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_int64:
            {
                using vertex_t = int64_t;
                return edge_dispatcher<vertex_t>(
                    edge_type, weight_type, edge_type_type, store_transposed, multi_gpu, functor);
            }
            break;
            case rocgraph_data_type_id_float32:
            {
                throw std::runtime_error("ERROR: FLOAT32 not supported for a vertex type");
            }
            break;
            case rocgraph_data_type_id_float64:
            {
                throw std::runtime_error("ERROR: FLOAT64 not supported for a vertex type");
            }
            break;
            default:
            {
                // std::stringstream ss;
                // ss << "ERROR: Unknown type enum:" << static_cast<int>(vertex_type);
                // throw std::runtime_error(ss.str());
                // TODO: need c++23 to have std::stringstream inside a constexpr function
                throw std::runtime_error("ERROR: Unknown type enum");
            }
            }
        }

    } // namespace c_api
} // namespace rocgraph
