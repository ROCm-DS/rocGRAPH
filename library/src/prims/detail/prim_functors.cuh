// Copyright (C) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "edge_partition_device_view_device.hpp"

namespace rocgraph
{

    namespace detail
    {

        template <typename GraphViewType,
                  typename key_t,
                  typename EdgePartitionSrcValueInputWrapper,
                  typename EdgePartitionDstValueInputWrapper,
                  typename EdgePartitionEdgeValueInputWrapper,
                  typename EdgeOp>
        struct call_e_op_t
        {
            edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                         typename GraphViewType::edge_type,
                                         GraphViewType::is_multi_gpu> const& edge_partition{};
            EdgePartitionSrcValueInputWrapper const&   edge_partition_src_value_input{};
            EdgePartitionDstValueInputWrapper const&   edge_partition_dst_value_input{};
            EdgePartitionEdgeValueInputWrapper const&  edge_partition_e_value_input{};
            EdgeOp const&                              e_op{};
            key_t                                      key{};
            typename GraphViewType::vertex_type        major_offset{};
            typename GraphViewType::vertex_type const* indices{
                nullptr}; // indices = edge_partition.incies() + edge_offset
            typename GraphViewType::edge_type edge_offset{};

            __host__ __device__ auto operator()(
                typename GraphViewType::edge_type i /* index in key's neighbor list */) const
            {
                auto minor        = indices[i];
                auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
                std::conditional_t<GraphViewType::is_storage_transposed,
                                   typename GraphViewType::vertex_type,
                                   key_t>
                    key_or_src{};
                std::conditional_t<GraphViewType::is_storage_transposed,
                                   key_t,
                                   typename GraphViewType::vertex_type>
                    key_or_dst{};
                if constexpr(GraphViewType::is_storage_transposed)
                {
                    key_or_src = minor;
                    key_or_dst = key;
                }
                else
                {
                    key_or_src = key;
                    key_or_dst = minor;
                }
                auto src_offset
                    = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
                auto dst_offset
                    = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
                return e_op(key_or_src,
                            key_or_dst,
                            edge_partition_src_value_input.get(src_offset),
                            edge_partition_dst_value_input.get(dst_offset),
                            edge_partition_e_value_input.get(edge_offset + i));
            }
        };

        template <typename GraphViewType,
                  typename key_t,
                  typename EdgePartitionSrcValueInputWrapper,
                  typename EdgePartitionDstValueInputWrapper,
                  typename EdgePartitionEdgeValueInputWrapper,
                  typename EdgeOp>
        struct call_e_op_with_key_t
        {
            edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                         typename GraphViewType::edge_type,
                                         GraphViewType::is_multi_gpu> const& edge_partition{};
            EdgePartitionSrcValueInputWrapper const&  edge_partition_src_value_input{};
            EdgePartitionDstValueInputWrapper const&  edge_partition_dst_value_input{};
            EdgePartitionEdgeValueInputWrapper const& edge_partition_e_value_input{};
            EdgeOp const&                             e_op{};

            __host__ __device__ auto operator()(
                key_t                             key,
                typename GraphViewType::edge_type i /* index in edge_partition's edge list */) const
            {
                typename GraphViewType::vertex_type major{};
                if constexpr(std::is_same_v<key_t, typename GraphViewType::vertex_type>)
                {
                    major = key;
                }
                else
                {
                    major = thrust::get<0>(key);
                }
                auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
                auto minor        = *(edge_partition.indices() + i);
                auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
                std::conditional_t<GraphViewType::is_storage_transposed,
                                   typename GraphViewType::vertex_type,
                                   key_t>
                    key_or_src{};
                std::conditional_t<GraphViewType::is_storage_transposed,
                                   key_t,
                                   typename GraphViewType::vertex_type>
                    key_or_dst{};
                if constexpr(GraphViewType::is_storage_transposed)
                {
                    key_or_src = minor;
                    key_or_dst = key;
                }
                else
                {
                    key_or_src = key;
                    key_or_dst = minor;
                }
                auto src_offset
                    = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
                auto dst_offset
                    = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
                return e_op(key_or_src,
                            key_or_dst,
                            edge_partition_src_value_input.get(src_offset),
                            edge_partition_dst_value_input.get(dst_offset),
                            edge_partition_e_value_input.get(i));
            }
        };

    } // namespace detail

} // namespace rocgraph
