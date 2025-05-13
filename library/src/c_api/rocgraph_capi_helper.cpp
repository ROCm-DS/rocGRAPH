// Copyright (C) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_capi_helper.hpp"

#include "structure/detail/structure_utils.cuh"

#include "detail/shuffle_wrappers.hpp"
#include "utilities/misc_utils_device.hpp"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

namespace rocgraph
{
    namespace c_api
    {
        namespace detail
        {

            template <typename vertex_t>
            std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>>
                shuffle_vertex_ids_and_offsets(raft::handle_t const&           handle,
                                               rmm::device_uvector<vertex_t>&& vertices,
                                               raft::device_span<size_t const> offsets)
            {
                auto ids = rocgraph::detail::expand_sparse_offsets(
                    offsets, vertex_t{0}, handle.get_stream());

                std::tie(vertices, ids) = rocgraph::detail::
                    shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                        handle, std::move(vertices), std::move(ids));

                thrust::sort(handle.get_thrust_policy(),
                             thrust::make_zip_iterator(ids.begin(), vertices.begin()),
                             thrust::make_zip_iterator(ids.end(), vertices.end()));

                auto return_offsets
                    = rocgraph::detail::compute_sparse_offsets<size_t>(ids.begin(),
                                                                       ids.end(),
                                                                       size_t{0},
                                                                       size_t{offsets.size() - 1},
                                                                       true,
                                                                       handle.get_stream());

                return std::make_tuple(std::move(vertices), std::move(return_offsets));
            }

            template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<size_t>>
                shuffle_vertex_ids_and_offsets(raft::handle_t const&           handle,
                                               rmm::device_uvector<int32_t>&&  vertices,
                                               raft::device_span<size_t const> offsets);

            template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
                shuffle_vertex_ids_and_offsets(raft::handle_t const&           handle,
                                               rmm::device_uvector<int64_t>&&  vertices,
                                               raft::device_span<size_t const> offsets);

            template <typename key_t, typename value_t>
            void sort_by_key(raft::handle_t const&      handle,
                             raft::device_span<key_t>   keys,
                             raft::device_span<value_t> values)
            {
                thrust::sort_by_key(
                    handle.get_thrust_policy(), keys.begin(), keys.end(), values.begin());
            }

            template void sort_by_key(raft::handle_t const&      handle,
                                      raft::device_span<int32_t> keys,
                                      raft::device_span<int32_t> values);
            template void sort_by_key(raft::handle_t const&      handle,
                                      raft::device_span<int64_t> keys,
                                      raft::device_span<int64_t> values);

            template <typename vertex_t, typename weight_t>
            std::tuple<rmm::device_uvector<size_t>,
                       rmm::device_uvector<vertex_t>,
                       rmm::device_uvector<vertex_t>,
                       std::optional<rmm::device_uvector<weight_t>>>
                reorder_extracted_egonets(
                    raft::handle_t const&                          handle,
                    rmm::device_uvector<size_t>&&                  source_indices,
                    rmm::device_uvector<size_t>&&                  offsets,
                    rmm::device_uvector<vertex_t>&&                edge_srcs,
                    rmm::device_uvector<vertex_t>&&                edge_dsts,
                    std::optional<rmm::device_uvector<weight_t>>&& edge_weights)
            {
                rmm::device_uvector<size_t> sort_indices(edge_srcs.size(), handle.get_stream());
                thrust::tabulate(
                    handle.get_thrust_policy(),
                    sort_indices.begin(),
                    sort_indices.end(),
                    [offset_lasts
                     = raft::device_span<size_t const>(offsets.begin() + 1, offsets.end()),
                     source_indices = raft::device_span<size_t const>(
                         source_indices.data(), source_indices.size())] __device__(size_t i) {
                        auto idx = static_cast<size_t>(thrust::distance(
                            offset_lasts.begin(),
                            thrust::upper_bound(
                                thrust::seq, offset_lasts.begin(), offset_lasts.end(), i)));
                        return source_indices[idx];
                    });
                source_indices.resize(0, handle.get_stream());
                source_indices.shrink_to_fit(handle.get_stream());

                auto triplet_first = thrust::make_zip_iterator(
                    sort_indices.begin(), edge_srcs.begin(), edge_dsts.begin());
                if(edge_weights)
                {
                    thrust::sort_by_key(handle.get_thrust_policy(),
                                        triplet_first,
                                        triplet_first + sort_indices.size(),
                                        (*edge_weights).begin());
                }
                else
                {
                    thrust::sort(handle.get_thrust_policy(),
                                 triplet_first,
                                 triplet_first + sort_indices.size());
                }

                thrust::tabulate(
                    handle.get_thrust_policy(),
                    offsets.begin() + 1,
                    offsets.end(),
                    [sort_indices = raft::device_span<size_t const>(
                         sort_indices.data(), sort_indices.size())] __device__(size_t i) {
                        return static_cast<size_t>(thrust::distance(
                            sort_indices.begin(),
                            thrust::upper_bound(
                                thrust::seq, sort_indices.begin(), sort_indices.end(), i)));
                    });

                return std::make_tuple(std::move(offsets),
                                       std::move(edge_srcs),
                                       std::move(edge_dsts),
                                       std::move(edge_weights));
            }

            template std::tuple<rmm::device_uvector<size_t>,
                                rmm::device_uvector<int32_t>,
                                rmm::device_uvector<int32_t>,
                                std::optional<rmm::device_uvector<float>>>
                reorder_extracted_egonets(raft::handle_t const&          handle,
                                          rmm::device_uvector<size_t>&&  source_indices,
                                          rmm::device_uvector<size_t>&&  offsets,
                                          rmm::device_uvector<int32_t>&& edge_srcs,
                                          rmm::device_uvector<int32_t>&& edge_dsts,
                                          std::optional<rmm::device_uvector<float>>&& edge_weights);

            template std::tuple<rmm::device_uvector<size_t>,
                                rmm::device_uvector<int32_t>,
                                rmm::device_uvector<int32_t>,
                                std::optional<rmm::device_uvector<double>>>
                reorder_extracted_egonets(
                    raft::handle_t const&                        handle,
                    rmm::device_uvector<size_t>&&                source_indices,
                    rmm::device_uvector<size_t>&&                offsets,
                    rmm::device_uvector<int32_t>&&               edge_srcs,
                    rmm::device_uvector<int32_t>&&               edge_dsts,
                    std::optional<rmm::device_uvector<double>>&& edge_weights);

            template std::tuple<rmm::device_uvector<size_t>,
                                rmm::device_uvector<int64_t>,
                                rmm::device_uvector<int64_t>,
                                std::optional<rmm::device_uvector<float>>>
                reorder_extracted_egonets(raft::handle_t const&          handle,
                                          rmm::device_uvector<size_t>&&  source_indices,
                                          rmm::device_uvector<size_t>&&  offsets,
                                          rmm::device_uvector<int64_t>&& edge_srcs,
                                          rmm::device_uvector<int64_t>&& edge_dsts,
                                          std::optional<rmm::device_uvector<float>>&& edge_weights);

            template std::tuple<rmm::device_uvector<size_t>,
                                rmm::device_uvector<int64_t>,
                                rmm::device_uvector<int64_t>,
                                std::optional<rmm::device_uvector<double>>>
                reorder_extracted_egonets(
                    raft::handle_t const&                        handle,
                    rmm::device_uvector<size_t>&&                source_indices,
                    rmm::device_uvector<size_t>&&                offsets,
                    rmm::device_uvector<int64_t>&&               edge_srcs,
                    rmm::device_uvector<int64_t>&&               edge_dsts,
                    std::optional<rmm::device_uvector<double>>&& edge_weights);

        } // namespace detail
    } // namespace c_api
} // namespace rocgraph
