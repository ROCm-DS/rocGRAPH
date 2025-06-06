// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "detail/utility_wrappers.hpp"
#include "utilities/error.hpp"
#include "utilities/host_scalar_comm.hpp"

#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace rocgraph
{
    namespace detail
    {

        template <typename value_t>
        void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                 value_t*                     d_value,
                                 size_t                       size,
                                 value_t                      min_value,
                                 value_t                      max_value,
                                 raft::random::RngState&      rng_state)
        {
            if constexpr(std::is_integral<value_t>::value)
            {
                raft::random::uniformInt<value_t, size_t>(
                    rng_state, d_value, size, min_value, max_value, stream_view.value());
            }
            else
            {
                raft::random::uniform<value_t, size_t>(
                    rng_state, d_value, size, min_value, max_value, stream_view.value());
            }
        }

        template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                          int32_t*                     d_value,
                                          size_t                       size,
                                          int32_t                      min_value,
                                          int32_t                      max_value,
                                          raft::random::RngState&      rng_state);

        template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                          int64_t*                     d_value,
                                          size_t                       size,
                                          int64_t                      min_value,
                                          int64_t                      max_value,
                                          raft::random::RngState&      rng_state);

        template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                          float*                       d_value,
                                          size_t                       size,
                                          float                        min_value,
                                          float                        max_value,
                                          raft::random::RngState&      rng_state);

        template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                          double*                      d_value,
                                          size_t                       size,
                                          double                       min_value,
                                          double                       max_value,
                                          raft::random::RngState&      rng_state);

        template <typename value_t>
        void scalar_fill(raft::handle_t const& handle, value_t* d_value, size_t size, value_t value)
        {
            thrust::fill_n(handle.get_thrust_policy(), d_value, size, value);
        }

        template void
            scalar_fill(raft::handle_t const& handle, int32_t* d_value, size_t size, int32_t value);

        template void
            scalar_fill(raft::handle_t const& handle, int64_t* d_value, size_t size, int64_t value);

        template void
            scalar_fill(raft::handle_t const& handle, size_t* d_value, size_t size, size_t value);

        template void
            scalar_fill(raft::handle_t const& handle, float* d_value, size_t size, float value);

        template void
            scalar_fill(raft::handle_t const& handle, double* d_value, size_t size, double value);

        template <typename value_t>
        void sequence_fill(rmm::cuda_stream_view const& stream_view,
                           value_t*                     d_value,
                           size_t                       size,
                           value_t                      start_value)
        {
            thrust::sequence(rmm::exec_policy(stream_view), d_value, d_value + size, start_value);
        }

        template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                                    int32_t*                     d_value,
                                    size_t                       size,
                                    int32_t                      start_value);

        template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                                    int64_t*                     d_value,
                                    size_t                       size,
                                    int64_t                      start_value);

        template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                                    uint64_t*                    d_value,
                                    size_t                       size,
                                    uint64_t                     start_value);

        template <typename vertex_t>
        vertex_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                           vertex_t const*              d_edgelist_srcs,
                                           vertex_t const*              d_edgelist_dsts,
                                           size_t                       num_edges)
        {
            auto edge_first
                = thrust::make_zip_iterator(thrust::make_tuple(d_edgelist_srcs, d_edgelist_dsts));

            return thrust::transform_reduce(
                rmm::exec_policy(stream_view),
                edge_first,
                edge_first + num_edges,
                [] __device__(auto e) { return std::max(thrust::get<0>(e), thrust::get<1>(e)); },
                vertex_t{0},
                thrust::maximum<vertex_t>());
        }

        template int32_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                                   int32_t const*               d_edgelist_srcs,
                                                   int32_t const*               d_edgelist_dsts,
                                                   size_t                       num_edges);

        template int64_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                                   int64_t const*               d_edgelist_srcs,
                                                   int64_t const*               d_edgelist_dsts,
                                                   size_t                       num_edges);

        template <typename vertex_t, typename edge_t>
        std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>>
            filter_degree_0_vertices(raft::handle_t const&           handle,
                                     rmm::device_uvector<vertex_t>&& d_vertices,
                                     rmm::device_uvector<edge_t>&&   d_out_degs)
        {
            auto zip_iter = thrust::make_zip_iterator(
                thrust::make_tuple(d_vertices.begin(), d_out_degs.begin()));

            ROCGRAPH_EXPECTS(d_vertices.size()
                                 < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                             "remove_if will fail, d_vertices.size() is too large");

            // FIXME: remove_if has a 32-bit overflow issue (https://github.com/NVIDIA/thrust/issues/1302)
            // Seems unlikely here so not going to work around this for now.
            auto zip_iter_end
                = thrust::remove_if(handle.get_thrust_policy(),
                                    zip_iter,
                                    zip_iter + d_vertices.size(),
                                    zip_iter,
                                    [] __device__(auto pair) { return thrust::get<1>(pair) == 0; });

            auto new_size = thrust::distance(zip_iter, zip_iter_end);
            d_vertices.resize(new_size, handle.get_stream());
            d_out_degs.resize(new_size, handle.get_stream());

            return std::make_tuple(std::move(d_vertices), std::move(d_out_degs));
        }

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
            filter_degree_0_vertices(raft::handle_t const&          handle,
                                     rmm::device_uvector<int32_t>&& d_vertices,
                                     rmm::device_uvector<int32_t>&& d_out_degs);

        template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int64_t>>
            filter_degree_0_vertices(raft::handle_t const&          handle,
                                     rmm::device_uvector<int32_t>&& d_vertices,
                                     rmm::device_uvector<int64_t>&& d_out_degs);

        template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
            filter_degree_0_vertices(raft::handle_t const&          handle,
                                     rmm::device_uvector<int64_t>&& d_vertices,
                                     rmm::device_uvector<int64_t>&& d_out_degs);

        template <typename data_t>
        bool is_sorted(raft::handle_t const& handle, raft::device_span<data_t> span)
        {
            return thrust::is_sorted(handle.get_thrust_policy(), span.begin(), span.end());
        }

        template bool is_sorted(raft::handle_t const& handle, raft::device_span<int32_t> span);
        template bool is_sorted(raft::handle_t const&            handle,
                                raft::device_span<int32_t const> span);
        template bool is_sorted(raft::handle_t const& handle, raft::device_span<int64_t> span);
        template bool is_sorted(raft::handle_t const&            handle,
                                raft::device_span<int64_t const> span);

        template <typename data_t>
        bool is_equal(raft::handle_t const&     handle,
                      raft::device_span<data_t> span1,
                      raft::device_span<data_t> span2)
        {
            return thrust::equal(
                handle.get_thrust_policy(), span1.begin(), span1.end(), span2.begin());
        }

        template bool is_equal(raft::handle_t const&      handle,
                               raft::device_span<int32_t> span1,
                               raft::device_span<int32_t> span2);
        template bool is_equal(raft::handle_t const&            handle,
                               raft::device_span<int32_t const> span1,
                               raft::device_span<int32_t const> span2);
        template bool is_equal(raft::handle_t const&      handle,
                               raft::device_span<int64_t> span1,
                               raft::device_span<int64_t> span2);
        template bool is_equal(raft::handle_t const&            handle,
                               raft::device_span<int64_t const> span1,
                               raft::device_span<int64_t const> span2);

        template <typename data_t>
        size_t count_values(raft::handle_t const&           handle,
                            raft::device_span<data_t const> span,
                            data_t                          value)
        {
            return thrust::count(handle.get_thrust_policy(), span.begin(), span.end(), value);
        }

        template size_t count_values<int32_t>(raft::handle_t const&            handle,
                                              raft::device_span<int32_t const> span,
                                              int32_t                          value);
        template size_t count_values<int64_t>(raft::handle_t const&            handle,
                                              raft::device_span<int64_t const> span,
                                              int64_t                          value);

    } // namespace detail
} // namespace rocgraph
