// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2023-2024, NVIDIA CORPORATION.
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

#include "detail/shuffle_wrappers.hpp"
#include "mtmg/handle.hpp"

// FIXME: Could use std::span once compiler supports C++20
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

namespace rocgraph
{
    namespace mtmg
    {
        namespace detail
        {

            /**
 * @brief An edgelist for each GPU
 *
 * Manages an edge list for edges associated with a particular GPU.  Multiple threads
 * can call the append() method, possibly concurrently.  To avoid constantly copying
 * when the buffers fill up, the class will create a device buffer containing a
 * number of elements specified in the constructor.  When that device buffer is full
 * we will create a new buffer.
 *
 * When we try and use the edgelist we will consolidate the buffers, since at that
 * time we know the entire size required.
 *
 * Important note, the expectation is that this object will be used in two phases:
 *  1) The append() method will be used to fill buffers with edges
 *  2) The edges will be consumed to create a graph
 *
 * These two phases are expected to be disjoint.  The calling process is expected to
 * manage some barrier so that all threads are guaranteed to be completed before changing
 * phases.  If an append() call (part of the filling phase) overlaps with calls to
 * finalize_buffer(), consolidate_and_shuffle(), get_src(), get_dst(), get_wgt(),
 * get_edge_id() and get_edge_type() then the behavior is undefined (data might change
 * in some non-deterministic way).
 */
            template <typename vertex_t, typename weight_t, typename edge_t, typename edge_type_t>
            class per_device_edgelist_t
            {
            public:
                per_device_edgelist_t()                                        = delete;
                per_device_edgelist_t(per_device_edgelist_t const&)            = delete;
                per_device_edgelist_t& operator=(per_device_edgelist_t const&) = delete;
                per_device_edgelist_t& operator=(per_device_edgelist_t&&)      = delete;

                /**
   * @brief Construct a new per device edgelist t object
   *
   * @param handle             MTMG resource handle - used to identify GPU resources
   * @param device_buffer_size Number of edges to store in each device buffer
   * @param use_weight         Whether or not the edgelist will have weights
   * @param use_edge_id        Whether or not the edgelist will have edge ids
   * @param use_edge_type      Whether or not the edgelist will have edge types
   */
                per_device_edgelist_t(rocgraph::mtmg::handle_t const& handle,
                                      size_t                          device_buffer_size,
                                      bool                            use_weight,
                                      bool                            use_edge_id,
                                      bool                            use_edge_type)
                    : device_buffer_size_{device_buffer_size}
                    , current_pos_{0}
                    , src_{}
                    , dst_{}
                    , wgt_{std::nullopt}
                    , edge_id_{std::nullopt}
                    , edge_type_{std::nullopt}
                {
                    if(use_weight)
                    {
                        wgt_ = std::make_optional(std::vector<rmm::device_uvector<weight_t>>());
                    }

                    if(use_edge_id)
                    {
                        edge_id_ = std::make_optional(std::vector<rmm::device_uvector<edge_t>>());
                    }

                    if(use_edge_type)
                    {
                        edge_type_
                            = std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>());
                    }

                    create_new_buffers(handle);
                }

                /**
   * @brief Move construct a new per device edgelist t object
   *
   * @param other Object to move into this instance
   */
                per_device_edgelist_t(per_device_edgelist_t&& other)
                    : device_buffer_size_{other.device_buffer_size_}
                    , current_pos_{other.current_pos_}
                    , src_{std::move(other.src_)}
                    , dst_{std::move(other.dst_)}
                    , wgt_{std::move(other.wgt_)}
                    , edge_id_{std::move(other.edge_id_)}
                    , edge_type_{std::move(other.edge_type_)}
                {
                }

                /**
   * @brief Append a list of edges to the edge list
   *
   * @param handle     The resource handle
   * @param src        Source vertex id
   * @param dst        Destination vertex id
   * @param wgt        Edge weight
   * @param edge_id    Edge id
   * @param edge_type  Edge type
   */
                void append(handle_t const&                                   handle,
                            raft::host_span<vertex_t const>                   src,
                            raft::host_span<vertex_t const>                   dst,
                            std::optional<raft::host_span<weight_t const>>    wgt,
                            std::optional<raft::host_span<edge_t const>>      edge_id,
                            std::optional<raft::host_span<edge_type_t const>> edge_type)
                {
                    std::vector<std::tuple<size_t, size_t, size_t, size_t>> copy_positions;

                    {
                        std::lock_guard<std::mutex> lock(lock_);

                        size_t count = src.size();
                        size_t pos   = 0;

                        while(count > 0)
                        {
                            size_t copy_count
                                = std::min(count, (src_.back().size() - current_pos_));

                            copy_positions.push_back(
                                std::make_tuple(src_.size() - 1, current_pos_, pos, copy_count));

                            count -= copy_count;
                            pos += copy_count;
                            current_pos_ += copy_count;

                            if(current_pos_ == src_.back().size())
                            {
                                create_new_buffers(handle);
                            }
                        }
                    }

                    std::for_each(copy_positions.begin(),
                                  copy_positions.end(),
                                  [&handle,
                                   &this_src = src_,
                                   &src,
                                   &this_dst = dst_,
                                   &dst,
                                   &this_wgt = wgt_,
                                   &wgt,
                                   &this_edge_id = edge_id_,
                                   &edge_id,
                                   &this_edge_type = edge_type_,
                                   &edge_type](auto tuple) {
                                      auto [buffer_idx, buffer_pos, input_pos, copy_count] = tuple;

                                      raft::update_device(this_src[buffer_idx].begin() + buffer_pos,
                                                          src.begin() + input_pos,
                                                          copy_count,
                                                          handle.get_stream());

                                      raft::update_device(this_dst[buffer_idx].begin() + buffer_pos,
                                                          dst.begin() + input_pos,
                                                          copy_count,
                                                          handle.get_stream());

                                      if(this_wgt)
                                          raft::update_device((*this_wgt)[buffer_idx].begin()
                                                                  + buffer_pos,
                                                              wgt->begin() + input_pos,
                                                              copy_count,
                                                              handle.get_stream());

                                      if(this_edge_id)
                                          raft::update_device((*this_edge_id)[buffer_idx].begin()
                                                                  + buffer_pos,
                                                              edge_id->begin() + input_pos,
                                                              copy_count,
                                                              handle.get_stream());

                                      if(this_edge_type)
                                          raft::update_device((*this_edge_type)[buffer_idx].begin()
                                                                  + buffer_pos,
                                                              edge_type->begin() + input_pos,
                                                              copy_count,
                                                              handle.get_stream());
                                  });

                    handle.sync_stream();
                }

                /**
   * @brief  Mark the edgelist as ready for reading (all writes are complete)
   *
   * @param handle     The resource handle
   */
                void finalize_buffer(handle_t const& handle)
                {
                    src_.back().resize(current_pos_, handle.get_stream());
                    dst_.back().resize(current_pos_, handle.get_stream());
                    if(wgt_)
                        wgt_->back().resize(current_pos_, handle.get_stream());
                    if(edge_id_)
                        edge_id_->back().resize(current_pos_, handle.get_stream());
                    if(edge_type_)
                        edge_type_->back().resize(current_pos_, handle.get_stream());
                }

                bool use_weight() const
                {
                    return wgt_.has_value();
                }

                bool use_edge_id() const
                {
                    return edge_id_.has_value();
                }

                bool use_edge_type() const
                {
                    return edge_type_.has_value();
                }

                std::vector<rmm::device_uvector<vertex_t>>& get_src()
                {
                    return src_;
                }
                std::vector<rmm::device_uvector<vertex_t>>& get_dst()
                {
                    return dst_;
                }
                std::optional<std::vector<rmm::device_uvector<weight_t>>>& get_wgt()
                {
                    return wgt_;
                }
                std::optional<std::vector<rmm::device_uvector<edge_t>>>& get_edge_id()
                {
                    return edge_id_;
                }
                std::optional<std::vector<rmm::device_uvector<edge_type_t>>>& get_edge_type()
                {
                    return edge_type_;
                }

                /**
   * @brief Consolidate edgelists (if necessary) and shuffle to the proper GPU
   *
   * @param handle    The resource handle
   */
                void consolidate_and_shuffle(rocgraph::mtmg::handle_t const& handle,
                                             bool                            store_transposed)
                {
                    if(src_.size() > 1)
                    {
                        size_t total_size
                            = std::transform_reduce(src_.begin(),
                                                    src_.end(),
                                                    size_t{0},
                                                    std::plus<size_t>(),
                                                    [](auto& d_vector) { return d_vector.size(); });

                        resize_and_copy_buffers(handle.get_stream(), src_, total_size);
                        resize_and_copy_buffers(handle.get_stream(), dst_, total_size);
                        if(wgt_)
                            resize_and_copy_buffers(handle.get_stream(), *wgt_, total_size);
                        if(edge_id_)
                            resize_and_copy_buffers(handle.get_stream(), *edge_id_, total_size);
                        if(edge_type_)
                            resize_and_copy_buffers(handle.get_stream(), *edge_type_, total_size);
                    }

                    auto tmp_wgt = wgt_ ? std::make_optional(std::move((*wgt_)[0])) : std::nullopt;
                    auto tmp_edge_id
                        = edge_id_ ? std::make_optional(std::move((*edge_id_)[0])) : std::nullopt;
                    auto tmp_edge_type = edge_type_
                                             ? std::make_optional(std::move((*edge_type_)[0]))
                                             : std::nullopt;

                    std::tie(store_transposed ? dst_[0] : src_[0],
                             store_transposed ? src_[0] : dst_[0],
                             tmp_wgt,
                             tmp_edge_id,
                             tmp_edge_type)
                        = rocgraph::detail::
                            shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
                                handle.raft_handle(),
                                store_transposed ? std::move(dst_[0]) : std::move(src_[0]),
                                store_transposed ? std::move(src_[0]) : std::move(dst_[0]),
                                std::move(tmp_wgt),
                                std::move(tmp_edge_id),
                                std::move(tmp_edge_type));

                    if(tmp_wgt)
                        ((*wgt_)[0]) = std::move(*tmp_wgt);
                    if(tmp_edge_id)
                        ((*edge_id_)[0]) = std::move(*tmp_edge_id);
                    if(tmp_edge_type)
                        ((*edge_type_)[0]) = std::move(*tmp_edge_type);
                }

            private:
                template <typename T>
                void resize_and_copy_buffers(rmm::cuda_stream_view                stream,
                                             std::vector<rmm::device_uvector<T>>& buffer,
                                             size_t                               total_size)
                {
                    size_t pos = buffer[0].size();
                    buffer[0].resize(total_size, stream);

                    for(size_t i = 1; i < buffer.size(); ++i)
                    {
                        raft::copy(
                            buffer[0].data() + pos, buffer[i].data(), buffer[i].size(), stream);
                        pos += buffer[i].size();
                        buffer[i].resize(0, stream);
                        buffer[i].shrink_to_fit(stream);
                    }

                    std::vector<rmm::device_uvector<T>> new_buffer;
                    new_buffer.push_back(std::move(buffer[0]));
                    buffer = std::move(new_buffer);
                }

                void create_new_buffers(rocgraph::mtmg::handle_t const& handle)
                {
                    src_.emplace_back(device_buffer_size_, handle.get_stream());
                    dst_.emplace_back(device_buffer_size_, handle.get_stream());

                    if(wgt_)
                    {
                        wgt_->emplace_back(device_buffer_size_, handle.get_stream());
                    }

                    if(edge_id_)
                    {
                        edge_id_->emplace_back(device_buffer_size_, handle.get_stream());
                    }

                    if(edge_type_)
                    {
                        edge_type_->emplace_back(device_buffer_size_, handle.get_stream());
                    }

                    current_pos_ = 0;
                }

                mutable std::mutex lock_{};

                size_t current_pos_{0};
                size_t device_buffer_size_{0};

                std::vector<rmm::device_uvector<vertex_t>>                   src_{};
                std::vector<rmm::device_uvector<vertex_t>>                   dst_{};
                std::optional<std::vector<rmm::device_uvector<weight_t>>>    wgt_{};
                std::optional<std::vector<rmm::device_uvector<edge_t>>>      edge_id_{};
                std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_type_{};
            };

        } // namespace detail
    } // namespace mtmg
} // namespace rocgraph
