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

#include "mtmg/detail/device_shared_device_span.hpp"
#include "mtmg/graph_view.hpp"
#include "mtmg/handle.hpp"
#include "mtmg/renumber_map.hpp"

#include <optional>

namespace rocgraph
{
    namespace mtmg
    {

        /**
 * @brief An MTMG device span for referencing a vertex result
 */
        template <typename result_t>
        class vertex_result_view_t : public detail::device_shared_device_span_t<result_t const>
        {
            using parent_t = detail::device_shared_device_span_t<result_t const>;

        public:
            vertex_result_view_t(parent_t&& other)
                : parent_t{std::move(other)}
            {
            }

            /**
   * @brief Gather results from specified vertices into a device vector
   */
            template <typename vertex_t, bool multi_gpu>
            rmm::device_uvector<result_t> gather(
                handle_t const&                                        handle,
                raft::device_span<vertex_t const>                      vertices,
                std::vector<vertex_t> const&                           vertex_partition_range_lasts,
                rocgraph::vertex_partition_view_t<vertex_t, multi_gpu> vertex_partition_view,
                std::optional<rocgraph::mtmg::renumber_map_view_t<vertex_t>>& renumber_map_view,
                result_t                                                      default_value = 0);
        };

    } // namespace mtmg
} // namespace rocgraph
