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

#include "graph_view.hpp"
#include "mtmg/detail/device_shared_wrapper.hpp"
#include "mtmg/handle.hpp"

namespace rocgraph
{
    namespace mtmg
    {

        /**
 * @brief Graph view for each GPU
 */
        template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
        class graph_view_t
            : public detail::device_shared_wrapper_t<
                  rocgraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>>
        {
        public:
            /**
   * @brief Get the vertex_partition_view for this graph
   */
            vertex_partition_view_t<vertex_t, multi_gpu>
                get_vertex_partition_view(rocgraph::mtmg::handle_t const& handle) const
            {
                return this->get(handle).local_vertex_partition_view();
            }

            /**
   * @brief Get the vertex_partition_view for this graph
   */
            std::vector<vertex_t>
                get_vertex_partition_range_lasts(rocgraph::mtmg::handle_t const& handle) const
            {
                return this->get(handle).vertex_partition_range_lasts();
            }
        };

    } // namespace mtmg
} // namespace rocgraph
