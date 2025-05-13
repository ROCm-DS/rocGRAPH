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

#include "mtmg/detail/device_shared_device_vector.hpp"
#include "mtmg/renumber_map_view.hpp"

namespace rocgraph
{
    namespace mtmg
    {

        /**
 * @brief An MTMG device vector for storing a renumber map
 */
        template <typename vertex_t>
        class renumber_map_t : public detail::device_shared_device_vector_t<vertex_t>
        {
            using parent_t = detail::device_shared_device_vector_t<vertex_t>;

        public:
            /**
   * @brief Return a view (read only) of the renumber map
   */
            auto view()
            {
                return static_cast<renumber_map_view_t<vertex_t>>(this->parent_t::view());
            }
        };

    } // namespace mtmg
} // namespace rocgraph
