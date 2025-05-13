// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <vector>

namespace raft
{

    /**
 * @brief A container policy for host mdarray.
 */
    template <typename ElementType, typename Allocator = std::allocator<ElementType>>
    class host_vector_policy
    {
    public:
        using element_type          = ElementType;
        using container_type        = std::vector<element_type, Allocator>;
        using allocator_type        = typename container_type::allocator_type;
        using pointer               = typename container_type::pointer;
        using const_pointer         = typename container_type::const_pointer;
        using reference             = element_type&;
        using const_reference       = element_type const&;
        using accessor_policy       = std::experimental::default_accessor<element_type>;
        using const_accessor_policy = std::experimental::default_accessor<element_type const>;

    public:
        auto create(raft::resources const&, size_t n) -> container_type
        {
            return container_type(n);
        }

        constexpr host_vector_policy() noexcept(
            std::is_nothrow_default_constructible_v<ElementType>)
            = default;

        [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
        {
            return c[n];
        }
        [[nodiscard]] constexpr auto access(container_type const& c,
                                            size_t n) const noexcept -> const_reference
        {
            return c[n];
        }

        [[nodiscard]] auto make_accessor_policy() noexcept
        {
            return accessor_policy{};
        }
        [[nodiscard]] auto make_accessor_policy() const noexcept
        {
            return const_accessor_policy{};
        }
    };

    // NOTE(HIP/AMD): This specialization is necessary,
    // as it is matched against a template template
    // parameter which only accepts one nested template parameter
    // (and host_vector_policy has two parameters, although one is defaulted).
    // This is a workaround for a problem specific to clang for
    // which the compile option -frelaxed-template-template-args might
    // be used. Unfortunately, this compiler flag is not compatible
    // with rocthrust.
    template <typename T>
    using host_vector_policy_default_allocator = host_vector_policy<T, std::allocator<T>>;

} // namespace raft
