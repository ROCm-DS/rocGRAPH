// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>

#include <cstdint>

namespace raft
{

    template <typename AccessorPolicy>
    using managed_accessor = host_device_accessor<AccessorPolicy, memory_type::managed>;

    /**
 * @brief std::experimental::mdspan with managed tag to indicate host/device accessibility
 */
    template <typename ElementType,
              typename Extents,
              typename LayoutPolicy   = layout_c_contiguous,
              typename AccessorPolicy = std::experimental::default_accessor<ElementType>>
    using managed_mdspan
        = mdspan<ElementType, Extents, LayoutPolicy, managed_accessor<AccessorPolicy>>;

    template <typename T, bool B>
    struct is_managed_mdspan : std::false_type
    {
    };
    template <typename T>
    struct is_managed_mdspan<T, true>
        : std::bool_constant<T::accessor_type::mem_type == memory_type::managed>
    {
    };

    /**
 * @\brief Boolean to determine if template type T is either raft::managed_mdspan or a derived type
 */
    template <typename T>
    using is_managed_mdspan_t = is_managed_mdspan<T, is_mdspan_v<T>>;

    template <typename T>
    using is_input_managed_mdspan_t = is_managed_mdspan<T, is_input_mdspan_v<T>>;

    template <typename T>
    using is_output_managed_mdspan_t = is_managed_mdspan<T, is_output_mdspan_v<T>>;

    /**
 * @\brief Boolean to determine if variadic template types Tn are either raft::managed_mdspan or a
 * derived type
 */
    template <typename... Tn>
    inline constexpr bool is_managed_mdspan_v = std::conjunction_v<is_managed_mdspan_t<Tn>...>;

    template <typename... Tn>
    inline constexpr bool is_input_managed_mdspan_v
        = std::conjunction_v<is_input_managed_mdspan_t<Tn>...>;

    template <typename... Tn>
    inline constexpr bool is_output_managed_mdspan_v
        = std::conjunction_v<is_output_managed_mdspan_t<Tn>...>;

    template <typename... Tn>
    using enable_if_managed_mdspan = std::enable_if_t<is_managed_mdspan_v<Tn...>>;

    template <typename... Tn>
    using enable_if_input_managed_mdspan = std::enable_if_t<is_input_managed_mdspan_v<Tn...>>;

    template <typename... Tn>
    using enable_if_output_managed_mdspan = std::enable_if_t<is_output_managed_mdspan_v<Tn...>>;

    /**
 * @brief Shorthand for 0-dim managed mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
    template <typename ElementType, typename IndexType = std::uint32_t>
    using managed_scalar_view = managed_mdspan<ElementType, scalar_extent<IndexType>>;

    /**
 * @brief Shorthand for 1-dim managed mdspan.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
    template <typename ElementType,
              typename IndexType    = std::uint32_t,
              typename LayoutPolicy = layout_c_contiguous>
    using managed_vector_view = managed_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;

    /**
 * @brief Shorthand for c-contiguous managed matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
    template <typename ElementType,
              typename IndexType    = std::uint32_t,
              typename LayoutPolicy = layout_c_contiguous>
    using managed_matrix_view = managed_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

    /**
 * @brief Shorthand for 128 byte aligned managed matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 */
    template <typename ElementType,
              typename IndexType    = std::uint32_t,
              typename LayoutPolicy = layout_right_padded<ElementType>,
              typename              = enable_if_layout_padded<ElementType, LayoutPolicy>>
    using managed_aligned_matrix_view = managed_mdspan<
        ElementType,
        matrix_extent<IndexType>,
        LayoutPolicy,
        std::experimental::aligned_accessor<ElementType, detail::alignment::value>>;

    /**
 * @brief Create a 2-dim 128 byte aligned mdspan instance for managed pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 * @tparam IndexType the index type of the extents
 * @param[in] ptr to managed memory to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
    template <typename ElementType,
              typename IndexType    = std::uint32_t,
              typename LayoutPolicy = layout_right_padded<ElementType>>
    auto constexpr make_managed_aligned_matrix_view(ElementType* ptr,
                                                    IndexType    n_rows,
                                                    IndexType    n_cols)
    {
        using data_handle_type = typename std::experimental::
            aligned_accessor<ElementType, detail::alignment::value>::data_handle_type;
        static_assert(std::is_same<LayoutPolicy, layout_left_padded<ElementType>>::value
                      || std::is_same<LayoutPolicy, layout_right_padded<ElementType>>::value);
        assert(reinterpret_cast<std::uintptr_t>(ptr)
               == std::experimental::details::alignTo(reinterpret_cast<std::uintptr_t>(ptr),
                                                      detail::alignment::value));

        data_handle_type aligned_pointer = ptr;

        matrix_extent<IndexType> extents{n_rows, n_cols};
        return managed_aligned_matrix_view<ElementType, IndexType, LayoutPolicy>{aligned_pointer,
                                                                                 extents};
    }

    /**
 * @brief Create a 0-dim (scalar) mdspan instance for managed value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr to managed memory to wrap
 */
    template <typename ElementType, typename IndexType = std::uint32_t>
    auto constexpr make_managed_scalar_view(ElementType* ptr)
    {
        scalar_extent<IndexType> extents;
        return managed_scalar_view<ElementType, IndexType>{ptr, extents};
    }

    /**
 * @brief Create a 2-dim c-contiguous mdspan instance for managed pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @tparam IndexType the index type of the extents
 * @param[in] ptr to managed memory to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
    template <typename ElementType,
              typename IndexType    = std::uint32_t,
              typename LayoutPolicy = layout_c_contiguous>
    auto constexpr make_managed_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
    {
        matrix_extent<IndexType> extents{n_rows, n_cols};
        return managed_matrix_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
    }

    /**
 * @brief Create a 2-dim mdspan instance for managed pointer with a strided layout
 *        that is restricted to stride 1 in the trailing dimension. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr to managed memory to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 * @param[in] stride leading dimension / stride of data
 */
    template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
    auto constexpr make_managed_strided_matrix_view(ElementType* ptr,
                                                    IndexType    n_rows,
                                                    IndexType    n_cols,
                                                    IndexType    stride)
    {
        constexpr auto is_row_major = std::is_same_v<LayoutPolicy, layout_c_contiguous>;
        IndexType      stride0      = is_row_major ? (stride > 0 ? stride : n_cols) : 1;
        IndexType      stride1      = is_row_major ? 1 : (stride > 0 ? stride : n_rows);

        assert(is_row_major ? stride0 >= n_cols : stride1 >= n_rows);
        matrix_extent<IndexType> extents{n_rows, n_cols};

        auto layout = make_strided_layout(extents, std::array<IndexType, 2>{stride0, stride1});
        return managed_matrix_view<ElementType, IndexType, layout_stride>{ptr, layout};
    }

    /**
 * @brief Create a 1-dim mdspan instance for managed pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr to managed memory to wrap
 * @param[in] n number of elements in pointer
 * @return raft::managed_vector_view
 */
    template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
    auto constexpr make_managed_vector_view(ElementType* ptr, IndexType n)
    {
        return managed_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, n};
    }

    /**
 * @brief Create a 1-dim mdspan instance for managed pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr to managed memory to wrap
 * @param[in] mapping The layout mapping to use for this vector
 * @return raft::managed_vector_view
 */
    template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
    auto constexpr make_managed_vector_view(
        ElementType*                                                             ptr,
        const typename LayoutPolicy::template mapping<vector_extent<IndexType>>& mapping)
    {
        return managed_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, mapping};
    }

    /**
 * @brief Create a raft::managed_mdspan
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param ptr Pointer to the data
 * @param exts dimensionality of the array (series of integers)
 * @return raft::managed_mdspan
 */
    template <typename ElementType,
              typename IndexType    = std::uint32_t,
              typename LayoutPolicy = layout_c_contiguous,
              size_t... Extents>
    auto constexpr make_managed_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
    {
        return make_mdspan<ElementType, IndexType, LayoutPolicy, true, true>(ptr, exts);
    }
} // end namespace raft
