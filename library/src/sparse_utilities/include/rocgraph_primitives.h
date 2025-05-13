/*! \file */

/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "handle.h"

namespace rocgraph
{
    namespace primitives
    {
        template <class T>
        class double_buffer
        {
        private:
            T* buffers[2];

            unsigned int selector;

        public:
            __device__ __host__ inline double_buffer()
            {
                selector   = 0;
                buffers[0] = nullptr;
                buffers[1] = nullptr;
            }
            __device__ __host__ inline double_buffer(T* current, T* alternate)
            {
                selector   = 0;
                buffers[0] = current;
                buffers[1] = alternate;
            }

            __device__ __host__ inline T* current() const
            {
                return buffers[selector];
            }
            __device__ __host__ inline T* alternate() const
            {
                return buffers[selector ^ 1];
            }
            __device__ __host__ inline void swap()
            {
                selector ^= 1;
            }
        };

        template <typename K, typename V>
        rocgraph_status radix_sort_pairs_buffer_size(rocgraph_handle handle,
                                                     size_t          length,
                                                     uint32_t        startbit,
                                                     uint32_t        endbit,
                                                     size_t*         buffer_size,
                                                     bool            using_double_buffers = true);

        template <typename K, typename V>
        rocgraph_status radix_sort_pairs(rocgraph_handle   handle,
                                         double_buffer<K>& keys,
                                         double_buffer<V>& values,
                                         size_t            length,
                                         uint32_t          startbit,
                                         uint32_t          endbit,
                                         size_t            buffer_size,
                                         void*             buffer);

        template <typename K, typename V>
        rocgraph_status radix_sort_pairs(rocgraph_handle handle,
                                         K*              keys_input,
                                         K*              keys_output,
                                         V*              values_input,
                                         V*              values_output,
                                         size_t          length,
                                         uint32_t        startbit,
                                         uint32_t        endbit,
                                         size_t          buffer_size,
                                         void*           buffer);

        template <typename K>
        rocgraph_status radix_sort_keys_buffer_size(rocgraph_handle handle,
                                                    size_t          length,
                                                    uint32_t        startbit,
                                                    uint32_t        endbit,
                                                    size_t*         buffer_size);

        template <typename K>
        rocgraph_status radix_sort_keys(rocgraph_handle   handle,
                                        double_buffer<K>& keys,
                                        size_t            length,
                                        uint32_t          startbit,
                                        uint32_t          endbit,
                                        size_t            buffer_size,
                                        void*             buffer);

        template <typename J>
        rocgraph_status run_length_encode_buffer_size(rocgraph_handle handle,
                                                      size_t          length,
                                                      size_t*         buffer_size);

        template <typename J>
        rocgraph_status run_length_encode(rocgraph_handle handle,
                                          J*              input,
                                          J*              unique_output,
                                          J*              counts_output,
                                          J*              runs_count_output,
                                          size_t          length,
                                          size_t          buffer_size,
                                          void*           buffer);

        template <typename I, typename J>
        rocgraph_status exclusive_scan_buffer_size(rocgraph_handle handle,
                                                   J               initial_value,
                                                   size_t          length,
                                                   size_t*         buffer_size);

        template <typename I, typename J>
        rocgraph_status exclusive_scan(rocgraph_handle handle,
                                       I*              input,
                                       J*              output,
                                       J               initial_value,
                                       size_t          length,
                                       size_t          buffer_size,
                                       void*           buffer);

        template <typename I, typename J>
        rocgraph_status
            inclusive_scan_buffer_size(rocgraph_handle handle, size_t length, size_t* buffer_size);

        template <typename I, typename J>
        rocgraph_status inclusive_scan(rocgraph_handle handle,
                                       I*              input,
                                       J*              output,
                                       size_t          length,
                                       size_t          buffer_size,
                                       void*           buffer);

        template <typename I, typename J>
        rocgraph_status
            find_max_buffer_size(rocgraph_handle handle, size_t length, size_t* buffer_size);

        template <typename I, typename J>
        rocgraph_status find_max(rocgraph_handle handle,
                                 I*              input,
                                 J*              max,
                                 size_t          length,
                                 size_t          buffer_size,
                                 void*           buffer);

        template <typename I, typename J>
        rocgraph_status
            find_sum_buffer_size(rocgraph_handle handle, size_t length, size_t* buffer_size);

        template <typename I, typename J>
        rocgraph_status find_sum(rocgraph_handle handle,
                                 I*              input,
                                 J*              sum,
                                 size_t          length,
                                 size_t          buffer_size,
                                 void*           buffer);

        template <typename K, typename V, typename I>
        rocgraph_status segmented_radix_sort_pairs_buffer_size(rocgraph_handle handle,
                                                               size_t          length,
                                                               size_t          segments,
                                                               uint32_t        startbit,
                                                               uint32_t        endbit,
                                                               size_t*         buffer_size);

        template <typename K, typename V, typename I>
        rocgraph_status segmented_radix_sort_pairs(rocgraph_handle   handle,
                                                   double_buffer<K>& keys,
                                                   double_buffer<V>& values,
                                                   size_t            length,
                                                   size_t            segments,
                                                   I*                begin_offsets,
                                                   I*                end_offsets,
                                                   uint32_t          startbit,
                                                   uint32_t          endbit,
                                                   size_t            buffer_size,
                                                   void*             buffer);

        template <typename K, typename I>
        rocgraph_status segmented_radix_sort_keys_buffer_size(rocgraph_handle handle,
                                                              size_t          length,
                                                              size_t          segments,
                                                              uint32_t        startbit,
                                                              uint32_t        endbit,
                                                              size_t*         buffer_size);

        template <typename K, typename I>
        rocgraph_status segmented_radix_sort_keys(rocgraph_handle   handle,
                                                  double_buffer<K>& keys,
                                                  size_t            length,
                                                  size_t            segments,
                                                  I*                begin_offsets,
                                                  I*                end_offsets,
                                                  uint32_t          startbit,
                                                  uint32_t          endbit,
                                                  size_t            buffer_size,
                                                  void*             buffer);

        template <typename I, typename J>
        rocgraph_status sort_csr_column_indices_buffer_size(
            rocgraph_handle handle, J m, J n, I nnz, const I* csr_row_ptr, size_t* buffer_size);

        template <typename I, typename J>
        rocgraph_status sort_csr_column_indices(rocgraph_handle handle,
                                                J               m,
                                                J               n,
                                                I               nnz,
                                                const I*        csr_row_ptr,
                                                const J*        csr_col_ind,
                                                J*              csr_col_ind_buffer1,
                                                J*              csr_col_ind_buffer2,
                                                void*           buffer);

        template <typename I, typename J>
        rocgraph_status sort_csr_column_indices(rocgraph_handle handle,
                                                J               m,
                                                J               n,
                                                I               nnz,
                                                const I*        csr_row_ptr,
                                                J*              csr_col_ind,
                                                J*              csr_col_ind_buffer1,
                                                void*           buffer);
    }
}
