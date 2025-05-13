// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
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

#include "rocgraph_clients_create_test_graph.hpp"
#include "rocgraph_test.hpp"

#include "rocgraph/rocgraph.h"
#include "testing.hpp"
#include "testing_rocgraph_generate_rmat_edgelist.hpp"

namespace
{

    // ugh.
    template <typename V, typename... T>
    constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)>
    {
        return {{std::forward<T>(t)...}};
    }

    /*
   * Simple rmat generator test
   */
    void RMAT(const Arguments& arg)
    //test_rmat_generation()
    {
        using vertex_t    = int32_t;
        auto expected_src = array_of<vertex_t>(17,
                                               18,
                                               0,
                                               16,
                                               1,
                                               24,
                                               16,
                                               1,
                                               6,
                                               4,
                                               2,
                                               1,
                                               14,
                                               2,
                                               16,
                                               2,
                                               5,
                                               23,
                                               4,
                                               10,
                                               4,
                                               3,
                                               0,
                                               4,
                                               11,
                                               0,
                                               0,
                                               2,
                                               24,
                                               0);
        auto expected_dst = array_of<vertex_t>(0,
                                               10,
                                               23,
                                               0,
                                               26,
                                               0,
                                               2,
                                               1,
                                               27,
                                               8,
                                               1,
                                               0,
                                               21,
                                               21,
                                               0,
                                               4,
                                               8,
                                               14,
                                               10,
                                               17,
                                               0,
                                               16,
                                               0,
                                               16,
                                               25,
                                               5,
                                               8,
                                               8,
                                               4,
                                               19);

        rocgraph_error_t* ret_error;

        rocgraph_handle_t*    p_handle  = nullptr;
        rocgraph_rng_state_t* rng_state = nullptr;
        rocgraph_coo_t*       coo       = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_generate_rmat_edgelist(p_handle,
                                                                     rng_state,
                                                                     5,
                                                                     30,
                                                                     0.57,
                                                                     0.19,
                                                                     0.19,
                                                                     rocgraph_bool_false,
                                                                     rocgraph_bool_false,
                                                                     &coo,
                                                                     &ret_error),
                                     ret_error);

        rocgraph_type_erased_device_array_view_t* src_view;
        rocgraph_type_erased_device_array_view_t* dst_view;

        src_view = rocgraph_coo_get_sources(coo);
        dst_view = rocgraph_coo_get_destinations(coo);

        size_t src_size = rocgraph_type_erased_device_array_view_size(src_view);

        std::vector<vertex_t> h_src(src_size);
        std::vector<vertex_t> h_dst(src_size);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_src.data(), src_view, &ret_error),
            ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_type_erased_device_array_view_copy_to_host(
                p_handle, (rocgraph_byte_t*)h_dst.data(), dst_view, &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(src_size, expected_src.data(), h_src.data());
        ROCGRAPH_CLIENTS_EXPECT_SEGMENT_EQ(src_size, expected_dst.data(), h_dst.data());

        rocgraph_type_erased_device_array_view_free(dst_view);
        rocgraph_type_erased_device_array_view_free(src_view);
        rocgraph_coo_free(coo);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }

    void RMATList(const Arguments& arg)
    {
        using vertex_t = int32_t;
        rocgraph_error_t* ret_error;

        rocgraph_handle_t*    p_handle  = nullptr;
        rocgraph_rng_state_t* rng_state = nullptr;
        ;
        rocgraph_coo_list_t* coo_list = nullptr;

        CHECK_ROCGRAPH_SUCCESS(rocgraph_create_handle(&p_handle, nullptr));

        //TEST_ALWAYS_ASSERT(ret_code == rocgraph_status_success, rocgraph_error_message(ret_error));

        //
        // NOTE: We can't exactly compare results for functions that make multiple RNG calls
        // within them.  When the RNG state is advanced, it is advanced by a multiple of
        // the number of possible threads involved, not based on how many of the values
        // were actually used.  So different GPU versions will result in subtly different
        // random sequences.
        //
        size_t   num_lists       = 3;
        vertex_t max_vertex_id[] = {32, 16, 32};
        size_t   expected_len[]  = {20, 16, 20};

        CHECK_ROCGRAPH_SUCCESS_ERROR(rocgraph_rng_state_create(p_handle, 0, &rng_state, &ret_error),
                                     ret_error);

        CHECK_ROCGRAPH_SUCCESS_ERROR(
            rocgraph_generate_rmat_edgelists(p_handle,
                                             rng_state,
                                             num_lists,
                                             4,
                                             6,
                                             4,
                                             rocgraph_generator_distribution_uniform,
                                             rocgraph_generator_distribution_power_law,
                                             rocgraph_bool_false,
                                             rocgraph_bool_false,
                                             &coo_list,
                                             &ret_error),
            ret_error);

        ROCGRAPH_CLIENTS_EXPECT_EQ(rocgraph_coo_list_size(coo_list), num_lists);

        for(size_t i = 0; i < num_lists; i++)
        {
            rocgraph_coo_t* coo = nullptr;

            coo = rocgraph_coo_list_element(coo_list, i);

            rocgraph_type_erased_device_array_view_t* src_view;
            rocgraph_type_erased_device_array_view_t* dst_view;

            src_view = rocgraph_coo_get_sources(coo);
            dst_view = rocgraph_coo_get_destinations(coo);

            size_t src_size = rocgraph_type_erased_device_array_view_size(src_view);

            ROCGRAPH_CLIENTS_EXPECT_EQ(src_size, expected_len[i]);

            std::vector<vertex_t> h_src(src_size);
            std::vector<vertex_t> h_dst(src_size);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_to_host(
                    p_handle, (rocgraph_byte_t*)h_src.data(), src_view, &ret_error),
                ret_error);

            CHECK_ROCGRAPH_SUCCESS_ERROR(
                rocgraph_type_erased_device_array_view_copy_to_host(
                    p_handle, (rocgraph_byte_t*)h_dst.data(), dst_view, &ret_error),
                ret_error);

            rocgraph_clients_expect_array_lt_scalar<vertex_t>(
                src_size, h_src.data(), max_vertex_id[i]);

            rocgraph_clients_expect_array_lt_scalar<vertex_t>(
                src_size, h_dst.data(), max_vertex_id[i]);

            rocgraph_type_erased_device_array_view_free(dst_view);
            rocgraph_type_erased_device_array_view_free(src_view);
        }

        rocgraph_coo_list_free(coo_list);
        rocgraph_destroy_handle(p_handle);
        rocgraph_error_free(ret_error);
    }
}

template <typename T>
void testing_rocgraph_generate_rmat_edgelist_bad_arg(const Arguments& arg)
{
#ifdef TO_FILL

    const rocgraph_handle_t* handle{};
    rocgraph_rng_state_t*    rng_state{};
    size_t                   scale{};
    size_t                   num_edges{};
    double                   a{};
    double                   b{};
    double                   c{};
    rocgraph_bool            clip_and_flip{};
    rocgraph_bool            scramble_vertex_ids{};
    rocgraph_coo_t**         result{};
    rocgraph_error_t**       error{};
    auto                     ret = rocgraph_generate_rmat_edgelist(handle,
                                               rng_state,
                                               scale,
                                               num_edges,
                                               a,
                                               b,
                                               c,
                                               clip_and_flip,
                                               scramble_vertex_ids,
                                               result,
                                               error);
    if(ret != rocgraph_status_success)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);
    }
    CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
#endif
}

template <typename T>
void testing_rocgraph_generate_rmat_edgelist(const Arguments& arg)
{
#ifdef TO_FILL
    //
    // Unit check.
    //
    if(arg.unit_check)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
    }

    if(arg.timing)
    {
        CHECK_ROCGRAPH_SUCCESS(rocgraph_status_not_implemented);
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        //
        // Warm-up
        //
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
        }

        double gpu_time_used = get_time_us();
        {
            //
            // Performance run
            //
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        display_timing_info(display_key_t::time_ms, get_gpu_time_msec(gpu_time_used));
    }
#endif
}

#define INSTANTIATE(TYPE)                                                                      \
    template void testing_rocgraph_generate_rmat_edgelist_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_rocgraph_generate_rmat_edgelist<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);

void testing_rocgraph_generate_rmat_edgelist_extra(const Arguments& arg)
{
    testing_dispatch_extra(arg, "RMAT", RMAT, "RMATList", RMATList);
}
