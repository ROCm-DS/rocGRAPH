/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_data.hpp"
#include "rocgraph_test.hpp"
#include "rocgraph_test_template_traits.hpp"

#include "test_check.hpp"

//
// INTERNAL MACRO TO SPECIALIZE TEST CALL NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCGRAPH_TEST_CALL(ROUTINE)                                                    \
    template <>                                                                                   \
    struct rocgraph_test_call<rocgraph_test_enum::ROUTINE>                                        \
    {                                                                                             \
        template <typename... P>                                                                  \
        static void testing_bad_arg(const Arguments& arg)                                         \
        {                                                                                         \
            test_check::reset_auto_testing_bad_arg();                                             \
            const int state_debug_arguments_verbose = rocgraph_state_debug_arguments_verbose();   \
            if(state_debug_arguments_verbose == 1)                                                \
            {                                                                                     \
                rocgraph_disable_debug_arguments_verbose();                                       \
            }                                                                                     \
            testing_##ROUTINE##_bad_arg<P...>(arg);                                               \
            if(state_debug_arguments_verbose == 1)                                                \
            {                                                                                     \
                rocgraph_enable_debug_arguments_verbose();                                        \
            }                                                                                     \
            if(false && false == test_check::did_auto_testing_bad_arg())                          \
            {                                                                                     \
                std::cerr << "rocgraph_test warning testing bad arguments of "                    \
                          << rocgraph_test_enum::to_string(rocgraph_test_enum::ROUTINE)           \
                          << " must use auto_testing_bad_arg, or bad_arg_analysis." << std::endl; \
                CHECK_ROCGRAPH_SUCCESS(rocgraph_status_internal_error);                           \
            }                                                                                     \
        }                                                                                         \
        static void testing_extra(const Arguments& arg)                                           \
        {                                                                                         \
            try                                                                                   \
            {                                                                                     \
                testing_##ROUTINE##_extra(arg);                                                   \
            }                                                                                     \
            catch(rocgraph_status & status)                                                       \
            {                                                                                     \
                CHECK_ROCGRAPH_SUCCESS(status);                                                   \
            }                                                                                     \
            catch(hipError_t & error)                                                             \
            {                                                                                     \
                CHECK_HIP_SUCCESS(error);                                                         \
            }                                                                                     \
            catch(std::exception & error)                                                         \
            {                                                                                     \
                CHECK_ROCGRAPH_SUCCESS(rocgraph_status_thrown_exception);                         \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        template <typename... P>                                                                  \
        static void testing(const Arguments& arg)                                                 \
        {                                                                                         \
            try                                                                                   \
            {                                                                                     \
                testing_##ROUTINE<P...>(arg);                                                     \
            }                                                                                     \
            catch(rocgraph_status & status)                                                       \
            {                                                                                     \
                CHECK_ROCGRAPH_SUCCESS(status);                                                   \
            }                                                                                     \
            catch(hipError_t & error)                                                             \
            {                                                                                     \
                CHECK_HIP_SUCCESS(error);                                                         \
            }                                                                                     \
            catch(std::exception & error)                                                         \
            {                                                                                     \
                CHECK_ROCGRAPH_SUCCESS(rocgraph_status_thrown_exception);                         \
            }                                                                                     \
        }                                                                                         \
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INTERNAL MACRO TO SPECIALIZE TEST FUNCTOR NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCGRAPH_TEST_FUNCTORS(ROUTINE, ...)                              \
    /**/ template <> /**/ struct rocgraph_test_functors<rocgraph_test_enum::ROUTINE> \
    /**/ {                                                                           \
        /**/ static std::string name_suffix(const Arguments& arg)                    \
        /**/ {                                                                       \
            /**/ std::ostringstream s;                                               \
            /**/ rocgraph_test_name_suffix_generator(s, __VA_ARGS__);                \
            /**/ return s.str();                                                     \
      /**/       }                                                                   \
    /**/ }
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INTERNAL MACRO TO SPECIALIZE TEST TRAITS NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCGRAPH_TEST_TRAITS(ROUTINE, CONFIG)                                    \
    /**/ template <> /**/ struct rocgraph_test_traits<rocgraph_test_enum::ROUTINE> : CONFIG \
    /**/ {                                                                                  \
  /**/ }
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INSTANTIATE TESTS
//

template <rocgraph_test_enum::value_type ROUTINE>
using test_template_traits_t
    = rocgraph_test_template_traits<ROUTINE, rocgraph_test_traits<ROUTINE>::s_dispatch>;

template <rocgraph_test_enum::value_type ROUTINE>
using test_dispatch_t = rocgraph_test_dispatch<rocgraph_test_traits<ROUTINE>::s_dispatch>;

#define INSTANTIATE_ROCGRAPH_TEST(ROUTINE, CATEGORY)                                               \
    /**/ using ROUTINE = test_template_traits_t<rocgraph_test_enum::ROUTINE>::filter;              \
    /**/                                                                                           \
    /**/ template <typename... P>                                                                  \
    /**/ using ROUTINE##_call = test_template_traits_t<rocgraph_test_enum::ROUTINE>::caller<P...>; \
    /**/                                                                                           \
    /**/ TEST_P(ROUTINE, CATEGORY)                                                                 \
    /**/ {                                                                                         \
        /**/ test_dispatch_t<rocgraph_test_enum::ROUTINE>::template dispatch<ROUTINE##_call>(      \
            GetParam());                                                                           \
    /**/ }                                                                                         \
    /**/                                                                                           \
    /**/ INSTANTIATE_TEST_CATEGORIES(ROUTINE)

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// DEFINE ALL REQUIRED INFORMATION FOR A TEST ROUTINE BUT WITH A PREDEFINED CONFIGURATION
// (i.e. [T (default) | <I,T> | <I,J,T>] + a selection of numeric types (all (default), real_only, complex_only, some other specific situations (?) ) )
//
#define TEST_ROUTINE_WITH_CONFIG(ROUTINE, CATEGORY, CONFIG, ...)  \
    /**/                                                          \
    /**/ SPECIALIZE_ROCGRAPH_TEST_TRAITS(ROUTINE, CONFIG);        \
    /**/ SPECIALIZE_ROCGRAPH_TEST_CALL(ROUTINE);                  \
    /**/ SPECIALIZE_ROCGRAPH_TEST_FUNCTORS(ROUTINE, __VA_ARGS__); \
    /**/ namespace                                                \
    /**/ {                                                        \
        /**/ INSTANTIATE_ROCGRAPH_TEST(ROUTINE, CATEGORY);        \
    /**/ }

//
// DEFINE ALL REQUIRED INFORMATION FOR A TEST ROUTINE WITH A DEFAULT CONFIGURATION (i.e  T + all numeric types)
//
#define TEST_ROUTINE(ROUTINE, CATEGORY, ...) \
    TEST_ROUTINE_WITH_CONFIG(ROUTINE, CATEGORY, rocgraph_test_config, __VA_ARGS__)
/////////////////////////////////////////////////////////////////////////////////////////////////////
