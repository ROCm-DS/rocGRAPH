/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief auto_testing_bad_arg.hpp provides common testing utilities
 */

#pragma once

#include "auto_testing_bad_arg_get_status.hpp"
#include "auto_testing_bad_arg_set_invalid.hpp"
#include "rocgraph_clients_envariables.hpp"
#include "rocgraph_test.hpp"
#include "test_check.hpp"
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <sstream>
#include <vector>

extern "C" {
rocgraph_status rocgraph_argdescr_create(void** argdescr);
rocgraph_status rocgraph_argdescr_free(void* argdescr);
rocgraph_status rocgraph_argdescr_get_msg(const void* argdescr, const char**);
rocgraph_status rocgraph_argdescr_get_status(const void* argdescr, rocgraph_status*);
rocgraph_status rocgraph_argdescr_get_index(const void* argdescr, int*);
rocgraph_status rocgraph_argdescr_get_name(const void* argdescr, const char**);
rocgraph_status rocgraph_argdescr_get_function_line(const void* argdescr, int*);
rocgraph_status rocgraph_argdescr_get_function_name(const void* argdescr, const char**);
}

template <typename... T>
struct auto_testing_bad_arg_t
{
    inline auto_testing_bad_arg_t() {};
    inline auto_testing_bad_arg_t(int current, int ith, rocgraph_status& status) {};
};

template <typename T, typename... Rest>
struct auto_testing_bad_arg_t<T, Rest...>
{
    inline auto_testing_bad_arg_t(T first, Rest... rest)
        : first(first)
        , rest(rest...)
    {
    }

    inline auto_testing_bad_arg_t(int ith, rocgraph_status& status, T& first, Rest&... rest)
        : auto_testing_bad_arg_t(0, ith, status, first, rest...)
    {
    }

    inline auto_testing_bad_arg_t(
        int current, int ith, rocgraph_status& status, T& first, Rest&... rest)
        : first(first)
        , rest(current + 1, ith, status, rest...)
    {
        if(current == ith)
        {
            status = auto_testing_bad_arg_get_status<T>(first);
            auto_testing_bad_arg_set_invalid(this->first);
        }
    }

    T                               first;
    auto_testing_bad_arg_t<Rest...> rest;
};

template <typename C, typename T>
inline void auto_testing_bad_arg_copy(const C& data, T& t)
{
    t = data.first;
}

template <typename C, typename T, typename... Ts>
inline void auto_testing_bad_arg_copy(const C& data, T& t, Ts&... ts)
{
    t = data.first;
    auto_testing_bad_arg_copy(data.rest, ts...);
}

template <typename T>
inline void auto_testing_bad_arg_print(T& t)
{
    std::cout << " " << t << "," << std::endl;
}

template <typename T, typename... Ts>
inline void auto_testing_bad_arg_print(T& t, Ts&... ts)
{
    std::cout << " " << t << "," << std::endl;
    auto_testing_bad_arg_print(ts...);
}

static constexpr uint32_t stringargs_count(const char* str, uint32_t pos = 0, uint32_t count = 0)
{
    if(str[pos] == '\0')
    {
        return ((pos == 0) ? 0 : count + 1);
    }
    else
    {
        return ((str[pos] == ',') ? stringargs_count(str, pos + 1, count + 1)
                                  : stringargs_count(str, pos + 1, count));
    }
}

static constexpr uint32_t stringargs_trim(const char* str, uint32_t pos)
{
    if(str[pos] == '\0' || str[pos] == ' ' || str[pos] == '\t' || str[pos] == ',')
    {
        return pos;
    }
    else
    {
        return stringargs_trim(str, pos + 1);
    }
}

constexpr uint32_t
    stringargs_to_lst(char str[], uint32_t pos, const char* strlst[], uint32_t strlst_pos)
{
    if(str[pos] == '\0')
    {
        return pos;
    }
    else
    {
        if(str[pos] == ' ' || str[pos] == '\t' || str[pos] == ',')
        {
            str[pos] = '\0';
            return stringargs_to_lst(str, pos + 1, strlst, strlst_pos);
        }
        else
        {
            strlst[strlst_pos] = &str[pos];
            pos                = stringargs_trim(str, pos);
            return stringargs_to_lst(str, pos, strlst, strlst_pos + 1);
        }
    }
}

#define LIST_ARG_STRINGS(...)                                                      \
    char                      stringargs[] = #__VA_ARGS__;                         \
    static constexpr uint32_t stringargs_c = stringargs_count(#__VA_ARGS__, 0, 0); \
    const char*               stringargs_lst[stringargs_c];                        \
    stringargs_to_lst(stringargs, 0, stringargs_lst, 0)

struct rocgraph_local_argdescr
{
private:
    void* argdescr{};

public:
    rocgraph_local_argdescr()
    {
        if(rocgraph_clients_envariables::get(rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS))
        {
            rocgraph_argdescr_create(&argdescr);
        }
    }

    ~rocgraph_local_argdescr()
    {
        rocgraph_argdescr_free(argdescr);
    }

    // Allow rocgraph_local_mat_descr to be used anywhere rocgraph_mat_descr is expected
    operator void*&()
    {
        return this->argdescr;
    }
    // clang-format off
  operator void* const &() const
  {
    return this->argdescr;
  }
    // clang-format on
};

template <typename F, typename... Ts>
inline void auto_testing_bad_arg_excluding(F f, int n, const int* idx, const char** names, Ts... ts)
{
    //
    // Tell we are passing here to summarize routines that are not.
    //
    test_check::set_auto_testing_bad_arg();

    //
    // Create argument descriptpr.
    //
    rocgraph_local_argdescr argdescr;

    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        bool exclude = false;
        for(int i = 0; i < n; ++i)
        {
            if(idx[i] == iarg)
            {
                exclude = true;
                break;
            }
        }

        if(!exclude)
        {
            //
            //
            //
            auto_testing_bad_arg_t<Ts...> arguments(ts...);

            //
            //
            //
            rocgraph_status               status = rocgraph_status_success;
            auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);

            //
            //
            //
            auto_testing_bad_arg_copy(invalid_data, ts...);

            //
            //
            //
            const rocgraph_status status_from_routine = f(ts...);
            if(rocgraph_clients_envariables::get(
                   rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS))
            {
                //
                // Get the argument name.
                //
                const char* argname;
                CHECK_ROCGRAPH_SUCCESS(rocgraph_argdescr_get_name(argdescr, &argname));

                //
                // If names do not fit.
                //
                const int cmp = strcmp(argname, names[iarg]);
                if(cmp)
                {
                    std::cout
                        << "auto testing bad arg failed on " //
                        << iarg //
                        << " 'th argument, '" //
                        << names[iarg] //
                        << "'" //
                        << std::endl //
                        << "   reason: argument names do not match, argument checking returns " //
                        << argname //
                        << std::endl;
#ifdef GOOGLE_TEST
                    EXPECT_EQ(cmp, 0);
#endif
                }

                //
                // Get the argument index.
                //
                int argidx;
                CHECK_ROCGRAPH_SUCCESS(rocgraph_argdescr_get_index(argdescr, &argidx));
                //
                // If argument indices do not fit.
                //
                if(argidx != iarg)
                {
                    std::cout
                        << "auto testing bad arg failed on " //
                        << iarg //
                        << " 'th argument, '" //
                        << names[iarg] //
                        << "'" //
                        << std::endl //
                        << "   reason: argument indices do not match, argument checking returns " //
                        << argidx //
                        << std::endl;
#ifdef GOOGLE_TEST
                    EXPECT_EQ(argidx, iarg);
#endif
                }
            }

            //
            // if statuses do not fit.
            //
            if(status != status_from_routine)
            {
                std::cout << "auto testing bad arg failed on " //
                          << iarg //
                          << " 'th argument, '" //
                          << ((names != nullptr) ? names[iarg] : "") //
                          << "'" //
                          << std::endl
                          << "   reason: statuses do not match, argument checking returns "
                          << status_from_routine << ", but it should return " << status
                          << std::endl;
                auto_testing_bad_arg_print(ts...);
                CHECK_ROCGRAPH_STATUS(status_from_routine, status);
            }

            //
            //
            //
            auto_testing_bad_arg_copy(arguments, ts...);
        }
    }
}

template <typename F, typename... Ts>
inline void auto_testing_bad_arg(F f, Ts... ts)
{
    //
    // Tell we are passing here to summarize routines that are not.
    //
    test_check::set_auto_testing_bad_arg();
    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        auto_testing_bad_arg_t<Ts...> arguments(ts...);

        {
            rocgraph_status               status;
            auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);
            auto_testing_bad_arg_copy(invalid_data, ts...);

            if(status != f(ts...))
            {
                std::cout << "auto testing bad arg failed on " << iarg << " 'th argument"
                          << std::endl;
                auto_testing_bad_arg_print(ts...);
                CHECK_ROCGRAPH_STATUS(f(ts...), status);
            }
        }

        auto_testing_bad_arg_copy(arguments, ts...);
    }
}

template <typename F, typename... Ts>
inline void auto_testing_bad_arg(F f, int n, const int* idx, Ts... ts)
{
    //
    // Tell we are passing here to summarize routines that are not.
    //
    test_check::set_auto_testing_bad_arg();
    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        bool exclude = false;
        for(int i = 0; i < n; ++i)
        {
            if(idx[i] == iarg)
            {
                exclude = true;
                break;
            }
        }

        if(!exclude)
        {
            auto_testing_bad_arg_t<Ts...> arguments(ts...);

            {
                rocgraph_status               status = rocgraph_status_success;
                auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);
                auto_testing_bad_arg_copy(invalid_data, ts...);

                if(status != f(ts...))
                {
                    std::cout << "auto testing bad arg failed on " << iarg << " 'th argument"
                              << std::endl;
                    auto_testing_bad_arg_print(ts...);
                    CHECK_ROCGRAPH_STATUS(f(ts...), status);
                }
            }

            auto_testing_bad_arg_copy(arguments, ts...);
        }
    }
}

#define bad_arg_analysis(f, ...)                                                                  \
    do                                                                                            \
    {                                                                                             \
        if(rocgraph_clients_envariables::get(rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS)) \
        {                                                                                         \
            LIST_ARG_STRINGS(__VA_ARGS__);                                                        \
            auto_testing_bad_arg_excluding(f, 0, nullptr, stringargs_lst, __VA_ARGS__);           \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            auto_testing_bad_arg(f, __VA_ARGS__);                                                 \
        }                                                                                         \
    } while(false)

#define select_bad_arg_analysis(f, n, idx, ...)                                                   \
    do                                                                                            \
    {                                                                                             \
        if(rocgraph_clients_envariables::get(rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS)) \
        {                                                                                         \
            LIST_ARG_STRINGS(__VA_ARGS__);                                                        \
            auto_testing_bad_arg_excluding(f, n, idx, stringargs_lst, __VA_ARGS__);               \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            auto_testing_bad_arg(f, n, idx, __VA_ARGS__);                                         \
        }                                                                                         \
    } while(false)
