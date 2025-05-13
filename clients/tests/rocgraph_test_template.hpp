/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_call.hpp"
#include "rocgraph_test_check.hpp"
#include "rocgraph_test_dispatch.hpp"
#include "rocgraph_test_functors.hpp"
#include "rocgraph_test_traits.hpp"

namespace
{
    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_template
    {
    private:
        using call_t     = rocgraph_test_call<ROUTINE>;
        using traits_t   = rocgraph_test_traits<ROUTINE>;
        using functors_t = rocgraph_test_functors<ROUTINE>;
        using dispatch_t = rocgraph_test_dispatch<traits_t::s_dispatch>;

    public:
        template <typename... P>
        struct test_call_proxy
        {
            explicit operator bool()
            {
                return true;
            }
            void operator()(const Arguments& arg)
            {
                const char* name_ROUTINE = rocgraph_test_enum::to_string(ROUTINE);
                if(!strcmp(arg.function, name_ROUTINE))
                {
                    call_t::template testing<P...>(arg);
                }
                else
                {
                    std::string s(name_ROUTINE);
                    s += "_bad_arg";
                    if(!strcmp(arg.function, s.c_str()))
                    {
                        call_t::template testing_bad_arg<P...>(arg);
                    }
                    else
                    {
                        std::string s1(name_ROUTINE);
                        s1 += "_extra";
                        if(!strcmp(arg.function, s1.c_str()))
                        {
                            call_t::testing_extra(arg);
                        }
                        else
                        {
                            FAIL() << "Internal error: Test called with unknown function: "
                                   << arg.function;
                        }
                    }
                }
            }
        };

        template <typename PROXY, template <typename...> class PROXY_CALL>
        struct test_proxy : RocGRAPH_Test<PROXY, PROXY_CALL>
        {
            using definition = RocGRAPH_Test<PROXY, PROXY_CALL>;
            static bool type_filter(const Arguments& arg)
            {
                return dispatch_t::template dispatch<definition::template type_filter_functor>(arg);
            }

            static bool function_filter(const Arguments& arg)
            {
                const char* name = rocgraph_test_enum::to_string(ROUTINE);
                std::string s(name);
                s += "_bad_arg";
                std::string s1(name);
                s1 += "_extra";
                return !strcmp(arg.function, name) || !strcmp(arg.function, s.c_str())
                       || !strcmp(arg.function, s1.c_str());
            }

            static bool arch_filter(const Arguments& arg)
            {
                static int             dev;
                static hipDeviceProp_t prop;

                static bool query_device = true;
                if(query_device)
                {
                    if(hipGetDevice(&dev) != hipSuccess)
                    {
                        return false;
                    }
                    if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
                    {
                        return false;
                    }
                    query_device = false;
                }

                if(strncmp("gfx", arg.hardware, 3) == 0)
                {
                    if(strncmp(arg.hardware, prop.gcnArchName, strlen(arg.hardware)) == 0)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                if(strncmp("none", arg.skip_hardware, 4) != 0)
                {
                    const char* b = arg.skip_hardware;
                    const char* e;

                    for(e = b; *e != '\0' && *e != ',' && *e != ' '; ++e)
                        ;
                    while(strncmp("gfx", b, 3) == 0)
                    {
                        if(strncmp(b, prop.gcnArchName, e - b) == 0)
                        {
                            return false;
                        }
                        else
                        {
                            for(; *b != '\0' && *b != ',' && *b != ' '; ++b)
                                ;
                            if(*b == ',')
                                ++b;
                            for(; *b != '\0' && *b == ' '; ++b)
                                ;
                            for(e = b; *e != '\0' && *e != ',' && *e != ' '; ++e)
                                ;
                        }
                    }
                }

                return true;
            }

            static bool memory_filter(const Arguments& arg)
            {
                static double available_memory_in_GB = 0.0;
                static bool   query_device_memory    = true;
                if(query_device_memory)
                {
                    size_t available_memory;
                    size_t total_memory;
                    if(hipMemGetInfo(&available_memory, &total_memory) != hipSuccess)
                    {
                        return false;
                    }

                    available_memory_in_GB = (double)available_memory / (1024 * 1024 * 1024);

                    query_device_memory = false;
                }

                if(available_memory_in_GB < arg.req_memory)
                {
                    std::cout << "Skipping test "
                              << (std::string(arg.category) + "/" + std::string(arg.function) + "/"
                                  + name_suffix(arg))
                              << " because insufficient memory avaiable. Required: "
                              << arg.req_memory << "GB. Available: " << available_memory_in_GB
                              << "GB." << std::endl;
                    return false;
                }

                return true;
            }

            static std::string name_suffix(const Arguments& arg)
            {
                //
                // Check if this is extra tests.
                //
                {
                    const char* name = rocgraph_test_enum::to_string(ROUTINE);
                    std::string s1(name);
                    s1 += "_extra";
                    if(!strcmp(arg.function, s1.c_str()))
                    {
                        //
                        // Return the name of the test.
                        //
                        return RocGRAPH_TestName<PROXY>{} << arg.name;
                    }
                }

                const bool         from_file = rocgraph_arguments_has_datafile(arg);
                std::ostringstream s;
                switch(traits_t::s_dispatch)
                {
                case rocgraph_test_dispatch_enum::t:
                {
                    s << rocgraph_datatype2string(arg.compute_type);
                    break;
                }

                case rocgraph_test_dispatch_enum::it:
                case rocgraph_test_dispatch_enum::it_plus_int8:
                {
                    s << rocgraph_indextype2string(arg.index_type_I) << '_'
                      << rocgraph_datatype2string(arg.compute_type);
                    break;
                }
                case rocgraph_test_dispatch_enum::ijt:
                {
                    s << rocgraph_indextype2string(arg.index_type_I) << '_'
                      << rocgraph_indextype2string(arg.index_type_J) << '_'
                      << rocgraph_datatype2string(arg.compute_type);
                    break;
                }
                case rocgraph_test_dispatch_enum::ixyt:
                {
                    s << rocgraph_indextype2string(arg.index_type_I) << '_'
                      << rocgraph_datatype2string(arg.x_type) << '_'
                      << rocgraph_datatype2string(arg.y_type) << '_'
                      << rocgraph_datatype2string(arg.compute_type);
                    break;
                }
                case rocgraph_test_dispatch_enum::iaxyt:
                {
                    s << rocgraph_indextype2string(arg.index_type_I) << '_'
                      << rocgraph_datatype2string(arg.a_type) << '_'
                      << rocgraph_datatype2string(arg.x_type) << '_'
                      << rocgraph_datatype2string(arg.y_type) << '_'
                      << rocgraph_datatype2string(arg.compute_type);
                    break;
                }
                case rocgraph_test_dispatch_enum::ijaxyt:
                {
                    s << rocgraph_indextype2string(arg.index_type_I) << '_'
                      << rocgraph_indextype2string(arg.index_type_J) << '_'
                      << rocgraph_datatype2string(arg.a_type) << '_'
                      << rocgraph_datatype2string(arg.x_type) << '_'
                      << rocgraph_datatype2string(arg.y_type) << '_'
                      << rocgraph_datatype2string(arg.compute_type);
                    break;
                }
                }

                //
                // Check if this is bad_arg
                //
                {
                    const char* name = rocgraph_test_enum::to_string(ROUTINE);
                    std::string s1(name);
                    s1 += "_bad_arg";
                    if(!strcmp(arg.function, s1.c_str()))
                    {
                        s << "_bad_arg";
                    }
                    else
                    {
                        const std::string suffix = functors_t::name_suffix(arg);
                        if(suffix.size() > 0)
                        {
                            s << '_' << suffix;
                        }

                        if(from_file)
                        {
                            s << '_' << rocgraph_filename2string(arg.filename);
                        }
                    }
                }

                return RocGRAPH_TestName<PROXY>{} << s.str();
            }
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_ixyt_template
    {
        template <typename X, typename Y, typename T, typename I, typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        template <typename I, typename X, typename Y, typename T>
        struct test_call<I, X, Y, T, typename std::enable_if<std::is_integral<I>::value>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<I, X, Y, T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_iaxyt_template
    {
        template <typename A, typename X, typename Y, typename T, typename I, typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        template <typename I, typename A, typename X, typename Y, typename T>
        struct test_call<I, A, X, Y, T, typename std::enable_if<std::is_integral<I>::value>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<I, A, X, Y, T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_ijaxyt_template
    {
        template <typename A,
                  typename X,
                  typename Y,
                  typename T,
                  typename I,
                  typename J,
                  typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        template <typename I, typename J, typename A, typename X, typename Y, typename T>
        struct test_call<I,
                         J,
                         A,
                         X,
                         Y,
                         T,
                         typename std::enable_if<std::is_integral<I>::value>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<I, J, A, X, Y, T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_ijt_template
    {
        using check_t = rocgraph_test_check<ROUTINE>;

        //
        template <typename T, typename I = int32_t, typename J = int32_t, typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        //
        template <typename I, typename J, typename T>
        struct test_call<I,
                         J,
                         T,
                         typename std::enable_if<check_t::template is_type_valid<I, J, T>()>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<I, J, T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_it_plus_int8_template
    {
        using check_t = rocgraph_test_check<ROUTINE>;
        //
        template <typename T, typename I = int32_t, typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        //
        template <typename I, typename T>
        struct test_call<I,
                         T,
                         typename std::enable_if<check_t::template is_type_valid<I, T>()>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<I, T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_it_template
    {
        using check_t = rocgraph_test_check<ROUTINE>;
        //
        template <typename T, typename I = int32_t, typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        //
        template <typename I, typename T>
        struct test_call<I,
                         T,
                         typename std::enable_if<check_t::template is_type_valid<I, T>()>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<I, T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <rocgraph_test_enum::value_type ROUTINE>
    struct rocgraph_test_t_template
    {
        using check_t = rocgraph_test_check<ROUTINE>;
        template <typename T, typename = void>
        struct test_call : rocgraph_test_invalid
        {
        };

        //
        template <typename T>
        struct test_call<T, typename std::enable_if<check_t::template is_type_valid<T>()>::type>
            : rocgraph_test_template<ROUTINE>::template test_call_proxy<T>
        {
        };

        struct test : rocgraph_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };
}
