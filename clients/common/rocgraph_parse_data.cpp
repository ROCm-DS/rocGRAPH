/*! \file */

// Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_parse_data.hpp"
#include "rocgraph_clients_matrices_dir.hpp"
#include "rocgraph_data.hpp"
#include "utility.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rocgraph_clients_envariables.hpp"

// Parse YAML data
static std::string rocgraph_parse_yaml(const std::string& yaml)
{
    char tmp[] = "/tmp/rocgraph-XXXXXX";
    int  fd    = mkostemp(tmp, O_CLOEXEC);
    if(fd == -1)
    {
        perror("Cannot open temporary file");
        exit(EXIT_FAILURE);
    }
    auto exepath = rocgraph_exepath();
    auto cmd = exepath + "rocgraph_gentest.py --template " + exepath + "rocgraph_template.yaml -o "
               + tmp + " " + yaml;
    std::cerr << cmd << std::endl;
    int status = system(cmd.c_str());
    if(status == -1 || !WIFEXITED(status) || WEXITSTATUS(status))
        exit(EXIT_FAILURE);
    return tmp;
}

// Parse --data and --yaml command-line arguments, -- matrices-dir and optionally memstat-report
bool rocgraph_parse_data(int& argc, char** argv, const std::string& default_file)
{
    std::string filename;
    char**      argv_p = argv + 1;
    bool        help = false, yaml = false;
#ifdef ROCGRAPH_WITH_MEMSTAT
    const char* memory_report_filename = nullptr;
#endif
    // Scan, process and remove any --yaml or --data options
    for(int i = 1; argv[i]; ++i)
    {
        if(!strcmp(argv[i], "--data") || (yaml |= !strcmp(argv[i], "--yaml")))
        {
            if(filename != "")
            {
                std::cerr << "Only one of the --yaml and --data options may be specified"
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            filename = argv[++i];
        }
#ifdef ROCGRAPH_WITH_MEMSTAT
        else if(!strcmp(argv[i], "--memstat-report"))
        {
            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            memory_report_filename = argv[++i];
        }
#endif
        else if(!strcmp(argv[i], "--matrices-dir"))
        {
            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            rocgraph_clients_matrices_dir_set(argv[++i]);
        }
        else if(!strcmp(argv[i], "--rocgraph-clients-enable-test-debug-arguments"))
        {
            rocgraph_clients_envariables::set(rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS,
                                              true);
        }
        else if(!strcmp(argv[i], "--rocgraph-clients-disable-test-debug-arguments"))
        {
            rocgraph_clients_envariables::set(rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS,
                                              false);
        }
        else if(!strcmp(argv[i], "--rocgraph-disable-debug"))
        {
            rocgraph_disable_debug();
        }
        else if(!strcmp(argv[i], "--rocgraph-enable-debug"))
        {
            rocgraph_enable_debug();
        }
        else if(!strcmp(argv[i], "--rocgraph-enable-debug-verbose"))
        {
            rocgraph_enable_debug_verbose();
        }
        else if(!strcmp(argv[i], "--rocgraph-disable-debug-verbose"))
        {
            rocgraph_disable_debug_verbose();
        }
        else if(!strcmp(argv[i], "--rocgraph-enable-debug-arguments"))
        {
            rocgraph_enable_debug_arguments();
        }
        else if(!strcmp(argv[i], "--rocgraph-disable-debug-arguments"))
        {
            rocgraph_disable_debug_arguments();
        }
        else if(!strcmp(argv[i], "--rocgraph-enable-debug-arguments-verbose"))
        {
            rocgraph_enable_debug_arguments_verbose();
        }
        else if(!strcmp(argv[i], "--rocgraph-disable-debug-arguments-verbose"))
        {
            rocgraph_disable_debug_arguments_verbose();
        }
        else
        {
            *argv_p++ = argv[i];
            if(!help && (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")))
            {
                help = true;
                std::cout
                    << "\n"
                    << argv[0]
                    << " [ --data <path> | --yaml <path> ] [--matrices-dir <path>] <options> ...\n"
                    << std::endl;

                std::cout << "" << std::endl;
                std::cout << "Rocgraph clients debug options:" << std::endl;
                std::cout << "--rocgraph-clients-enable-test-debug-arguments   enable rocgraph "
                             "clients test debug arguments, it discards any environment variable "
                             "definition of ROCGRAPH_CLIENTS_TEST_DEBUG_ARGUMENTS."
                          << std::endl;
                std::cout << "--rocgraph-clients-disable-test-debug-arguments  disable rocgraph "
                             "clients test debug arguments, it discards any environment variable "
                             "definition of ROCGRAPH_CLIENTS_TEST_DEBUG_ARGUMENTS."
                          << std::endl;
                std::cout << "" << std::endl;
                std::cout << "Rocgraph debug options:" << std::endl;
                std::cout << "--rocgraph-enable-debug                     enable rocgraph debug, "
                             "it discards any environment variable definition of ROCGRAPH_DEBUG."
                          << std::endl;
                std::cout
                    << "--rocgraph-disable-debug                    disable rocgraph debug, it "
                       "discards any environment variable definition of ROCGRAPH_DEBUG."
                    << std::endl;
                std::cout << "--rocgraph-enable-debug-verbose             enable rocgraph debug "
                             "verbose, it discards any environment variable definition of "
                             "ROCGRAPH_DEBUG_VERBOSE."
                          << std::endl;
                std::cout << "--rocgraph-disable-debug-verbose            disable rocgraph debug "
                             "verbose, it discards any environment variable definition of "
                             "ROCGRAPH_DEBUG_VERBOSE"
                          << std::endl;
                std::cout << "--rocgraph-enable-debug-arguments           enable rocgraph debug "
                             "arguments, it discards any environment variable definition of "
                             "ROCGRAPH_DEBUG_ARGUMENTS."
                          << std::endl;
                std::cout << "--rocgraph-disable-debug-arguments          disable rocgraph debug "
                             "arguments, it discards any environment variable definition of "
                             "ROCGRAPH_DEBUG_ARGUMENTS."
                          << std::endl;
                std::cout << "--rocgraph-enable-debug-arguments-verbose   enable rocgraph debug "
                             "arguments verbose, it discards any environment variable definition "
                             "of ROCGRAPH_DEBUG_ARGUMENTS_VERBOSE"
                          << std::endl;
                std::cout << "--rocgraph-disable-debug-arguments-verbose  disable rocgraph debug "
                             "arguments verbose, it discards any environment variable definition "
                             "of ROCGRAPH_DEBUG_ARGUMENTS_VERBOSE"
                          << std::endl
                          << std::endl;
                std::cout << "Shortcuts for specific suite of tests" << std::endl;
                std::cout
                    << "--extended-test      triggers emulation smoke tests (exclusive to the "
                       "following options: --gtest_filter, --regression-test and --smoke-test"
                    << std::endl;
                std::cout
                    << "--regression-test      riggers emulation smoke tests (exclusive to the "
                       "following options: --gtest_filter, --extended-test and --smoke-test"
                    << std::endl;
                std::cout
                    << "--smoke-test      triggers emulation smoke tests (exclusive to the "
                       "following options: --gtest_filter, --extended-test and --regression-test"
                    << std::endl;
                std::cout << std::endl << std::endl;
                std::cout << "Specific environment variables:" << std::endl;
                for(const auto v : rocgraph_clients_envariables::s_var_bool_all)
                {
                    std::cout << rocgraph_clients_envariables::get_name(v) << " "
                              << rocgraph_clients_envariables::get_description(v) << std::endl;
                }
                for(const auto v : rocgraph_clients_envariables::s_var_string_all)
                {
                    std::cout << rocgraph_clients_envariables::get_name(v) << " "
                              << rocgraph_clients_envariables::get_description(v) << std::endl;
                }
                std::cout << "" << std::endl;
            }
        }
    }

#ifdef ROCGRAPH_WITH_MEMSTAT
    rocgraph_status status = rocgraph_memstat_report(
        memory_report_filename ? memory_report_filename : "rocgraph_test_memstat.json");
    if(status != rocgraph_status_success)
    {
        std::cerr << "rocgraph_memstat_report failed " << std::endl;
        exit(EXIT_FAILURE);
    }
#endif

    // argc and argv contain remaining options and non-option arguments
    *argv_p = nullptr;
    argc    = argv_p - argv;

    if(filename == "-")
        filename = "/dev/stdin";
    else if(filename == "")
        filename = default_file;

    if(yaml)
        filename = rocgraph_parse_yaml(filename);

    if(filename != "")
    {
        RocGRAPH_TestData::set_filename(filename, yaml);
        return true;
    }

    return false;
}
