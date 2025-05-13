/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_parse_data.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

#include "test_check.hpp"
bool test_check::s_auto_testing_bad_arg;

bool display_timing_info_is_stdout_disabled()
{
    return false;
}

rocgraph_status rocgraph_record_output(const std::string& s)
{
    return rocgraph_status_success;
}

rocgraph_status rocgraph_record_output_legend(const std::string& s)
{
    return rocgraph_status_success;
}

rocgraph_status rocgraph_record_timing(double msec, double gflops, double gbs)
{
    return rocgraph_status_success;
}

class ConfigurableEventListener : public testing::TestEventListener
{
    testing::TestEventListener* eventListener;

public:
    bool showTestCases; // Show the names of each test case.
    bool showTestNames; // Show the names of each test.
    bool showSuccesses; // Show each success.
    bool showInlineFailures; // Show each failure as it occurs.
    bool showEnvironment; // Show the setup of the global environment.

    explicit ConfigurableEventListener(testing::TestEventListener* theEventListener)
        : eventListener(theEventListener)
        , showTestCases(true)
        , showTestNames(true)
        , showSuccesses(true)
        , showInlineFailures(true)
        , showEnvironment(true)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const testing::UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    void OnTestIterationStart(const testing::UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const testing::UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpStart(unit_test);
        }
    }

    void OnEnvironmentsSetUpEnd(const testing::UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
        }
    }

    void OnTestCaseStart(const testing::TestCase& test_case) override
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseStart(test_case);
        }
    }

    void OnTestStart(const testing::TestInfo& test_info) override
    {
        if(showTestNames)
        {
            eventListener->OnTestStart(test_info);
        }
    }

    void OnTestPartResult(const testing::TestPartResult& result) override
    {
        eventListener->OnTestPartResult(result);
    }

    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
        {
            eventListener->OnTestEnd(test_info);
        }
    }

    void OnTestCaseEnd(const testing::TestCase& test_case) override
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseEnd(test_case);
        }
    }

    void OnEnvironmentsTearDownStart(const testing::UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownStart(unit_test);
        }
    }

    void OnEnvironmentsTearDownEnd(const testing::UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
        }
    }

    void OnTestIterationEnd(const testing::UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const testing::UnitTest& unit_test) override
    {
        eventListener->OnTestProgramEnd(unit_test);
    }
};

/* =====================================================================
      Main function:
=================================================================== */
static char s_extended_gtest_filter[]   = {"--gtest_filter=*AlgorithmTest*:*PlumbingTest*"};
static char s_regression_gtest_filter[] = {"--gtest_filter=*AlgorithmTest*:*PlumbingTest*"};
static char s_smoke_gtest_filter[]      = {"--gtest_filter=*AlgorithmTest*:*PlumbingTest*"};

int main(int argc, char** argv)
{
    // Get version
    rocgraph_handle handle;
    rocgraph_create_handle(&handle, nullptr);

    int  ver;
    char rev[64];

    rocgraph_get_version(handle, &ver);
    rocgraph_get_git_rev(handle, rev);

    rocgraph_destroy_handle(handle);

    // Get user device id from command line
    int dev = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--device") == 0 && argc > i + 1)
        {
            dev = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "--version") == 0)
        {
            // Print version and exit, if requested
            std::cout << "rocGRAPH version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
                      << ver % 100 << "-" << rev << std::endl;

            return 0;
        }
    }

    //
    // Switch few options with other options.
    //
    int i;
    for(i = 1; i < argc; ++i)
    {
        if(!strcmp(argv[i], "--smoke-test"))
        {
            for(int i = 1; i < argc; ++i)
            {
                if(!strncmp(argv[i], "--gtest_filter", 14))
                {
                    std::cerr << "--gtest_filter is incompatible with --smoke-test" << std::endl;
                    return rocgraph_status_invalid_value;
                }
            }
            argv[i] = s_smoke_gtest_filter;
            break;
        }
    }

    if(i >= argc)
    {
        for(i = 1; i < argc; ++i)
        {
            if(!strcmp(argv[i], "--regression-test"))
            {
                for(int i = 1; i < argc; ++i)
                {
                    if(!strncmp(argv[i], "--gtest_filter", 14))
                    {
                        std::cerr << "--gtest_filter is incompatible with --regression-test"
                                  << std::endl;
                        return rocgraph_status_invalid_value;
                    }
                }
                argv[i] = s_regression_gtest_filter;
                break;
            }
        }
    }
    if(i >= argc)
    {
        for(i = 1; i < argc; ++i)
        {
            if(!strcmp(argv[i], "--extended-test"))
            {
                for(int i = 1; i < argc; ++i)
                {
                    if(!strncmp(argv[i], "--gtest_filter", 14))
                    {
                        std::cerr << "--gtest_filter is incompatible with --extended-test"
                                  << std::endl;
                        return rocgraph_status_invalid_value;
                    }
                }
                argv[i] = s_extended_gtest_filter;
                break;
            }
        }
    }

    // Device query
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "Error: cannot get device count" << std::endl;
        return -1;
    }

    std::cout << "Query device success: there are " << devs << " devices" << std::endl;

    for(int i = 0; i < devs; ++i)
    {
        hipDeviceProp_t prop;

        if(hipGetDeviceProperties(&prop, i) != hipSuccess)
        {
            std::cerr << "Error: cannot get device properties" << std::endl;
            return -1;
        }

        std::cout << "Device ID " << i << ": " << prop.name << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
        std::cout << "with " << (prop.totalGlobalMem >> 20) << "MB memory, clock rate "
                  << prop.clockRate / 1000 << "MHz @ computing capability " << prop.major << "."
                  << prop.minor << std::endl;
        std::cout << "maxGridDimX " << prop.maxGridSize[0] << ", sharedMemPerBlock "
                  << (prop.sharedMemPerBlock >> 10) << "KB, maxThreadsPerBlock "
                  << prop.maxThreadsPerBlock << std::endl;
        std::cout << "wavefrontSize " << prop.warpSize << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
    }

    // Set device
    if(hipSetDevice(dev) != hipSuccess || dev >= devs)
    {
        std::cerr << "Error: cannot set device ID " << dev << std::endl;
        return -1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);

    std::cout << "Using device ID " << dev << " (" << prop.name << ") for rocGRAPH" << std::endl;
    std::cout << "-------------------------------------------------------------------------"
              << std::endl;

    // Print version
    std::cout << "rocGRAPH version: " << ver / 100000 << "." << ver / 100 % 1000 << "." << ver % 100
              << "-" << rev << std::endl;

    std::string datapath = rocgraph_datapath();

    // Print test data path being used
    std::cout << "rocGRAPH data path: " << datapath << std::endl;

    // Set data file path
    rocgraph_parse_data(argc, argv, datapath + "rocgraph_test.data");

    // Initialize google test
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // Remove the default listener
    auto& listeners       = testing::UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // Add our listener, by default everything is on (the same as using the default listener)
    // Here turning everything off so only the 3 lines for the result are visible
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto listener       = new ConfigurableEventListener(default_printer);
    auto gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        listener->showTestNames = listener->showSuccesses = listener->showInlineFailures = false;
    }

    listeners.Append(listener);

    // Run all tests
    int ret = RUN_ALL_TESTS();

    // Reset HIP device
    hipDeviceReset();

    return ret;
}
