/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#ifndef TEST_CLEANUP_HPP
#define TEST_CLEANUP_HPP

#include <functional>
#include <stack>

/*! \brief  Test cleanup handler. Frees memory or performs other cleanup at
specified points in program. */

class test_cleanup
{
    static auto& stack()
    {
        // Placed inside function to avoid dependency on initialization order
        static std::stack<std::function<void()>> stack;
        return stack;
    }

public:
    // Run all cleanup handlers pushed so far, in LIFO order
    static void cleanup()
    {
        while(!stack().empty())
        {
            stack().top()();
            stack().pop();
        }
    }

    // Create an object and register a cleanup handler
    template <typename T, typename... Args>
    static T* allocate(T** ptr, Args&&... args)
    {
        *ptr = nullptr;
        stack().push([=] {
            delete *ptr;
            *ptr = nullptr;
        });
        return new T(std::forward<Args>(args)...);
    }
};

#endif // TEST_CLEANUP_HPP
