/*! \file */

/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "envariables.h"
#include "internal/types/rocgraph_status.h"

namespace rocgraph
{
    ///
    /// @brief Structure to store debug global variables.
    ///
    struct debug_variables_st
    {
    private:
        bool debug;
        bool debug_arguments;
        bool debug_verbose;
        bool debug_arguments_verbose;
        bool debug_kernel_launch;
        bool debug_force_host_assert;

    public:
        bool get_debug() const;
        bool get_debug_verbose() const;
        bool get_debug_kernel_launch() const;
        bool get_debug_arguments() const;
        bool get_debug_arguments_verbose() const;
        bool get_debug_force_host_assert() const;

        void set_debug(bool value);
        void set_debug_verbose(bool value);
        void set_debug_arguments(bool value);
        void set_debug_kernel_launch(bool value);
        void set_debug_arguments_verbose(bool value);
        void set_debug_force_host_assert(bool value);
    };

    struct debug_st
    {
    private:
        debug_variables_st m_var{};

    public:
        static debug_st& instance()
        {
            static debug_st self;
            return self;
        }

        static debug_variables_st& var()
        {
            return instance().m_var;
        }

        ~debug_st() = default;

    private:
        debug_st()
        {
            const bool debug = ROCGRAPH_ENVARIABLES.get(rocgraph::envariables::DEBUG);
            m_var.set_debug(debug);

            const bool debug_arguments
                = (!getenv(rocgraph::envariables::names[rocgraph::envariables::DEBUG_ARGUMENTS]))
                      ? debug
                      : ROCGRAPH_ENVARIABLES.get(rocgraph::envariables::DEBUG);
            m_var.set_debug_arguments(debug_arguments);

            m_var.set_debug_verbose(
                (!getenv(rocgraph::envariables::names[rocgraph::envariables::DEBUG_VERBOSE]))
                    ? debug
                    : ROCGRAPH_ENVARIABLES.get(rocgraph::envariables::DEBUG_VERBOSE));
            m_var.set_debug_arguments_verbose(
                (!getenv(
                    rocgraph::envariables::names[rocgraph::envariables::DEBUG_ARGUMENTS_VERBOSE]))
                    ? debug_arguments
                    : ROCGRAPH_ENVARIABLES.get(rocgraph::envariables::DEBUG_ARGUMENTS_VERBOSE));

            m_var.set_debug_force_host_assert(
                (!getenv(
                    rocgraph::envariables::names[rocgraph::envariables::DEBUG_FORCE_HOST_ASSERT]))
                    ? debug
                    : ROCGRAPH_ENVARIABLES.get(rocgraph::envariables::DEBUG_FORCE_HOST_ASSERT));

            const bool debug_kernel_launch
                = ROCGRAPH_ENVARIABLES.get(rocgraph::envariables::DEBUG_KERNEL_LAUNCH);
            m_var.set_debug_kernel_launch(debug_kernel_launch);
        };
    };

#define rocgraph_debug_variables rocgraph::debug_st::instance().var()
}
