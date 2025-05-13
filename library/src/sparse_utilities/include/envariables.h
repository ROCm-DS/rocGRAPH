/*! \file */

/*
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace rocgraph
{
    template <std::size_t N, typename T>
    inline constexpr std::size_t countof(T (&)[N])
    {
        return N;
    }

    //
    // Definition of a utility class to grab environment variables
    // for the rocgraph library.
    //
    // The corresponding environment variable is the literal enum string
    // with the prefix ROCGRAPH_.
    // Example: envariables::VERBOSE will have a one to one correspondance with the environment variable
    // ROCGRAPH_VERBOSE.
    // Obviously it loads environment variables at run time.
    //
    class envariables
    {
    public:
#define ROCGRAPH_FOREACH_ENVARIABLES    \
    ENVARIABLE(DEBUG)                   \
    ENVARIABLE(DEBUG_ARGUMENTS)         \
    ENVARIABLE(DEBUG_ARGUMENTS_VERBOSE) \
    ENVARIABLE(DEBUG_KERNEL_LAUNCH)     \
    ENVARIABLE(DEBUG_VERBOSE)           \
    ENVARIABLE(VERBOSE)                 \
    ENVARIABLE(MEMSTAT)                 \
    ENVARIABLE(MEMSTAT_FORCE_MANAGED)   \
    ENVARIABLE(DEBUG_FORCE_HOST_ASSERT) \
    ENVARIABLE(MEMSTAT_GUARDS)

        //
        // Specification of the enum and the array of all values.
        //
#define ENVARIABLE(x_) x_,

        typedef enum bool_var_ : int32_t
        {
            ROCGRAPH_FOREACH_ENVARIABLES
        } bool_var;
        static constexpr bool_var all[] = {ROCGRAPH_FOREACH_ENVARIABLES};

#undef ENVARIABLE

        //
        // Specification of names.
        //
#define ENVARIABLE(x_) "ROCGRAPH_" #x_,
        static constexpr const char* names[] = {ROCGRAPH_FOREACH_ENVARIABLES};
#undef ENVARIABLE

        //
        // Number of values.
        //
        static constexpr size_t size          = countof(all);
        static constexpr size_t bool_var_size = size;

        //
        // \brief Return value of a Boolean variable.
        //
        inline bool get(bool_var v) const
        {
            return this->m_bool_var[v];
        };

        //
        // Return the unique instance.
        //
        static envariables& Instance();

    private:
        envariables();
        ~envariables()                             = default;
        envariables(const envariables&)            = delete;
        envariables& operator=(const envariables&) = delete;
        bool         m_bool_var[bool_var_size]{};
    };
}

#define ROCGRAPH_ENVARIABLES rocgraph::envariables::Instance()
