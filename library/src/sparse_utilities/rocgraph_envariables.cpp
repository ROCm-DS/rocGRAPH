// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "envariables.h"
#include <iostream>
//
//
//
template <typename T>
static bool rocgraph_getenv(const char* name, T& val);
template <>
bool rocgraph_getenv<bool>(const char* name, bool& val)
{
    val                    = false;
    const char* getenv_str = getenv(name);
    if(getenv_str != nullptr)
    {
        auto getenv_int = atoi(getenv_str);
        if((getenv_int != 0) && (getenv_int != 1))
        {
            std::cerr << "rocgraph error, invalid environment variable " << name
                      << " must be 0 or 1." << std::endl;
            val = false;
            return false;
        }
        else
        {
            val = (getenv_int == 1);
            return true;
        }
    }
    return true;
}

constexpr rocgraph::envariables::bool_var rocgraph::envariables::all[];

rocgraph::envariables& rocgraph::envariables::Instance()
{
    static rocgraph::envariables instance;
    return instance;
}

rocgraph::envariables::envariables()
{
    //
    // Query variables.
    //
    for(auto tag : rocgraph::envariables::all)
    {
        switch(tag)
        {
#define ENVARIABLE(x_)                                                                       \
    case rocgraph::envariables::x_:                                                          \
    {                                                                                        \
        auto success                                                                         \
            = rocgraph_getenv("ROCGRAPH_" #x_, this->m_bool_var[rocgraph::envariables::x_]); \
        if(!success)                                                                         \
        {                                                                                    \
            std::cerr << "rocgraph_getenv failed " << std::endl;                             \
            exit(1);                                                                         \
        }                                                                                    \
        break;                                                                               \
    }

            ROCGRAPH_FOREACH_ENVARIABLES;

#undef ENVARIABLE
        }
    }

    if(this->m_bool_var[rocgraph::envariables::VERBOSE])
    {
        for(auto tag : rocgraph::envariables::all)
        {
            switch(tag)
            {
#define ENVARIABLE(x_)                                                                       \
    case rocgraph::envariables::x_:                                                          \
    {                                                                                        \
        const bool v = this->m_bool_var[rocgraph::envariables::x_];                          \
        std::cout << ""                                                                      \
                  << "env variable ROCGRAPH_" #x_ << " : " << ((v) ? "enabled" : "disabled") \
                  << std::endl;                                                              \
        break;                                                                               \
    }

                ROCGRAPH_FOREACH_ENVARIABLES;

#undef ENVARIABLE
            }
        }
    }
}
