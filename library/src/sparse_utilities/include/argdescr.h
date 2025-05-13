/*! \file */

/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "debug.h"
#include "enum_utils.hpp"

namespace rocgraph
{
    ///
    /// @brief Log in argument error description.
    /// @param function_file_ name of the file this routine is called from.
    /// @param function_name_ name of the routine this routine is called from.
    /// @param function_line_ line of the file this routine is called from.
    /// @param arg_name_ name of the argument of the routine this routine is called from.
    /// @param arg_index_ index of the argument of the routine this routine is called from.
    /// @param status_  returned status of the routine this routine is called from.
    /// @param msg_ index of the argument of the routine this routine is called from.
    ///
    void argdescr_log(const char*     function_file_,
                      const char*     function_name_,
                      int             function_line_,
                      const char*     arg_name_,
                      int             arg_index_,
                      rocgraph_status status_,
                      const char*     msg_);
}

///
/// @brief Check the argument array, i.e. a non null pointer for a positive size.
/// @param ITH__ index of the argument.
/// @param SIZE__ size of the array.
/// @param POINTER__ pointer of the array.
///
#define ROCGRAPH_CHECKARG_ARRAY(ITH__, SIZE__, POINTER__)                                     \
    do                                                                                        \
    {                                                                                         \
        if((SIZE__) > 0 && (POINTER__) == nullptr)                                            \
        {                                                                                     \
            if(rocgraph_debug_variables.get_debug_arguments())                                \
            {                                                                                 \
                std::stringstream s;                                                          \
                s << "array pointer is " #POINTER__ " null with ('" #SIZE__ " = " << (SIZE__) \
                  << "' > 0)";                                                                \
                rocgraph::argdescr_log(__FILE__,                                              \
                                       __FUNCTION__,                                          \
                                       __LINE__,                                              \
                                       #POINTER__,                                            \
                                       (ITH__),                                               \
                                       rocgraph_status_invalid_pointer,                       \
                                       s.str().c_str());                                      \
            }                                                                                 \
            return rocgraph_status_invalid_pointer;                                           \
        }                                                                                     \
    } while(false)

///
/// @brief Check the argument size, i.e. a non negative size.
/// @param ITH__ index of the argument.
/// @param SIZE__ size.
///
#define ROCGRAPH_CHECKARG_SIZE(ITH__, SIZE__)                        \
    do                                                               \
    {                                                                \
        if((SIZE__) < 0)                                             \
        {                                                            \
            if(rocgraph_debug_variables.get_debug_arguments())       \
                rocgraph::argdescr_log(__FILE__,                     \
                                       __FUNCTION__,                 \
                                       __LINE__,                     \
                                       #SIZE__,                      \
                                       (ITH__),                      \
                                       rocgraph_status_invalid_size, \
                                       "size is negative.");         \
            return rocgraph_status_invalid_size;                     \
        }                                                            \
    } while(false)

///
/// @brief Check the argument enum, i.e. a valid value.
/// @param ITH__ index of the argument.
/// @param ENUM__ The enum value.
///
#define ROCGRAPH_CHECKARG_ENUM(ITH__, ENUM__)                         \
    do                                                                \
    {                                                                 \
        if(rocgraph::enum_utils::is_invalid((ENUM__)))                \
        {                                                             \
            if(rocgraph_debug_variables.get_debug_arguments())        \
                rocgraph::argdescr_log(__FILE__,                      \
                                       __FUNCTION__,                  \
                                       __LINE__,                      \
                                       #ENUM__,                       \
                                       (ITH__),                       \
                                       rocgraph_status_invalid_value, \
                                       "enum has an invalid value."); \
            return rocgraph_status_invalid_value;                     \
        }                                                             \
    } while(false)

///
/// @brief Check the argument handle, i.e. a non null pointer.
/// @param ITH__ index of the argument.
/// @param HANDLE__ The handle.
///
#define ROCGRAPH_CHECKARG_HANDLE(ITH__, HANDLE__)                      \
    do                                                                 \
    {                                                                  \
        if((HANDLE__) == nullptr)                                      \
        {                                                              \
            if(rocgraph_debug_variables.get_debug_arguments())         \
                rocgraph::argdescr_log(__FILE__,                       \
                                       __FUNCTION__,                   \
                                       __LINE__,                       \
                                       #HANDLE__,                      \
                                       (ITH__),                        \
                                       rocgraph_status_invalid_handle, \
                                       "handle is null.");             \
            return rocgraph_status_invalid_handle;                     \
        }                                                              \
    } while(false)

///
/// @brief Check the argument pointer, i.e. a non null pointer.
/// @param ITH__ index of the argument.
/// @param POINTER__ The pointer.
///
#define ROCGRAPH_CHECKARG_POINTER(ITH__, POINTER__)                     \
    do                                                                  \
    {                                                                   \
        if((POINTER__) == nullptr)                                      \
        {                                                               \
            if(rocgraph_debug_variables.get_debug_arguments())          \
                rocgraph::argdescr_log(__FILE__,                        \
                                       __FUNCTION__,                    \
                                       __LINE__,                        \
                                       #POINTER__,                      \
                                       (ITH__),                         \
                                       rocgraph_status_invalid_pointer, \
                                       "pointer is null.");             \
            return rocgraph_status_invalid_pointer;                     \
        }                                                               \
    } while(false)

///
/// @brief Check the argument pointer, i.e. a non null pointer.
/// @param ITH__ index of the argument.
/// @param ARG__ The argument.
/// @param CONDITION__ condition to log.
/// @param STATUS__ status to return.
///
#define ROCGRAPH_CHECKARG(ITH__, ARG__, CONDITION__, STATUS__)                    \
    do                                                                            \
    {                                                                             \
        if((CONDITION__))                                                         \
        {                                                                         \
            if(rocgraph_debug_variables.get_debug_arguments())                    \
                rocgraph::argdescr_log(__FILE__,                                  \
                                       __FUNCTION__,                              \
                                       __LINE__,                                  \
                                       #ARG__,                                    \
                                       (ITH__),                                   \
                                       (STATUS__),                                \
                                       "failed on condition '" #CONDITION__ "'"); \
            return (STATUS__);                                                    \
        }                                                                         \
    } while(false)
