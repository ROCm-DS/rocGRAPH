/*! \file */

// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph-export.h"

#include "control.h"
#include "envariables.h"
#include <map>

#include "debug.h"
#include "to_string.hpp"

namespace rocgraph
{
    int64_t get_pid()
    {
        return 0;
    }

    int64_t get_tid()
    {
        return 0;
    }

    ///
    /// @brief The rocgraph argument descriptor.
    ///
    struct argdescr_st
    {
        ///
        /// @brief Thread id.
        ///
        int64_t m_tid{};
        ///

        /// @brief Process id.
        ///
        int64_t m_pid{};

        ///
        /// @brief Line in the file.
        ///
        int m_function_line{};

        ///
        /// @brief 0-based index of the argument.
        ///
        int m_arg_index{};

        ///
        /// @brief Resulting status of the argument checking.
        ///
        rocgraph_status m_status{};

        ///
        /// @brief Function name.
        ///
        char m_function_name[128]{};

        ///
        /// @brief Function name.
        ///
        char m_function_filename[128]{};

        ///
        /// @brief Argument name.
        ///
        char m_arg_name[128]{};

        ///
        /// @brief Message.
        ///
        char m_msg[256]{};

        ///
        /// @brief Define..
        /// @param function_name_ Name of the function.
        /// @param function_line_ Line number in the file where the argument checking is performed.
        /// @param arg_name_ Name of the function argument.
        /// @param arg_index_ 0-based index of the function argument.
        /// @param status_ status of the argument checking.
        /// @param msg_ message of the argument checking.
        ///
        void define(const char*     function_filename_,
                    const char*     function_name_,
                    int             function_line_,
                    const char*     arg_name_,
                    int             arg_index_,
                    rocgraph_status status_,
                    const char*     msg_);

        argdescr_st()  = default;
        ~argdescr_st() = default;
    };

    struct argdescr
    {
    private:
        std::map<int64_t, argdescr_st> map;

    protected:
        static argdescr& instance()
        {
            static argdescr self;
            return self;
        };

    public:
        static argdescr_st& arg(int64_t tid)
        {
            const auto it = instance().map.find(tid);
            if(it != instance().map.end())
            {
                return it->second;
            }
            else
            {
                argdescr_st& that = instance().map[tid];
                that.m_tid        = tid;
                return that;
            }
        }

        static bool find(void** c, int64_t tid)
        {
            c[0]          = nullptr;
            const auto it = instance().map.find(tid);
            if(it != instance().map.end())
            {
                c[0] = &it->second;
                return true;
            }
            return false;
        }

        static bool find(void** c)
        {
            return find(c, rocgraph::get_tid());
        }

        static bool free(void* obj)
        {
            argdescr_st* elm = (argdescr_st*)obj;
            const auto   it  = instance().map.find(elm->m_tid);
            if(it != instance().map.end())
            {
                if(&it->second == obj)
                {
                    instance().map.erase(it);
                    return true;
                }
                return false;
            }
            return false;
        }
    };

    void argdescr_st::define(const char*     function_filename_,
                             const char*     function_name_,
                             int             function_line_,
                             const char*     arg_name_,
                             int             arg_index_,
                             rocgraph_status status_,
                             const char*     msg_)
    {
        this->m_function_line = function_line_;
        this->m_arg_index     = arg_index_;
        this->m_status        = status_;
        strncpy(this->m_function_filename, function_filename_, 128);
        strncpy(this->m_function_name, function_name_, 128);
        strncpy(this->m_arg_name, arg_name_, 128);
        strncpy(this->m_msg, msg_, 256);
    }

    std::ostream& operator<<(std::ostream& os, const rocgraph::argdescr_st& that_)
    {
        os << "// rocGRAPH.argument.error: { \"function\"  : \"" << that_.m_function_name << "\","
           << std::endl
           << "//                             \"file\"      : \"" << that_.m_function_filename
           << "\"," << std::endl
           << "//                             \"line\"      : \"" << that_.m_function_line << "\","
           << std::endl
           << "//                             \"arg\"       : \"" << that_.m_arg_name << "\","
           << std::endl
           << "//                             \"arg_index\" : \"" << that_.m_arg_index << "\","
           << std::endl
           << "//                             \"status\"    : \""
           << rocgraph::to_string(that_.m_status) << "\"";

        if(that_.m_msg[0] != '\0')
        {
            os << "," << std::endl
               << "//                             \"msg\"       : \"" << that_.m_msg << "\" }"
               << std::endl;
        }
        else
        {
            os << "}" << std::endl;
        }

        return os;
    }
}

extern "C" {
///
/// @brief Create an argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_create(void** argdescr);

ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_free(void* argdescr);

///
/// @brief Get the message of the argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_msg(const void* argdescr, const char**);

///
/// @brief Get the status of the argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_status(const void* argdescr, rocgraph_status*);

///
/// @brief Get the thread id.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_tid(const void* argdescr, int64_t* tid);

///
/// @brief Get the process id.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_pid(const void* argdescr, int64_t* pid);

///
/// @brief Get the index of the argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_index(const void* argdescr, int*);

///
/// @brief Get the name of the argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_name(const void* argdescr, const char**);

///
/// @brief Get the function line of the argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_function_line(const void* argdescr, int*);

///
/// @brief Get the function name of the argument descriptor.
///
ROCGRAPH_EXPORT
rocgraph_status rocgraph_argdescr_get_function_name(const void* argdescr, const char**);
};

void rocgraph::argdescr_log(const char*     function_filename_,
                            const char*     function_name_,
                            int             function_line_,
                            const char*     arg_name_,
                            int             arg_index_,
                            rocgraph_status status_,
                            const char*     msg_)
{
    void* p_argdescr;
    if(rocgraph::argdescr::find(&p_argdescr))
    {

        ((rocgraph::argdescr_st*)p_argdescr)
            ->define(function_filename_,
                     function_name_,
                     function_line_,
                     arg_name_,
                     arg_index_,
                     status_,
                     msg_);

        if(rocgraph_debug_variables.get_debug_arguments_verbose())
        {
            std::cerr << ((rocgraph::argdescr_st*)p_argdescr)[0];
        }
    }
    else
    {
        rocgraph::argdescr_st argdescr;
        argdescr.define(function_filename_,
                        function_name_,
                        function_line_,
                        arg_name_,
                        arg_index_,
                        status_,
                        msg_);
        if(rocgraph_debug_variables.get_debug_arguments_verbose())
        {
            std::cerr << argdescr;
        }
    }
}

extern "C" {
rocgraph_status rocgraph_argdescr_get_msg(const void* argdescr, const char** res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = &that->m_msg[0];
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_status(const void* argdescr, rocgraph_status* res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = that->m_status;
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_pid(const void* argdescr, int64_t* res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = that->m_pid;
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_tid(const void* argdescr, int64_t* res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = that->m_tid;
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_index(const void* argdescr, int* res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = that->m_arg_index;
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_name(const void* argdescr, const char** res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = &that->m_arg_name[0];
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_function_line(const void* argdescr, int* res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    if(!res)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = that->m_function_line;
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_get_function_name(const void* argdescr, const char** res)
{
    if(!argdescr)
    {
        RETURN_IF_ROCGRAPH_ERROR(rocgraph_status_invalid_pointer);
    }
    const rocgraph::argdescr_st* that = ((const rocgraph::argdescr_st*)argdescr);
    res[0]                            = &that->m_function_name[0];
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_create(void** argdescr)
{
    rocgraph::argdescr_st& elm = rocgraph::argdescr::arg(rocgraph::get_tid());
    argdescr[0]                = &elm;
    return rocgraph_status_success;
}

rocgraph_status rocgraph_argdescr_free(void* argdescr)
{
    if(argdescr != nullptr)
    {
        rocgraph::argdescr::free(argdescr);
    }
    return rocgraph_status_success;
}
}
