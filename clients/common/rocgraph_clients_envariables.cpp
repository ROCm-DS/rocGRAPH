// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_envariables.hpp"
#include "rocgraph-types.h"
#include <iostream>
#include <mutex>
constexpr rocgraph_clients_envariables::var_bool   rocgraph_clients_envariables::s_var_bool_all[];
constexpr rocgraph_clients_envariables::var_string rocgraph_clients_envariables::s_var_string_all[];

template <std::size_t N, typename T>
static inline constexpr std::size_t countof(T (&)[N])
{
    return N;
}

static constexpr size_t s_var_bool_size   = countof(rocgraph_clients_envariables::s_var_bool_all);
static constexpr size_t s_var_string_size = countof(rocgraph_clients_envariables::s_var_string_all);

static constexpr const char* s_var_bool_names[s_var_bool_size]
    = {"ROCGRAPH_CLIENTS_VERBOSE", "ROCGRAPH_CLIENTS_TEST_DEBUG_ARGUMENTS"};
static constexpr const char* s_var_string_names[s_var_string_size]
    = {"ROCGRAPH_CLIENTS_MATRICES_DIR", "ROCGRAPH_TEST_DATA"};
static constexpr const char* s_var_bool_descriptions[s_var_bool_size]
    = {"0: disabled, 1: enabled", "0: disabled, 1: enabled"};
static constexpr const char* s_var_string_descriptions[s_var_string_size]
    = {"Full path of the matrices directory", "The path where the test data file is located"};

///
/// @brief Grab an environment variable value.
/// @return true if the operation is successful, false otherwise.
///
template <typename T>
static bool rocgraph_getenv(const char* name, bool& defined, T& val);

template <>
bool rocgraph_getenv<bool>(const char* name, bool& defined, bool& val)
{
    val                    = false;
    const char* getenv_str = getenv(name);
    defined                = (getenv_str != nullptr);
    if(defined)
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
    else
    {
        return true;
    }
}

template <>
bool rocgraph_getenv<std::string>(const char* name, bool& defined, std::string& val)
{
    const char* getenv_str = getenv(name);
    defined                = (getenv_str != nullptr);
    if(defined)
    {
        val = getenv_str;
    }
    return true;
}

struct rocgraph_clients_envariables_impl
{
    static std::mutex s_mutex;

public:
    //
    // \brief Return value of a Boolean variable.
    //
    inline bool get(rocgraph_clients_envariables::var_bool v) const
    {
        return this->m_var_bool[v];
    };

    //
    // \brief Return value of a string variable.
    //
    inline const char* get(rocgraph_clients_envariables::var_string v) const
    {
        return this->m_var_string[v].c_str();
    };

    //
    // \brief Set value of a Boolean variable.
    //
    inline void set(rocgraph_clients_envariables::var_bool v, bool value)
    {

        rocgraph_clients_envariables_impl::s_mutex.lock();
        this->m_var_bool[v] = value;
        rocgraph_clients_envariables_impl::s_mutex.unlock();
    };

    inline void set(rocgraph_clients_envariables::var_string v, const char* value)
    {
        rocgraph_clients_envariables_impl::s_mutex.lock();
        this->m_var_string[v] = value;
        rocgraph_clients_envariables_impl::s_mutex.unlock();
    };

    //
    // \brief Return value of a string variable.
    //
    //
    // \brief Is a Boolean variable defined ?
    //
    inline bool is_defined(rocgraph_clients_envariables::var_bool v) const
    {
        return this->m_var_bool_defined[v];
    };

    //
    // \brief Is a string variable defined ?
    //
    inline bool is_defined(rocgraph_clients_envariables::var_string v) const
    {
        return this->m_var_string_defined[v];
    };

    //
    // Return the unique instance.
    //
    static rocgraph_clients_envariables_impl& Instance();

private:
    ~rocgraph_clients_envariables_impl()                                        = default;
    rocgraph_clients_envariables_impl(const rocgraph_clients_envariables_impl&) = delete;
    rocgraph_clients_envariables_impl& operator=(const rocgraph_clients_envariables_impl&) = delete;

    bool m_var_bool[s_var_bool_size]{};
    bool m_var_bool_defined[s_var_bool_size]{};

    std::string m_var_string[s_var_string_size]{};
    bool        m_var_string_defined[s_var_string_size]{};

    rocgraph_clients_envariables_impl()
    {
        for(auto tag : rocgraph_clients_envariables::s_var_bool_all)
        {
            switch(tag)
            {
            case rocgraph_clients_envariables::VERBOSE:
            {
                const bool success = rocgraph_getenv(
                    s_var_bool_names[tag], this->m_var_bool_defined[tag], this->m_var_bool[tag]);
                if(!success)
                {
                    std::cerr << "rocgraph_getenv failed on fetching " << s_var_bool_names[tag]
                              << std::endl;
                    throw(rocgraph_status_invalid_value);
                }
                break;
            }
            case rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS:
            {
                const bool success = rocgraph_getenv(
                    s_var_bool_names[tag], this->m_var_bool_defined[tag], this->m_var_bool[tag]);
                if(!success)
                {
                    std::cerr << "rocgraph_getenv failed on fetching " << s_var_bool_names[tag]
                              << std::endl;
                    throw(rocgraph_status_invalid_value);
                }
                break;
            }
            }
        }

        for(auto tag : rocgraph_clients_envariables::s_var_string_all)
        {
            switch(tag)
            {
            case rocgraph_clients_envariables::MATRICES_DIR:
            {
                const bool success = rocgraph_getenv(s_var_string_names[tag],
                                                     this->m_var_string_defined[tag],
                                                     this->m_var_string[tag]);
                if(!success)
                {
                    std::cerr << "rocgraph_getenv failed on fetching " << s_var_string_names[tag]
                              << std::endl;
                    throw(rocgraph_status_invalid_value);
                }
                break;
            }
            case rocgraph_clients_envariables::TEST_DATA_DIR:
            {
                const bool success = rocgraph_getenv(s_var_string_names[tag],
                                                     this->m_var_string_defined[tag],
                                                     this->m_var_string[tag]);
                if(!success)
                {
                    std::cerr << "rocgraph_getenv failed on fetching " << s_var_string_names[tag]
                              << std::endl;
                    throw(rocgraph_status_invalid_value);
                }
                break;
            }
            }
        }

        if(this->m_var_bool[rocgraph_clients_envariables::VERBOSE])
        {
            for(auto tag : rocgraph_clients_envariables::s_var_bool_all)
            {
                switch(tag)
                {
                case rocgraph_clients_envariables::VERBOSE:
                {
                    const bool v = this->m_var_bool[tag];
                    std::cout << ""
                              << "env variable " << s_var_bool_names[tag] << " : "
                              << ((this->m_var_bool_defined[tag]) ? ((v) ? "enabled" : "disabled")
                                                                  : "<undefined>")
                              << std::endl;
                    break;
                }
                case rocgraph_clients_envariables::TEST_DEBUG_ARGUMENTS:
                {
                    const bool v = this->m_var_bool[tag];
                    std::cout << ""
                              << "env variable " << s_var_bool_names[tag] << " : "
                              << ((this->m_var_bool_defined[tag]) ? ((v) ? "enabled" : "disabled")
                                                                  : "<undefined>")
                              << std::endl;
                    break;
                }
                }
            }

            for(auto tag : rocgraph_clients_envariables::s_var_string_all)
            {
                switch(tag)
                {
                case rocgraph_clients_envariables::MATRICES_DIR:
                {
                    const std::string v = this->m_var_string[tag];
                    std::cout << ""
                              << "env variable " << s_var_string_names[tag] << " : "
                              << ((this->m_var_string_defined[tag]) ? this->m_var_string[tag]
                                                                    : "<undefined>")
                              << std::endl;
                    break;
                }
                case rocgraph_clients_envariables::TEST_DATA_DIR:
                {
                    const std::string v = this->m_var_string[tag];
                    std::cout << ""
                              << "env variable " << s_var_string_names[tag] << " : "
                              << ((this->m_var_string_defined[tag]) ? this->m_var_string[tag]
                                                                    : "<undefined>")
                              << std::endl;
                    break;
                }
                }
            }
        }
    }
};

std::mutex rocgraph_clients_envariables_impl::s_mutex;

rocgraph_clients_envariables_impl& rocgraph_clients_envariables_impl::Instance()
{
    static rocgraph_clients_envariables_impl instance;
    return instance;
}

bool rocgraph_clients_envariables::is_defined(rocgraph_clients_envariables::var_string v)
{
    return rocgraph_clients_envariables_impl::Instance().is_defined(v);
}

const char* rocgraph_clients_envariables::get(rocgraph_clients_envariables::var_string v)
{
    return rocgraph_clients_envariables_impl::Instance().get(v);
}

void rocgraph_clients_envariables::set(rocgraph_clients_envariables::var_string v,
                                       const char*                              value)
{
    rocgraph_clients_envariables_impl::Instance().set(v, value);
}

const char* rocgraph_clients_envariables::get_name(rocgraph_clients_envariables::var_string v)
{
    return s_var_string_names[v];
}

const char*
    rocgraph_clients_envariables::get_description(rocgraph_clients_envariables::var_string v)
{
    return s_var_string_descriptions[v];
}

bool rocgraph_clients_envariables::is_defined(rocgraph_clients_envariables::var_bool v)
{
    return rocgraph_clients_envariables_impl::Instance().is_defined(v);
}

bool rocgraph_clients_envariables::get(rocgraph_clients_envariables::var_bool v)
{
    return rocgraph_clients_envariables_impl::Instance().get(v);
}

void rocgraph_clients_envariables::set(rocgraph_clients_envariables::var_bool v, bool value)
{
    rocgraph_clients_envariables_impl::Instance().set(v, value);
}

const char* rocgraph_clients_envariables::get_name(rocgraph_clients_envariables::var_bool v)
{
    return s_var_bool_names[v];
}

const char* rocgraph_clients_envariables::get_description(rocgraph_clients_envariables::var_bool v)
{
    return s_var_bool_descriptions[v];
}
