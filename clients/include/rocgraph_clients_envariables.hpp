/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

///
/// @brief Definition of a utility struct to grab environment variables
/// for the rocgraph clients.
//
/// The corresponding environment variable is the literal enum string
/// with the prefix ROCGRAPH_CLIENTS_.
/// Example: rocgraph_clients_envariables::VERBOSE will have a one to one correspondance with the environment variable
/// ROCGRAPH_CLIENTS_VERBOSE.
/// Obviously it loads environment variables at run time.
///
struct rocgraph_clients_envariables
{

    ///
    /// @brief Enumerate Boolean environment variables.
    ///
    typedef enum var_bool_ : int32_t
    {
        VERBOSE,
        TEST_DEBUG_ARGUMENTS
    } var_bool;

    static constexpr var_bool s_var_bool_all[] = {VERBOSE, TEST_DEBUG_ARGUMENTS};

    ///
    /// @brief Return value of a Boolean variable.
    ///
    static bool get(var_bool v);

    ///
    /// @brief Set value of a Boolean variable.
    ///
    static void set(var_bool v, bool value);

    ///
    /// @brief Is the Boolean enviromnent variable defined ?
    ///
    static bool is_defined(var_bool v);

    ///
    /// @brief Return the name of a Boolean variable.
    ///
    static const char* get_name(var_bool v);

    ///
    /// @brief Return the description of a Boolean variable.
    ///
    static const char* get_description(var_bool v);

    ///
    /// @brief Enumerate string environment variables.
    ///
    typedef enum var_string_ : int32_t
    {
        MATRICES_DIR,
        TEST_DATA_DIR
    } var_string;

    static constexpr var_string s_var_string_all[2] = {MATRICES_DIR, TEST_DATA_DIR};

    ///
    /// @brief Return value of a string variable.
    ///
    static const char* get(var_string v);

    ///
    /// @brief Set value of a string variable.
    ///
    static void set(var_string v, const char* value);

    ///
    /// @brief Return the name of a string variable.
    ///
    static const char* get_name(var_string v);

    ///
    /// @brief Return the description of a string variable.
    ///
    static const char* get_description(var_string v);

    ///
    /// @brief Is the string enviromnent variable defined ?
    ///
    static bool is_defined(var_string v);
};
