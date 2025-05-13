/*! \file */

// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph-auxiliary.h"

#include "control.h"
#include "envariables.h"
#include <map>
#include <mutex>

#include "debug.h"

static std::mutex s_mutex;

bool rocgraph::debug_variables_st::get_debug() const
{
    return this->debug;
}

bool rocgraph::debug_variables_st::get_debug_force_host_assert() const
{
    return this->debug_force_host_assert;
}

bool rocgraph::debug_variables_st::get_debug_verbose() const
{
    return this->debug_verbose;
}

bool rocgraph::debug_variables_st::get_debug_kernel_launch() const
{
    return this->debug_kernel_launch;
}

bool rocgraph::debug_variables_st::get_debug_arguments() const
{
    return this->debug_arguments;
}

bool rocgraph::debug_variables_st::get_debug_arguments_verbose() const
{
    return this->debug_arguments_verbose;
}

void rocgraph::debug_variables_st::set_debug(bool value)
{
    if(value != this->debug)
    {
        s_mutex.lock();
        this->debug = value;
        s_mutex.unlock();
    }
}

void rocgraph::debug_variables_st::set_debug_verbose(bool value)
{
    if(value != this->debug_verbose)
    {
        s_mutex.lock();
        this->debug_verbose = value;
        s_mutex.unlock();
    }
}

void rocgraph::debug_variables_st::set_debug_force_host_assert(bool value)
{
    if(value != this->debug_force_host_assert)
    {
        s_mutex.lock();
        this->debug_force_host_assert = value;
        s_mutex.unlock();
    }
}

void rocgraph::debug_variables_st::set_debug_arguments(bool value)
{
    if(value != this->debug_arguments)
    {
        s_mutex.lock();
        this->debug_arguments = value;
        s_mutex.unlock();
    }
}

void rocgraph::debug_variables_st::set_debug_kernel_launch(bool value)
{
    if(value != this->debug_kernel_launch)
    {
        s_mutex.lock();
        this->debug_kernel_launch = value;
        s_mutex.unlock();
    }
}

void rocgraph::debug_variables_st::set_debug_arguments_verbose(bool value)
{
    if(value != this->debug_arguments_verbose)
    {
        s_mutex.lock();
        this->debug_arguments_verbose = value;
        s_mutex.unlock();
    }
}

extern "C" {

int rocgraph_state_debug_arguments_verbose()
{
    return rocgraph_debug_variables.get_debug_arguments_verbose() ? 1 : 0;
}

int rocgraph_state_debug_verbose()
{
    return rocgraph_debug_variables.get_debug_verbose() ? 1 : 0;
}

int rocgraph_state_debug_force_host_assert()
{
    return rocgraph_debug_variables.get_debug_force_host_assert() ? 1 : 0;
}

int rocgraph_state_debug_kernel_launch()
{
    return rocgraph_debug_variables.get_debug_kernel_launch() ? 1 : 0;
}

int rocgraph_state_debug_arguments()
{
    return rocgraph_debug_variables.get_debug_arguments() ? 1 : 0;
}

int rocgraph_state_debug()
{
    return rocgraph_debug_variables.get_debug() ? 1 : 0;
}

void rocgraph_enable_debug_arguments_verbose()
{
    rocgraph_debug_variables.set_debug_arguments_verbose(true);
}

void rocgraph_disable_debug_arguments_verbose()
{
    rocgraph_debug_variables.set_debug_arguments_verbose(false);
}

void rocgraph_enable_debug_kernel_launch()
{
    rocgraph_debug_variables.set_debug_kernel_launch(true);
}

void rocgraph_disable_debug_kernel_launch()
{
    rocgraph_debug_variables.set_debug_kernel_launch(false);
}

void rocgraph_enable_debug_arguments()
{
    rocgraph_debug_variables.set_debug_arguments(true);
    rocgraph_enable_debug_arguments_verbose();
}

void rocgraph_disable_debug_arguments()
{
    rocgraph_debug_variables.set_debug_arguments(false);
    rocgraph_disable_debug_arguments_verbose();
}

void rocgraph_enable_debug_verbose()
{
    rocgraph_debug_variables.set_debug_verbose(true);
    rocgraph_enable_debug_arguments_verbose();
}

void rocgraph_disable_debug_verbose()
{
    rocgraph_debug_variables.set_debug_verbose(false);
    rocgraph_disable_debug_arguments_verbose();
}

void rocgraph_enable_debug_force_host_assert()
{
    rocgraph_debug_variables.set_debug_force_host_assert(true);
}

void rocgraph_disable_debug_force_host_assert()
{
    rocgraph_debug_variables.set_debug_force_host_assert(false);
}

void rocgraph_enable_debug()
{
    rocgraph_debug_variables.set_debug(true);
    rocgraph_enable_debug_arguments();
}

void rocgraph_disable_debug()
{
    rocgraph_debug_variables.set_debug(false);
    rocgraph_disable_debug_arguments();
}
}
