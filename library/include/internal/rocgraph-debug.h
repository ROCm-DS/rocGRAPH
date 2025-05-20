// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file*/
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCGRAPH_DEBUG_H
#define ROCGRAPH_DEBUG_H

#include "rocgraph-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
   *  \brief Enable debug kernel launch.
   * \details If the debug kernel launch is enabled then hip errors are checked before and after every kernel launch.
   * \note This routine ignores the environment variable ROCGRAPH_DEBUG_KERNEL_LAUNCH.
   */
ROCGRAPH_EXPORT
void rocgraph_enable_debug_kernel_launch();

/*! \ingroup aux_module
   *  \brief Disable debug kernel launch.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG_KERNEL_LAUNCH.
   */
ROCGRAPH_EXPORT void rocgraph_disable_debug_kernel_launch();

/*! \ingroup aux_module
   * \return 1 if enabled, 0 otherwise.
   */
ROCGRAPH_EXPORT int rocgraph_state_debug_kernel_launch();

/*! \ingroup aux_module
   *  \brief Enable debug arguments.
   * \details If the debug arguments is enabled then argument descriptors are internally available when an argument checking occurs. It provide information to the user depending of the setup of the verbosity
   * \ref rocgraph_enable_debug_arguments_verbose, \ref rocgraph_disable_debug_arguments_verbose and \ref rocgraph_state_debug_arguments_verbose.
   * \note This routine ignores the environment variable ROCGRAPH_DEBUG_ARGUMENTS.
   * \note This routine enables debug arguments verbose with \ref rocgraph_enable_debug_arguments_verbose.
   */
ROCGRAPH_EXPORT
void rocgraph_enable_debug_arguments();

/*! \ingroup aux_module
   *  \brief Disable debug arguments.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG_ARGUMENTS.
   *  \note This routines disables debug arguments verbose.
   */
ROCGRAPH_EXPORT
void rocgraph_disable_debug_arguments();

/*! \ingroup aux_module
   * \return 1 if enabled, 0 otherwise.
   */
ROCGRAPH_EXPORT
int rocgraph_state_debug_arguments();

/*! \ingroup aux_module
   *  \brief Enable debug arguments verbose.
   *  \details The debug argument verbose displays information related to argument descriptors created from argument checking failures.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG_ARGUMENTS_VERBOSE)
   */
ROCGRAPH_EXPORT
void rocgraph_enable_debug_arguments_verbose();

/*! \ingroup aux_module
   *  \brief Disable debug arguments verbose.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG_ARGUMENTS_VERBOSE)
   */
ROCGRAPH_EXPORT
void rocgraph_disable_debug_arguments_verbose();

/*! \ingroup aux_module
   * \brief Get state of debug arguments verbose.
   * \return 1 if enabled, 0 otherwise.
   */
ROCGRAPH_EXPORT
int rocgraph_state_debug_arguments_verbose();

/*! \ingroup aux_module
   *  \brief Enable debug.
   * \details If the debug is enabled then code traces are generated when unsuccessful status returns occur. It provides information to the user depending of the set of the verbosity
   * (\ref rocgraph_enable_debug_verbose, \ref rocgraph_disable_debug_verbose and \ref rocgraph_state_debug_verbose).
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG.
   * \note \ref rocgraph_enable_debug_verbose and \ref rocgraph_enable_debug_arguments are called.
   */
ROCGRAPH_EXPORT
void rocgraph_enable_debug();

/*! \ingroup aux_module
   *  \brief Disable debug.
   *  \note This routine also disables debug arguments with \ref rocgraph_disable_debug_arguments.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG.
   */
ROCGRAPH_EXPORT
void rocgraph_disable_debug();
/*! \ingroup aux_module
   * \brief Get state of  debug.
   * \return 1 if enabled, 0 otherwise.
   */
ROCGRAPH_EXPORT
int rocgraph_state_debug();

/*! \ingroup aux_module
   *  \brief Enable debug verbose.
   *  \details The debug verbose displays a stack of code traces showing where the code is handling a unsuccessful status.
   *  \note This routine enables debug arguments verbose with \ref rocgraph_enable_debug_arguments_verbose.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG_VERBOSE.
   */
ROCGRAPH_EXPORT
void rocgraph_enable_debug_verbose();

/*! \ingroup aux_module
   *  \brief Disable debug verbose.
   *  \note This routine disables debug arguments verbose with  \ref rocgraph_disable_debug_arguments.
   *  \note This routine ignores the environment variable ROCGRAPH_DEBUG_VERBOSE.
   */
ROCGRAPH_EXPORT
void rocgraph_disable_debug_verbose();
/*! \ingroup aux_module
   * \brief Get state of  debug verbose.
   * \return 1 if enabled, 0 otherwise.
   */
ROCGRAPH_EXPORT
int rocgraph_state_debug_verbose();

/*! \ingroup aux_module
   *  \brief Enable debug force host assert.
   *  \details The debug force host assert forces the evaluation of assert on host when the compiler directive NDEBUG is used.
   */
ROCGRAPH_EXPORT
void rocgraph_enable_debug_force_host_assert();

/*! \ingroup aux_module
   *  \brief Disable debug force host assert.
   */
ROCGRAPH_EXPORT
void rocgraph_disable_debug_force_host_assert();

/*! \ingroup aux_module
   * \brief Get state of  debug force host assert.
   * \return 1 if enabled, 0 otherwise.
   */
ROCGRAPH_EXPORT
int rocgraph_state_debug_force_host_assert();

#ifdef __cplusplus
}
#endif

#endif /* ROCGRAPH_DEBUG_H */
