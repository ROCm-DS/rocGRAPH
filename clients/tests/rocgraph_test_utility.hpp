/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "rocgraph_test_enum.hpp"

template <typename T>
inline void rocgraph_test_name_suffix_generator_print(std::ostream& s, T item)
{
    s << item;
}

#define ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(ENUM_TYPE, TOSTRING)      \
    template <>                                                                        \
    inline void rocgraph_test_name_suffix_generator_print<ENUM_TYPE>(std::ostream & s, \
                                                                     ENUM_TYPE item)   \
    {                                                                                  \
        s << TOSTRING(item);                                                           \
    }

ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_matrix_init, rocgraph_matrix2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_indextype, rocgraph_indextype2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_datatype, rocgraph_datatype2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_index_base,
                                                     rocgraph_indexbase2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_operation, rocgraph_operation2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_matrix_type,
                                                     rocgraph_matrixtype2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_diag_type, rocgraph_diagtype2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_fill_mode, rocgraph_fillmode2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_storage_mode,
                                                     rocgraph_storagemode2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_action, rocgraph_action2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_direction, rocgraph_direction2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_order, rocgraph_order2string);
ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE(rocgraph_format, rocgraph_format2string);

#undef ROCGRAPH_TEST_NAME_SUFFIX_GENERATOR_PRINT_SPECIALIZE

template <typename T>
inline void rocgraph_test_name_suffix_generator_remain(std::ostream& s, T item)
{
    rocgraph_test_name_suffix_generator_print(s << "_", item);
}

inline void rocgraph_test_name_suffix_generator_remain(std::ostream& s) {}
template <typename T, typename... R>
inline void rocgraph_test_name_suffix_generator_remain(std::ostream& s, T item, R... remains)
{
    rocgraph_test_name_suffix_generator_print(s << "_", item);
    rocgraph_test_name_suffix_generator_remain(s, remains...);
}

template <typename T, typename... R>
inline void rocgraph_test_name_suffix_generator(std::ostream& s, T item, R... remains)
{
    rocgraph_test_name_suffix_generator_print(s, item);
    rocgraph_test_name_suffix_generator_remain(s, remains...);
}
