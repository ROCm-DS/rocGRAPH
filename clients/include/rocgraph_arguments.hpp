/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief rocgraph_arguments.hpp provides a class to parse command arguments in both,
 *  clients and gtest. If class structure is changed, rocgraph_common.yaml must also be
 *  changed.
 */

#pragma once
#ifndef ROCGRAPH_ARGUMENTS_HPP
#define ROCGRAPH_ARGUMENTS_HPP

#include "rocgraph_datatype2string.hpp"
#include "rocgraph_math.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>

struct Arguments
{
    rocgraph_int M;
    rocgraph_int N;
    rocgraph_int K;
    rocgraph_int nnz;
    rocgraph_int block_dim;
    rocgraph_int row_block_dimA;
    rocgraph_int col_block_dimA;
    rocgraph_int row_block_dimB;
    rocgraph_int col_block_dimB;

    rocgraph_int dimx;
    rocgraph_int dimy;
    rocgraph_int dimz;

    rocgraph_int ll;
    rocgraph_int l;
    rocgraph_int u;
    rocgraph_int uu;

    rocgraph_indextype index_type_I;
    rocgraph_indextype index_type_J;

    rocgraph_datatype a_type;
    rocgraph_datatype b_type;
    rocgraph_datatype c_type;
    rocgraph_datatype x_type;
    rocgraph_datatype y_type;
    rocgraph_datatype compute_type;

    rocgraph_indextype A_row_indextype;
    rocgraph_indextype A_col_indextype;

    rocgraph_indextype B_row_indextype;
    rocgraph_indextype B_col_indextype;

    rocgraph_indextype C_row_indextype;
    rocgraph_indextype C_col_indextype;

    double alpha;
    double alphai;
    double beta;
    double betai;
    double threshold;
    double percentage;

    rocgraph_operation       transA;
    rocgraph_operation       transB;
    rocgraph_index_base      baseA;
    rocgraph_index_base      baseB;
    rocgraph_index_base      baseC;
    rocgraph_index_base      baseD;
    rocgraph_action          action;
    rocgraph_hyb_partition   part;
    rocgraph_matrix_type     matrix_type;
    rocgraph_diag_type       diag;
    rocgraph_fill_mode       uplo;
    rocgraph_storage_mode    storage;
    rocgraph_analysis_policy apol;
    rocgraph_solve_policy    spol;
    rocgraph_direction       direction;
    rocgraph_order           order;
    rocgraph_order           orderB;
    rocgraph_order           orderC;
    rocgraph_format          formatA;
    rocgraph_format          formatB;

    rocgraph_matrix_init      matrix;
    rocgraph_matrix_init_kind matrix_init_kind;

    rocgraph_int unit_check;
    rocgraph_int timing;
    rocgraph_int iters;

    int64_t      denseld;
    rocgraph_int batch_count;
    rocgraph_int batch_count_A;
    rocgraph_int batch_count_B;
    rocgraph_int batch_count_C;
    rocgraph_int batch_stride;
    rocgraph_int ld_multiplier_B;
    rocgraph_int ld_multiplier_C;

    uint32_t algo;

    int    numericboost;
    double boosttol;
    double boostval;
    double boostvali;

    double tolm;

    bool graph_test;

    char filename[128];
    char function[64];
    char name[64];
    char category[32];
    char hardware[32];
    char skip_hardware[32];

    uint32_t req_memory;

    // Validate input format.
    // rocgraph_gentest.py is expected to conform to this format.
    // rocgraph_gentest.py uses rocgraph_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocgraph_arguments.hpp and rocgraph_common.yaml\n"
                         "define exactly the same Arguments, that rocgraph_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[9]{}, trailer[9]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));
        if(strcmp(header, "rocGRAPH"))
            error("header");
        else if(strcmp(trailer, "ROCgraph"))
            error("trailer");

        auto check_func = [&, sig = (uint8_t)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(uint8_t i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const uint8_t*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define ROCGRAPH_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        ROCGRAPH_FORMAT_CHECK(M);
        ROCGRAPH_FORMAT_CHECK(N);
        ROCGRAPH_FORMAT_CHECK(K);
        ROCGRAPH_FORMAT_CHECK(nnz);
        ROCGRAPH_FORMAT_CHECK(block_dim);
        ROCGRAPH_FORMAT_CHECK(row_block_dimA);
        ROCGRAPH_FORMAT_CHECK(col_block_dimA);
        ROCGRAPH_FORMAT_CHECK(row_block_dimB);
        ROCGRAPH_FORMAT_CHECK(col_block_dimB);
        ROCGRAPH_FORMAT_CHECK(dimx);
        ROCGRAPH_FORMAT_CHECK(dimy);
        ROCGRAPH_FORMAT_CHECK(dimz);
        ROCGRAPH_FORMAT_CHECK(ll);
        ROCGRAPH_FORMAT_CHECK(l);
        ROCGRAPH_FORMAT_CHECK(u);
        ROCGRAPH_FORMAT_CHECK(uu);
        ROCGRAPH_FORMAT_CHECK(index_type_I);
        ROCGRAPH_FORMAT_CHECK(index_type_J);
        ROCGRAPH_FORMAT_CHECK(a_type);
        ROCGRAPH_FORMAT_CHECK(b_type);
        ROCGRAPH_FORMAT_CHECK(c_type);
        ROCGRAPH_FORMAT_CHECK(x_type);
        ROCGRAPH_FORMAT_CHECK(y_type);
        ROCGRAPH_FORMAT_CHECK(compute_type);
        ROCGRAPH_FORMAT_CHECK(A_row_indextype);
        ROCGRAPH_FORMAT_CHECK(A_col_indextype);
        ROCGRAPH_FORMAT_CHECK(B_row_indextype);
        ROCGRAPH_FORMAT_CHECK(B_col_indextype);
        ROCGRAPH_FORMAT_CHECK(C_row_indextype);
        ROCGRAPH_FORMAT_CHECK(C_col_indextype);
        ROCGRAPH_FORMAT_CHECK(alpha);
        ROCGRAPH_FORMAT_CHECK(alphai);
        ROCGRAPH_FORMAT_CHECK(beta);
        ROCGRAPH_FORMAT_CHECK(betai);
        ROCGRAPH_FORMAT_CHECK(threshold);
        ROCGRAPH_FORMAT_CHECK(percentage);
        ROCGRAPH_FORMAT_CHECK(transA);
        ROCGRAPH_FORMAT_CHECK(transB);
        ROCGRAPH_FORMAT_CHECK(baseA);
        ROCGRAPH_FORMAT_CHECK(baseB);
        ROCGRAPH_FORMAT_CHECK(baseC);
        ROCGRAPH_FORMAT_CHECK(baseD);
        ROCGRAPH_FORMAT_CHECK(action);
        ROCGRAPH_FORMAT_CHECK(part);
        ROCGRAPH_FORMAT_CHECK(matrix_type);
        ROCGRAPH_FORMAT_CHECK(diag);
        ROCGRAPH_FORMAT_CHECK(uplo);
        ROCGRAPH_FORMAT_CHECK(storage);
        ROCGRAPH_FORMAT_CHECK(apol);
        ROCGRAPH_FORMAT_CHECK(spol);
        ROCGRAPH_FORMAT_CHECK(direction);
        ROCGRAPH_FORMAT_CHECK(order);
        ROCGRAPH_FORMAT_CHECK(orderB);
        ROCGRAPH_FORMAT_CHECK(orderC);
        ROCGRAPH_FORMAT_CHECK(formatA);
        ROCGRAPH_FORMAT_CHECK(formatB);
        ROCGRAPH_FORMAT_CHECK(matrix);
        ROCGRAPH_FORMAT_CHECK(matrix_init_kind);
        ROCGRAPH_FORMAT_CHECK(unit_check);
        ROCGRAPH_FORMAT_CHECK(timing);
        ROCGRAPH_FORMAT_CHECK(iters);
        ROCGRAPH_FORMAT_CHECK(denseld);
        ROCGRAPH_FORMAT_CHECK(batch_count);
        ROCGRAPH_FORMAT_CHECK(batch_count_A);
        ROCGRAPH_FORMAT_CHECK(batch_count_B);
        ROCGRAPH_FORMAT_CHECK(batch_count_C);
        ROCGRAPH_FORMAT_CHECK(batch_stride);
        ROCGRAPH_FORMAT_CHECK(ld_multiplier_B);
        ROCGRAPH_FORMAT_CHECK(ld_multiplier_C);
        ROCGRAPH_FORMAT_CHECK(algo);
        ROCGRAPH_FORMAT_CHECK(numericboost);
        ROCGRAPH_FORMAT_CHECK(boosttol);
        ROCGRAPH_FORMAT_CHECK(boostval);
        ROCGRAPH_FORMAT_CHECK(boostvali);
        ROCGRAPH_FORMAT_CHECK(tolm);
        ROCGRAPH_FORMAT_CHECK(graph_test);
        ROCGRAPH_FORMAT_CHECK(filename);
        ROCGRAPH_FORMAT_CHECK(function);
        ROCGRAPH_FORMAT_CHECK(name);
        ROCGRAPH_FORMAT_CHECK(category);
        ROCGRAPH_FORMAT_CHECK(hardware);
        ROCGRAPH_FORMAT_CHECK(skip_hardware);
        ROCGRAPH_FORMAT_CHECK(req_memory);
    }

    template <typename T>
    T get_alpha() const
    {
        return (rocgraph_isnan(alpha) || rocgraph_isnan(alphai))
                   ? static_cast<T>(0)
                   : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return (rocgraph_isnan(beta) || rocgraph_isnan(betai)) ? static_cast<T>(0)
                                                               : convert_alpha_beta<T>(beta, betai);
    }

    template <typename T>
    T get_boostval() const
    {
        return (rocgraph_isnan(boostval) || rocgraph_isnan(boostvali))
                   ? static_cast<T>(0)
                   : convert_alpha_beta<T>(boostval, boostvali);
    }

    template <typename T>
    T get_threshold() const
    {
        return (rocgraph_isnan(threshold)) ? static_cast<T>(0) : threshold;
    }

    template <typename T>
    T get_percentage() const
    {
        return (rocgraph_isnan(percentage)) ? static_cast<T>(0) : percentage;
    }

private:
    template <typename T>
    static T convert_alpha_beta(double r, double i)
    {
        return static_cast<T>(r);
    }

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("function", arg.function);
        print("index_type_I", rocgraph_indextype2string(arg.index_type_I));
        print("index_type_J", rocgraph_indextype2string(arg.index_type_J));
        print("a_type", rocgraph_datatype2string(arg.a_type));
        print("b_type", rocgraph_datatype2string(arg.b_type));
        print("c_type", rocgraph_datatype2string(arg.c_type));
        print("x_type", rocgraph_datatype2string(arg.x_type));
        print("y_type", rocgraph_datatype2string(arg.y_type));
        print("compute_type", rocgraph_datatype2string(arg.compute_type));
        print("A_row_indextype", rocgraph_indextype2string(arg.A_row_indextype));
        print("A_col_indextype", rocgraph_indextype2string(arg.A_col_indextype));
        print("B_row_indextype", rocgraph_indextype2string(arg.B_row_indextype));
        print("B_col_indextype", rocgraph_indextype2string(arg.B_col_indextype));
        print("C_row_indextype", rocgraph_indextype2string(arg.C_row_indextype));
        print("C_col_indextype", rocgraph_indextype2string(arg.C_col_indextype));
        print("transA", rocgraph_operation2string(arg.transA));
        print("transB", rocgraph_operation2string(arg.transB));
        print("baseA", rocgraph_indexbase2string(arg.baseA));
        print("baseB", rocgraph_indexbase2string(arg.baseB));
        print("baseC", rocgraph_indexbase2string(arg.baseC));
        print("baseD", rocgraph_indexbase2string(arg.baseD));
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("nnz", arg.nnz);
        print("block_dim", arg.block_dim);
        print("row_block_dimA", arg.row_block_dimA);
        print("col_block_dimA", arg.col_block_dimA);
        print("row_block_dimB", arg.row_block_dimB);
        print("col_block_dimB", arg.col_block_dimB);
        print("dim_x", arg.dimx);
        print("dim_y", arg.dimy);
        print("dim_z", arg.dimz);
        print("ll", arg.ll);
        print("l", arg.l);
        print("u", arg.u);
        print("uu", arg.uu);
        print("alpha", arg.alpha);
        print("alphai", arg.alphai);
        print("beta", arg.beta);
        print("betai", arg.betai);
        print("threshold", arg.threshold);
        print("percentage", arg.percentage);
        print("action", rocgraph_action2string(arg.action));
        print("part", rocgraph_partition2string(arg.part));
        print("matrix_type", rocgraph_matrixtype2string(arg.matrix_type));
        print("diag", rocgraph_diagtype2string(arg.diag));
        print("uplo", rocgraph_fillmode2string(arg.uplo));
        print("storage", rocgraph_storagemode2string(arg.storage));
        print("analysis_policy", rocgraph_analysis2string(arg.apol));
        print("solve_policy", rocgraph_solve2string(arg.spol));
        print("direction", rocgraph_direction2string(arg.direction));
        print("order", rocgraph_order2string(arg.order));
        print("orderB", rocgraph_order2string(arg.orderB));
        print("orderC", rocgraph_order2string(arg.orderC));
        print("formatA", rocgraph_format2string(arg.formatA));
        print("formatB", rocgraph_format2string(arg.formatB));
        print("matrix", rocgraph_matrix2string(arg.matrix));
        print("matrix_init_kind", rocgraph_matrix_init_kind2string(arg.matrix_init_kind));
        print("file", arg.filename);
        print("algo", arg.algo);
        print("numeric_boost", arg.numericboost);
        print("boost_tol", arg.boosttol);
        print("boost_val", arg.boostval);
        print("boost_vali", arg.boostvali);
        print("tolm", arg.tolm);
        print("graph_test", arg.graph_test);
        print("name", arg.name);
        print("category", arg.category);
        print("hardware", arg.hardware);
        print("skip_hardware", arg.skip_hardware);
        print("req_memory", arg.req_memory);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        print("denseld", arg.denseld);
        print("batch_count", arg.batch_count);
        print("batch_count_A", arg.batch_count_A);
        print("batch_count_B", arg.batch_count_B);
        print("batch_count_C", arg.batch_count_C);
        print("batch_stride", arg.batch_stride);
        print("ld_multiplier_B", arg.ld_multiplier_B);
        print("ld_multiplier_C", arg.ld_multiplier_C);
        return str << " }\n";
    }
};

static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

inline bool rocgraph_arguments_has_datafile(const Arguments& arg)
{
    return (arg.matrix == rocgraph_matrix_file_rocalution)
           || (arg.matrix == rocgraph_matrix_file_mtx) || (arg.matrix == rocgraph_matrix_file_smtx);
}

#endif // ROCGRAPH_ARGUMENTS_HPP
