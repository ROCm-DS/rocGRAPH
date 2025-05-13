/*! \file */

// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_exporter_matrixmarket.hpp"
template <typename X, typename Y>
rocgraph_status rocgraph_type_conversion(const X& x, Y& y);

rocgraph_exporter_matrixmarket::~rocgraph_exporter_matrixmarket()
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Export done." << std::endl;
    }
}

rocgraph_exporter_matrixmarket::rocgraph_exporter_matrixmarket(const std::string& filename_)
    : m_filename(filename_)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_exporter_matrixmarket::write_graph_csx(rocgraph_direction dir_,
                                                                J                  m_,
                                                                J                  n_,
                                                                I                  nnz_,
                                                                const I* __restrict__ ptr_,
                                                                const J* __restrict__ ind_,
                                                                const T* __restrict__ val_,
                                                                rocgraph_index_base base_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocgraph_status_internal_error;
    }
    if(std::is_same<T, double>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix coordinate ";
    out << "real";
    out << " general" << std::endl;
    out << m_ << " " << n_ << " " << nnz_ << std::endl;
    switch(dir_)
    {
    case rocgraph_direction_row:
    {
        for(J i = 0; i < m_; ++i)
        {
            for(I at = ptr_[i] - base_; at < ptr_[i + 1] - base_; ++at)
            {
                J j = ind_[at] - base_;
                T x = val_[at];
                out << (i + 1) << " " << (j + 1);
                out << " " << x;
                out << std::endl;
            }
        }
        out.close();
        return rocgraph_status_success;
    }
    case rocgraph_direction_column:
    {
        for(J j = 0; j < n_; ++j)
        {
            for(I at = ptr_[j] - base_; at < ptr_[j + 1] - base_; ++at)
            {
                J i = ind_[at] - base_;
                T x = val_[at];
                out << (i + 1) << " " << (j + 1);
                out << " " << x;
                out << std::endl;
            }
        }
        out.close();
        return rocgraph_status_success;
    }
    }
    return rocgraph_status_invalid_value;
}

template <typename T, typename I, typename J>
rocgraph_status rocgraph_exporter_matrixmarket::write_graph_gebsx(rocgraph_direction dir_,
                                                                  rocgraph_direction dirb_,
                                                                  J                  mb_,
                                                                  J                  nb_,
                                                                  I                  nnzb_,
                                                                  J                  block_dim_row_,
                                                                  J block_dim_column_,
                                                                  const I* __restrict__ ptr_,
                                                                  const J* __restrict__ ind_,
                                                                  const T* __restrict__ val_,
                                                                  rocgraph_index_base base_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocgraph_status_internal_error;
    }
    if(std::is_same<T, double>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix coordinate ";
    out << "real";
    out << " general" << std::endl;
    out << mb_ * block_dim_row_ << " " << nb_ * block_dim_column_ << " "
        << nnzb_ * block_dim_row_ * block_dim_column_ << std::endl;
    switch(dir_)
    {
    case rocgraph_direction_row:
    {
        for(J ib = 0; ib < mb_; ++ib)
        {
            I i = ib * block_dim_row_;
            for(I at = ptr_[ib] - base_; at < ptr_[ib + 1] - base_; ++at)
            {
                J j = (ind_[at] - base_) * block_dim_column_;
                switch(dirb_)
                {
                case rocgraph_direction_row:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_column_ * k + l];
                            out << (i + k) << " " << (j + l);
                            out << " " << v;
                            out << std::endl;
                        }
                    }
                    break;
                }
                case rocgraph_direction_column:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_row_ * l + k];
                            out << (i + k) << " " << (j + l);
                            out << " " << std::real(v) << " " << std::imag(v);
                            out << std::endl;
                        }
                    }
                    break;
                }
                }
            }
        }
        out.close();
        return rocgraph_status_success;
    }

    case rocgraph_direction_column:
    {
        for(J jb = 0; jb < nb_; ++jb)
        {
            I j = jb * block_dim_column_;
            for(I at = ptr_[jb] - base_; at < ptr_[jb + 1] - base_; ++at)
            {
                J i = (ind_[at] - base_) * block_dim_row_;
                switch(dirb_)
                {

                case rocgraph_direction_row:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_column_ * k + l];
                            out << (i + k) << " " << (j + l);
                            out << " " << v;
                            out << std::endl;
                        }
                    }
                    break;
                }

                case rocgraph_direction_column:
                {
                    for(J k = 0; k < block_dim_row_; ++k)
                    {
                        for(J l = 0; l < block_dim_column_; ++l)
                        {
                            auto v = val_[at * block_dim_row_ * block_dim_column_
                                          + block_dim_row_ * l + k];
                            out << (i + k) << " " << (j + l);
                            out << " " << v;
                            out << std::endl;
                        }
                    }
                    break;
                }
                }
            }
        }
        out.close();
        return rocgraph_status_success;
    }
    }
    return rocgraph_status_invalid_value;

    std::cerr << "rocgraph_exporter_matrixmarket, gebsx not supported." << std::endl;
    return rocgraph_status_not_implemented;
}

template <typename T, typename I>
rocgraph_status
    rocgraph_exporter_matrixmarket::write_dense_vector(I nmemb_, const T* __restrict__ x_, I incx_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocgraph_status_internal_error;
    }
    if(std::is_same<T, double>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix array ";
    out << "real";
    out << " general" << std::endl;
    out << nmemb_ << " 1" << std::endl;
    for(I i = 0; i < nmemb_; ++i)
    {
        out << x_[i * incx_] << std::endl;
    }
    out.close();
    return rocgraph_status_success;
}

template <typename T, typename I>
rocgraph_status rocgraph_exporter_matrixmarket::write_dense_matrix(
    rocgraph_order order_, I m_, I n_, const T* __restrict__ x_, I ld_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocgraph_status_internal_error;
    }
    if(std::is_same<T, double>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix array ";
    out << "real";
    out << " general" << std::endl;
    out << m_ << " " << n_ << std::endl;
    switch(order_)
    {
    case rocgraph_order_row:
    {
        for(I i = 0; i < m_; ++i)
        {
            for(I j = 0; j < n_; ++j)
            {
                out << " " << x_[i * ld_ + j];
            }
            out << std::endl;
        }
        out.close();
        return rocgraph_status_success;
    }
    case rocgraph_order_column:
    {
        for(I i = 0; i < m_; ++i)
        {
            for(I j = 0; j < n_; ++j)
            {
                out << " " << x_[j * ld_ + i];
            }
            out << std::endl;
        }
        out.close();
        return rocgraph_status_success;
    }
    }
    return rocgraph_status_invalid_value;
}

template <typename T, typename I>
rocgraph_status rocgraph_exporter_matrixmarket::write_graph_coo(I m_,
                                                                I n_,
                                                                I nnz_,
                                                                const I* __restrict__ row_ind_,
                                                                const I* __restrict__ col_ind_,
                                                                const T* __restrict__ val_,
                                                                rocgraph_index_base base_)
{
    std::ofstream out(this->m_filename);
    if(!out.is_open())
    {
        return rocgraph_status_internal_error;
    }

    if(std::is_same<T, double>())
    {
        out.precision(15);
        out.setf(std::ios::scientific);
    }
    else
    {
        out.precision(7);
        out.setf(std::ios::scientific);
    }

    out << "%%MatrixMarket matrix coordinate ";
    out << "real";

    out << " general" << std::endl;
    out << m_ << " " << n_ << " " << nnz_ << std::endl;

    for(I i = 0; i < nnz_; ++i)
    {
        out << ((row_ind_[i] - base_) + 1) << " " << ((col_ind_[i] - base_) + 1);
        out << " " << std::real(val_[i]) << " " << std::imag(val_[i]) << std::endl;
    }

    out.close();
    return rocgraph_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                \
    template rocgraph_status rocgraph_exporter_matrixmarket::write_graph_csx(   \
        rocgraph_direction,                                                     \
        J,                                                                      \
        J,                                                                      \
        I,                                                                      \
        const I* __restrict__,                                                  \
        const J* __restrict__,                                                  \
        const T* __restrict__,                                                  \
        rocgraph_index_base);                                                   \
    template rocgraph_status rocgraph_exporter_matrixmarket::write_graph_gebsx( \
        rocgraph_direction,                                                     \
        rocgraph_direction,                                                     \
        J,                                                                      \
        J,                                                                      \
        I,                                                                      \
        J,                                                                      \
        J,                                                                      \
        const I* __restrict__,                                                  \
        const J* __restrict__,                                                  \
        const T* __restrict__,                                                  \
        rocgraph_index_base)

#define INSTANTIATE_TI(T, I)                                                     \
    template rocgraph_status rocgraph_exporter_matrixmarket::write_dense_vector( \
        I, const T* __restrict__, I);                                            \
    template rocgraph_status rocgraph_exporter_matrixmarket::write_dense_matrix( \
        rocgraph_order, I, I, const T* __restrict__, I);                         \
    template rocgraph_status rocgraph_exporter_matrixmarket::write_graph_coo(    \
        I,                                                                       \
        I,                                                                       \
        I,                                                                       \
        const I* __restrict__,                                                   \
        const I* __restrict__,                                                   \
        const T* __restrict__,                                                   \
        rocgraph_index_base)

INSTANTIATE_TIJ(float, int32_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int64_t);

INSTANTIATE_TIJ(double, int32_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int64_t);

INSTANTIATE_TI(float, int32_t);
INSTANTIATE_TI(float, int64_t);

INSTANTIATE_TI(double, int32_t);
INSTANTIATE_TI(double, int64_t);
