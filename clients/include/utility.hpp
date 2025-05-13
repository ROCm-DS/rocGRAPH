/*! \file */

// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief utility.hpp provides common utilities
 */

#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "rocgraph_matrix.hpp"
#include "rocgraph_test.hpp"

#include <hip/hip_runtime_api.h>
#include <vector>

// Return index type
template <typename I>
inline rocgraph_indextype get_indextype(void);

// Return data type
template <typename T>
inline rocgraph_datatype get_datatype(void);

/*! \brief  Return \ref rocgraph_indextype */
template <>
inline rocgraph_indextype get_indextype<uint16_t>(void)
{
    return rocgraph_indextype_u16;
}

template <>
inline rocgraph_indextype get_indextype<int32_t>(void)
{
    return rocgraph_indextype_i32;
}

template <>
inline rocgraph_indextype get_indextype<int64_t>(void)
{
    return rocgraph_indextype_i64;
}

/*! \brief  Return \ref rocgraph_datatype */
template <>
inline rocgraph_datatype get_datatype<int8_t>(void)
{
    return rocgraph_datatype_i8_r;
}

template <>
inline rocgraph_datatype get_datatype<uint8_t>(void)
{
    return rocgraph_datatype_u8_r;
}

template <>
inline rocgraph_datatype get_datatype<int32_t>(void)
{
    return rocgraph_datatype_i32_r;
}

template <>
inline rocgraph_datatype get_datatype<uint32_t>(void)
{
    return rocgraph_datatype_u32_r;
}

template <>
inline rocgraph_datatype get_datatype<float>(void)
{
    return rocgraph_datatype_f32_r;
}

template <>
inline rocgraph_datatype get_datatype<double>(void)
{
    return rocgraph_datatype_f64_r;
}

inline constexpr size_t rocgraph_indextype_sizeof(rocgraph_indextype indextype_)
{
    switch(indextype_)
    {
    case rocgraph_indextype_u16:
    {
        return sizeof(uint16_t);
    }
    case rocgraph_indextype_i32:
    {
        return sizeof(int32_t);
    }
    case rocgraph_indextype_i64:
    {
        return sizeof(int64_t);
    }
    }
    return static_cast<size_t>(0);
}

inline constexpr size_t rocgraph_datatype_sizeof(rocgraph_datatype datatype_)
{
    switch(datatype_)
    {
    case rocgraph_datatype_f32_r:
    {
        return sizeof(float);
    }
    case rocgraph_datatype_i8_r:
    {
        return sizeof(int8_t);
    }
    case rocgraph_datatype_u8_r:
    {
        return sizeof(uint8_t);
    }
    case rocgraph_datatype_i32_r:
    {
        return sizeof(int32_t);
    }
    case rocgraph_datatype_u32_r:
    {
        return sizeof(uint32_t);
    }
    case rocgraph_datatype_f64_r:
    {
        return sizeof(double);
    }
    }
    return static_cast<size_t>(0);
}

/*! \brief  local handle which is automatically created and destroyed  */
class rocgraph_local_handle
{
    rocgraph_handle handle{};

public:
    rocgraph_local_handle()
        : capture_started(false)
        , graph_testing(false)
    {
        static constexpr void* raft_handle = nullptr;
        const rocgraph_status  status      = rocgraph_create_handle(&this->handle, raft_handle);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }
    rocgraph_local_handle(const Arguments& arg)
        : capture_started(false)
        , graph_testing(arg.graph_test)
    {
        static constexpr void* raft_handle = nullptr;
        const rocgraph_status  status      = rocgraph_create_handle(&this->handle, raft_handle);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }
    ~rocgraph_local_handle()
    {
        rocgraph_destroy_handle(this->handle);
    }

    // Allow rocgraph_local_handle to be used anywhere rocgraph_handle is expected
    operator rocgraph_handle&()
    {
        return this->handle;
    }
    operator const rocgraph_handle&() const
    {
        return this->handle;
    }

    void rocgraph_stream_begin_capture()
    {
        if(!(this->graph_testing))
        {
            return;
        }

#ifdef GOOGLE_TEST
        ASSERT_EQ(capture_started, false);
#endif

        CHECK_HIP_SUCCESS(hipStreamCreate(&this->graph_stream));
        CHECK_ROCGRAPH_SUCCESS(rocgraph_get_stream(*this, &this->old_stream));
        CHECK_ROCGRAPH_SUCCESS(rocgraph_set_stream(*this, this->graph_stream));

        // BEGIN GRAPH CAPTURE
        CHECK_HIP_SUCCESS(hipStreamBeginCapture(this->graph_stream, hipStreamCaptureModeGlobal));

        capture_started = true;
    }

    void rocgraph_stream_end_capture(rocgraph_int runs = 1)
    {
        if(!(this->graph_testing))
        {
            return;
        }

#ifdef GOOGLE_TEST
        ASSERT_EQ(capture_started, true);
#endif

        hipGraph_t     graph;
        hipGraphExec_t instance;

        // END GRAPH CAPTURE
        CHECK_HIP_SUCCESS(hipStreamEndCapture(this->graph_stream, &graph));
        CHECK_HIP_SUCCESS(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

        CHECK_HIP_SUCCESS(hipGraphDestroy(graph));
        CHECK_HIP_SUCCESS(hipGraphLaunch(instance, this->graph_stream));
        CHECK_HIP_SUCCESS(hipStreamSynchronize(this->graph_stream));
        CHECK_HIP_SUCCESS(hipGraphExecDestroy(instance));

        CHECK_ROCGRAPH_SUCCESS(rocgraph_set_stream(*this, this->old_stream));
        CHECK_HIP_SUCCESS(hipStreamDestroy(this->graph_stream));
        this->graph_stream = nullptr;

        capture_started = false;
    }

    hipStream_t get_stream()
    {
        hipStream_t stream;
        rocgraph_get_stream(*this, &stream);
        return stream;
    }

private:
    hipStream_t graph_stream;
    hipStream_t old_stream;
    bool        capture_started;
    bool        graph_testing;
};

/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class rocgraph_local_mat_descr
{
    rocgraph_mat_descr descr{};

public:
    rocgraph_local_mat_descr()
    {
        const rocgraph_status status = rocgraph_create_mat_descr(&this->descr);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    ~rocgraph_local_mat_descr()
    {
        rocgraph_destroy_mat_descr(this->descr);
    }

    // Allow rocgraph_local_mat_descr to be used anywhere rocgraph_mat_descr is expected
    operator rocgraph_mat_descr&()
    {
        return this->descr;
    }
    operator const rocgraph_mat_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local matrix info which is automatically created and destroyed  */
class rocgraph_local_mat_info
{
    rocgraph_mat_info info{};

public:
    rocgraph_local_mat_info()
    {
        const rocgraph_status status = rocgraph_create_mat_info(&this->info);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }
    ~rocgraph_local_mat_info()
    {
        rocgraph_destroy_mat_info(this->info);
    }

    // Sometimes useful to reset local info
    void reset()
    {
        rocgraph_destroy_mat_info(this->info);
        const rocgraph_status status = rocgraph_create_mat_info(&this->info);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    // Allow rocgraph_local_mat_info to be used anywhere rocgraph_mat_info is expected
    operator rocgraph_mat_info&()
    {
        return this->info;
    }
    operator const rocgraph_mat_info&() const
    {
        return this->info;
    }
};

/*! \brief  local color info which is automatically created and destroyed  */
class rocgraph_local_color_info
{
    rocgraph_color_info info{};

public:
    rocgraph_local_color_info()
    {
        const rocgraph_status status = rocgraph_create_color_info(&this->info);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }
    ~rocgraph_local_color_info()
    {
        rocgraph_destroy_color_info(this->info);
    }

    // Sometimes useful to reset local info
    void reset()
    {
        rocgraph_destroy_color_info(this->info);
        const rocgraph_status status = rocgraph_create_color_info(&this->info);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    // Allow rocgraph_local_color_info to be used anywhere rocgraph_color_info is expected
    operator rocgraph_color_info&()
    {
        return this->info;
    }
    operator const rocgraph_color_info&() const
    {
        return this->info;
    }
};

/*! \brief  hyb matrix structure helper to access data for tests  */
struct test_hyb
{
    rocgraph_int           m;
    rocgraph_int           n;
    rocgraph_hyb_partition partition;
    rocgraph_int           ell_nnz;
    rocgraph_int           ell_width;
    rocgraph_int*          ell_col_ind;
    void*                  ell_val;
    rocgraph_int           coo_nnz;
    rocgraph_int*          coo_row_ind;
    rocgraph_int*          coo_col_ind;
    void*                  coo_val;
};

/*! \brief  local hyb matrix structure which is automatically created and destroyed  */
class rocgraph_local_hyb_mat
{
    rocgraph_hyb_mat hyb{};

public:
    rocgraph_local_hyb_mat()
    {
        const rocgraph_status status = rocgraph_create_hyb_mat(&this->hyb);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }
    ~rocgraph_local_hyb_mat()
    {
        rocgraph_destroy_hyb_mat(this->hyb);
    }

    // Allow rocgraph_local_hyb_mat to be used anywhere rocgraph_hyb_mat is expected
    operator rocgraph_hyb_mat&()
    {
        return this->hyb;
    }
    operator const rocgraph_hyb_mat&() const
    {
        return this->hyb;
    }
};

/*! \brief  local dense vector structure which is automatically created and destroyed  */
class rocgraph_local_spvec
{
    rocgraph_spvec_descr descr{};

public:
    rocgraph_local_spvec(int64_t             size,
                         int64_t             nnz,
                         void*               indices,
                         void*               values,
                         rocgraph_indextype  idx_type,
                         rocgraph_index_base idx_base,
                         rocgraph_datatype   compute_type)
    {
        const rocgraph_status status = rocgraph_create_spvec_descr(
            &this->descr, size, nnz, indices, values, idx_type, idx_base, compute_type);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }
    ~rocgraph_local_spvec()
    {
        if(this->descr != nullptr)
        {
            rocgraph_destroy_spvec_descr(this->descr);
        }
    }

    // Allow rocgraph_local_spvec to be used anywhere rocgraph_spvec_descr is expected
    operator rocgraph_spvec_descr&()
    {
        return this->descr;
    }
    operator const rocgraph_spvec_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local graph matrix structure which is automatically created and destroyed  */
class rocgraph_local_spmat
{
    rocgraph_spmat_descr descr{};

public:
    rocgraph_local_spmat(int64_t             m,
                         int64_t             n,
                         int64_t             nnz,
                         void*               coo_row_ind,
                         void*               coo_col_ind,
                         void*               coo_val,
                         rocgraph_indextype  idx_type,
                         rocgraph_index_base idx_base,
                         rocgraph_datatype   compute_type)
    {
        const rocgraph_status status = rocgraph_create_coo_descr(&this->descr,
                                                                 m,
                                                                 n,
                                                                 nnz,
                                                                 coo_row_ind,
                                                                 coo_col_ind,
                                                                 coo_val,
                                                                 idx_type,
                                                                 idx_base,
                                                                 compute_type);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocgraph_int>
    explicit rocgraph_local_spmat(coo_matrix<MODE, T, I>& h)
        : rocgraph_local_spmat(h.m,
                               h.n,
                               h.nnz,
                               h.row_ind,
                               h.col_ind,
                               h.val,
                               get_indextype<I>(),
                               h.base,
                               get_datatype<T>())
    {
    }

    rocgraph_local_spmat(int64_t             m,
                         int64_t             n,
                         int64_t             nnz,
                         void*               row_col_ptr,
                         void*               row_col_ind,
                         void*               val,
                         rocgraph_indextype  row_col_ptr_type,
                         rocgraph_indextype  row_col_ind_type,
                         rocgraph_index_base idx_base,
                         rocgraph_datatype   compute_type,
                         bool                csc_format = false)
    {
        if(csc_format == false)
        {
            const rocgraph_status status = rocgraph_create_csr_descr(&this->descr,
                                                                     m,
                                                                     n,
                                                                     nnz,
                                                                     row_col_ptr,
                                                                     row_col_ind,
                                                                     val,
                                                                     row_col_ptr_type,
                                                                     row_col_ind_type,
                                                                     idx_base,
                                                                     compute_type);
            if(status != rocgraph_status_success)
            {
                throw(status);
            }
        }
        else
        {
            const rocgraph_status status = rocgraph_create_csc_descr(&this->descr,
                                                                     m,
                                                                     n,
                                                                     nnz,
                                                                     row_col_ptr,
                                                                     row_col_ind,
                                                                     val,
                                                                     row_col_ptr_type,
                                                                     row_col_ind_type,
                                                                     idx_base,
                                                                     compute_type);
            if(status != rocgraph_status_success)
            {
                throw(status);
            }
        }
    }

    template <memory_mode::value_t MODE,
              rocgraph_direction   DIRECTION_,
              typename T,
              typename I = rocgraph_int,
              typename J = rocgraph_int>
    explicit rocgraph_local_spmat(csx_matrix<MODE, DIRECTION_, T, I, J>& h)
        : rocgraph_local_spmat(h.m,
                               h.n,
                               h.nnz,
                               h.ptr,
                               h.ind,
                               h.val,
                               get_indextype<I>(),
                               get_indextype<J>(),
                               h.base,
                               get_datatype<T>(),
                               (DIRECTION_ == rocgraph_direction_column))
    {
    }

    rocgraph_local_spmat(int64_t             mb,
                         int64_t             nb,
                         int64_t             nnzb,
                         rocgraph_direction  block_dir,
                         int64_t             block_dim,
                         void*               row_col_ptr,
                         void*               row_col_ind,
                         void*               val,
                         rocgraph_indextype  row_col_ptr_type,
                         rocgraph_indextype  row_col_ind_type,
                         rocgraph_index_base idx_base,
                         rocgraph_datatype   compute_type,
                         rocgraph_format     format)
    {

        if(format == rocgraph_format_bsr)
        {
            const rocgraph_status status = rocgraph_create_bsr_descr(&this->descr,
                                                                     mb,
                                                                     nb,
                                                                     nnzb,
                                                                     block_dir,
                                                                     block_dim,
                                                                     row_col_ptr,
                                                                     row_col_ind,
                                                                     val,
                                                                     row_col_ptr_type,
                                                                     row_col_ind_type,
                                                                     idx_base,
                                                                     compute_type);
            if(status != rocgraph_status_success)
            {
                throw(status);
            }
        }
    }

    rocgraph_local_spmat(int64_t             m,
                         int64_t             n,
                         void*               ell_col_ind,
                         void*               ell_val,
                         int64_t             ell_width,
                         rocgraph_indextype  idx_type,
                         rocgraph_index_base idx_base,
                         rocgraph_datatype   compute_type)
    {
        const rocgraph_status status = rocgraph_create_ell_descr(
            &this->descr, m, n, ell_col_ind, ell_val, ell_width, idx_type, idx_base, compute_type);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    ~rocgraph_local_spmat()
    {
        if(this->descr != nullptr)
            rocgraph_destroy_spmat_descr(this->descr);
    }

    // Allow rocgraph_local_spmat to be used anywhere rocgraph_spmat_descr is expected
    operator rocgraph_spmat_descr&()
    {
        return this->descr;
    }
    operator const rocgraph_spmat_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local dense vector structure which is automatically created and destroyed  */
class rocgraph_local_dnvec
{
    rocgraph_dnvec_descr descr{};

public:
    rocgraph_local_dnvec(int64_t size, void* values, rocgraph_datatype compute_type)
    {
        const rocgraph_status status
            = rocgraph_create_dnvec_descr(&this->descr, size, values, compute_type);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T>
    explicit rocgraph_local_dnvec(dense_matrix<MODE, T>& h)
        : rocgraph_local_dnvec(h.m, (T*)h, get_datatype<T>())
    {
    }

    ~rocgraph_local_dnvec()
    {
        if(this->descr != nullptr)
            rocgraph_destroy_dnvec_descr(this->descr);
    }

    // Allow rocgraph_local_dnvec to be used anywhere rocgraph_dnvec_descr is expected
    operator rocgraph_dnvec_descr&()
    {
        return this->descr;
    }
    operator const rocgraph_dnvec_descr&() const
    {
        return this->descr;
    }
};

/*! \brief  local dense matrix structure which is automatically created and destroyed  */
class rocgraph_local_dnmat
{
    rocgraph_dnmat_descr descr{};

public:
    rocgraph_local_dnmat(int64_t           rows,
                         int64_t           cols,
                         int64_t           ld,
                         void*             values,
                         rocgraph_datatype compute_type,
                         rocgraph_order    order)
    {
        const rocgraph_status status = rocgraph_create_dnmat_descr(
            &this->descr, rows, cols, ld, values, compute_type, order);
        if(status != rocgraph_status_success)
        {
            throw(status);
        }
    }

    template <memory_mode::value_t MODE, typename T>
    explicit rocgraph_local_dnmat(dense_matrix<MODE, T>& h)
        : rocgraph_local_dnmat(h.m, h.n, h.ld, (T*)h, get_datatype<T>(), h.order)
    {
    }

    ~rocgraph_local_dnmat()
    {
        if(this->descr != nullptr)
            rocgraph_destroy_dnmat_descr(this->descr);
    }

    // Allow rocgraph_local_dnmat to be used anywhere rocgraph_dnmat_descr is expected
    operator rocgraph_dnmat_descr&()
    {
        return this->descr;
    }
    operator const rocgraph_dnmat_descr&() const
    {
        return this->descr;
    }
};

/*  timing: HIP only provides very limited timers function clock() and not general;
            rocgraph sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return
 *  wall time
 */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return
 *  wall time
 */
double get_time_us_sync(hipStream_t stream);

/*! \brief Return path of this executable */
std::string rocgraph_exepath();

/*! \brief Return path where the test data file (rocgraph_test.data) is located */
std::string rocgraph_datapath();

#endif // UTILITY_HPP
