// Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "handle.h"
#include "control.h"
#include "utility.h"

#include <hip/hip_runtime.h>
/********************************************************************************
 * \brief rocgraph_csrmv_info is a structure holding the rocgraph csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * create_csrmv_info() routine. It should be destroyed at the end
 * using destroy_csrmv_info().
 *******************************************************************************/
rocgraph_status rocgraph::create_csrmv_info(rocgraph_csrmv_info* info)
{
    if(info == nullptr)
    {
        return rocgraph_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *info = new _rocgraph_csrmv_info;
        }
        catch(const rocgraph_status& status)
        {
            return status;
        }
        return rocgraph_status_success;
    }
}

/********************************************************************************
 * \brief Copy csrmv info.
 *******************************************************************************/
rocgraph_status rocgraph::copy_csrmv_info(rocgraph_csrmv_info dest, const rocgraph_csrmv_info src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocgraph_status_invalid_pointer;
    }

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;

    previously_created |= (dest->adaptive.size != 0);
    previously_created |= (dest->adaptive.row_blocks != nullptr);
    previously_created |= (dest->adaptive.wg_flags != nullptr);
    previously_created |= (dest->adaptive.wg_ids != nullptr);

    previously_created |= (dest->lrb.size != 0);
    previously_created |= (dest->lrb.wg_flags != nullptr);
    previously_created |= (dest->lrb.rows_offsets_scratch != nullptr);
    previously_created |= (dest->lrb.rows_bins != nullptr);
    previously_created |= (dest->lrb.n_rows_bins != nullptr);

    previously_created |= (dest->trans != rocgraph_operation_none);
    previously_created |= (dest->m != 0);
    previously_created |= (dest->n != 0);
    previously_created |= (dest->nnz != 0);
    previously_created |= (dest->max_rows != 0);
    previously_created |= (dest->descr != nullptr);
    previously_created |= (dest->csr_row_ptr != nullptr);
    previously_created |= (dest->csr_col_ind != nullptr);
    previously_created |= (dest->index_type_I != rocgraph_indextype_u16);
    previously_created |= (dest->index_type_J != rocgraph_indextype_u16);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->adaptive.size != src->adaptive.size);
        invalid |= (dest->lrb.size != src->lrb.size);
        invalid |= (dest->trans != src->trans);
        invalid |= (dest->m != src->m);
        invalid |= (dest->n != src->n);
        invalid |= (dest->nnz != src->nnz);
        invalid |= (dest->max_rows != src->max_rows);
        invalid |= (dest->index_type_I != src->index_type_I);
        invalid |= (dest->index_type_J != src->index_type_J);

        if(invalid)
        {
            return rocgraph_status_invalid_pointer;
        }
    }

    size_t I_size = sizeof(uint16_t);
    switch(src->index_type_I)
    {
    case rocgraph_indextype_u16:
    {
        I_size = sizeof(uint16_t);
        break;
    }
    case rocgraph_indextype_i32:
    {
        I_size = sizeof(int32_t);
        break;
    }
    case rocgraph_indextype_i64:
    {
        I_size = sizeof(int64_t);
        break;
    }
    }

    size_t J_size = sizeof(uint16_t);
    switch(src->index_type_J)
    {
    case rocgraph_indextype_u16:
    {
        J_size = sizeof(uint16_t);
        break;
    }
    case rocgraph_indextype_i32:
    {
        J_size = sizeof(int32_t);
        break;
    }
    case rocgraph_indextype_i64:
    {
        J_size = sizeof(int64_t);
        break;
    }
    }

    if(src->adaptive.row_blocks != nullptr)
    {
        if(dest->adaptive.row_blocks == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocgraph_hipMalloc((void**)&dest->adaptive.row_blocks,
                                                   I_size * src->adaptive.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->adaptive.row_blocks,
                                      src->adaptive.row_blocks,
                                      I_size * src->adaptive.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->adaptive.wg_flags != nullptr)
    {
        if(dest->adaptive.wg_flags == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocgraph_hipMalloc((void**)&dest->adaptive.wg_flags,
                                                   sizeof(uint32_t) * src->adaptive.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->adaptive.wg_flags,
                                      src->adaptive.wg_flags,
                                      sizeof(uint32_t) * src->adaptive.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->adaptive.wg_ids != nullptr)
    {
        if(dest->adaptive.wg_ids == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocgraph_hipMalloc((void**)&dest->adaptive.wg_ids, J_size * src->adaptive.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->adaptive.wg_ids,
                                      src->adaptive.wg_ids,
                                      J_size * src->adaptive.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->lrb.wg_flags != nullptr)
    {
        if(dest->lrb.wg_flags == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocgraph_hipMalloc((void**)&dest->lrb.wg_flags, sizeof(uint32_t) * src->lrb.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->lrb.wg_flags,
                                      src->lrb.wg_flags,
                                      sizeof(uint32_t) * src->lrb.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->lrb.rows_offsets_scratch != nullptr)
    {
        if(dest->lrb.rows_offsets_scratch == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocgraph_hipMalloc((void**)&dest->lrb.rows_offsets_scratch, J_size * src->m));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->lrb.rows_offsets_scratch,
                                      src->lrb.rows_offsets_scratch,
                                      J_size * src->m,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->lrb.rows_bins != nullptr)
    {
        if(dest->lrb.rows_bins == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocgraph_hipMalloc((void**)&dest->lrb.rows_bins, J_size * src->m));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->lrb.rows_bins, src->lrb.rows_bins, J_size * src->m, hipMemcpyDeviceToDevice));
    }

    if(src->lrb.n_rows_bins != nullptr)
    {
        if(dest->lrb.n_rows_bins == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocgraph_hipMalloc((void**)&dest->lrb.n_rows_bins, J_size * 32));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->lrb.n_rows_bins, src->lrb.n_rows_bins, J_size * 32, hipMemcpyDeviceToDevice));
    }

    dest->adaptive.size = src->adaptive.size;
    dest->lrb.size      = src->lrb.size;
    dest->trans         = src->trans;
    dest->m             = src->m;
    dest->n             = src->n;
    dest->nnz           = src->nnz;
    dest->max_rows      = src->max_rows;
    dest->index_type_I  = src->index_type_I;
    dest->index_type_J  = src->index_type_J;

    // Not owned by the info struct. Just pointers to externally allocated memory
    dest->descr       = src->descr;
    dest->csr_row_ptr = src->csr_row_ptr;
    dest->csr_col_ind = src->csr_col_ind;

    return rocgraph_status_success;
}

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
rocgraph_status rocgraph::destroy_csrmv_info(rocgraph_csrmv_info info)
{
    if(info == nullptr)
    {
        return rocgraph_status_success;
    }

    // Clean up adaptive arrays
    if(info->adaptive.size > 0)
    {
        RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->adaptive.row_blocks));
        RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->adaptive.wg_flags));
        RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->adaptive.wg_ids));
    }

    if(info->lrb.size > 0)
    {
        RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->lrb.wg_flags));
    }

    if(info->m > 0)
    {
        RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->lrb.rows_offsets_scratch));
        RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->lrb.rows_bins));
    }

    RETURN_IF_HIP_ERROR(rocgraph_hipFree(info->lrb.n_rows_bins));

    // Destruct
    try
    {
        delete info;
    }
    catch(const rocgraph_status& status)
    {
        return status;
    }
    return rocgraph_status_success;
}

// Emulate C++17 std::void_t
template <typename...>
using void_t = void;

// By default, use gcnArch converted to a string prepended by gfx
template <typename PROP, typename = void>
struct ArchName
{
    std::string operator()(const PROP* prop) const;
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP* prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop->gcnArchName);
        std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
        return gcnArch;
    }
};

// If gcnArchName not present, no xnack mode
template <typename PROP, typename = void>
struct XnackMode
{
    std::string operator()(const PROP* prop) const
    {
        return "";
    }
};

// If gcnArchName exists as a member, use it
template <typename PROP>
struct XnackMode<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP* prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop->gcnArchName);
        auto        loc = gcnArchName.find("xnack");
        std::string xnackMode;
        if(loc != std::string::npos)
        {
            xnackMode = gcnArchName.substr(loc, 6);
            // guard against missing +/- at end of xnack mode
            if(xnackMode.size() < 6)
                xnackMode = "";
        }
        return xnackMode;
    }
};

std::string rocgraph::handle_get_arch_name(rocgraph_handle handle)
{
    return ArchName<hipDeviceProp_t>{}(handle->get_properties());
}

std::string rocgraph::handle_get_xnack_mode(rocgraph_handle handle)
{
    return XnackMode<hipDeviceProp_t>{}(handle->get_properties());
}

/*******************************************************************************
 * \brief convert hipError_t to rocgraph_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from
 * those calls
 ******************************************************************************/
rocgraph_status rocgraph::get_rocgraph_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess:
        return rocgraph_status_success;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return rocgraph_status_memory_error;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return rocgraph_status_invalid_pointer;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return rocgraph_status_invalid_handle;

    // library using hip incorrectly
    case hipErrorInvalidValue:
        return rocgraph_status_internal_error;

    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default:
        return rocgraph_status_internal_error;
    }
}
