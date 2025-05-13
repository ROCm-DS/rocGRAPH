// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "c_api/rocgraph_array.hpp"

#include "c_api/rocgraph_error.hpp"
#include "c_api/rocgraph_handle.hpp"

#include "internal/types/rocgraph_byte_t.h"
#include "internal/types/rocgraph_handle_t.h"
#include "internal/types/rocgraph_type_erased_host_array_t.h"

#include "internal/aux/rocgraph_type_erased_device_array_aux.h"
#include "internal/aux/rocgraph_type_erased_device_array_view_aux.h"
#include "internal/aux/rocgraph_type_erased_host_array_aux.h"
#include "internal/aux/rocgraph_type_erased_host_array_view_aux.h"

namespace rocgraph
{
    namespace c_api
    {

        size_t data_type_sz[] = {4, 8, 4, 8, 8};

    } // namespace c_api
} // namespace rocgraph

extern "C" rocgraph_status rocgraph_type_erased_device_array_create_from_view(
    const rocgraph_handle_t*                        handle,
    const rocgraph_type_erased_device_array_view_t* view,
    rocgraph_type_erased_device_array_t**           array,
    rocgraph_error_t**                              error)
{
    *array = nullptr;
    *error = nullptr;

    try
    {
        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        auto p_handle = handle;
        auto internal_pointer
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                view);

        size_t n_bytes
            = internal_pointer->size_ * (rocgraph::c_api::data_type_sz[internal_pointer->type_]);

        auto ret_value = new rocgraph::c_api::rocgraph_type_erased_device_array_t(
            internal_pointer->size_,
            n_bytes,
            internal_pointer->type_,
            p_handle->get_raft_handle()->get_stream());

        raft::copy(reinterpret_cast<rocgraph_byte_t*>(ret_value->data_.data()),
                   reinterpret_cast<rocgraph_byte_t const*>(internal_pointer->data_),
                   internal_pointer->num_bytes(),
                   p_handle->get_raft_handle()->get_stream());

        *array = reinterpret_cast<rocgraph_type_erased_device_array_t*>(ret_value);
        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        *error
            = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{ex.what()});
        return rocgraph_status_unknown_error;
    }
}

extern "C" rocgraph_status
    rocgraph_type_erased_device_array_create(const rocgraph_handle_t*              handle,
                                             size_t                                n_elems,
                                             rocgraph_data_type_id                 dtype,
                                             rocgraph_type_erased_device_array_t** array,
                                             rocgraph_error_t**                    error)
{
    *array = nullptr;
    *error = nullptr;

    try
    {
        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        auto p_handle = handle;

        size_t n_bytes = n_elems * (rocgraph::c_api::data_type_sz[dtype]);

        auto ret_value = new rocgraph::c_api::rocgraph_type_erased_device_array_t(
            n_elems, n_bytes, dtype, p_handle->get_raft_handle()->get_stream());

        *array = reinterpret_cast<rocgraph_type_erased_device_array_t*>(ret_value);
        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        *error
            = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{ex.what()});
        return rocgraph_status_unknown_error;
    }
}

extern "C" void rocgraph_type_erased_device_array_free(rocgraph_type_erased_device_array_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(p);
    delete internal_pointer;
}

#if 0
// NOTE:  This can't work.  rmm::device_buffer doesn't support release, that would leave a raw
//        pointer in the wild with no idea how to free it.  I suppose that could be done
//        (I imagine you can do that with unique_ptr), but it's not currently supported and I'm
//        not sure *this* use case is sufficient justification to adding a potentially
//        dangerous feature.
extern "C" void* rocgraph_type_erased_device_array_release(rocgraph_type_erased_device_array_t* p)
{
  auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(p);
  return internal_pointer->data_.release();
}
#endif

extern "C" rocgraph_type_erased_device_array_view_t*
    rocgraph_type_erased_device_array_view(rocgraph_type_erased_device_array_t* array)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(array);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(internal_pointer->view());
}

rocgraph_type_erased_device_array_view_t* rocgraph_type_erased_device_array_view_create(
    void* pointer, size_t n_elems, rocgraph_data_type_id dtype)
{
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        new rocgraph::c_api::rocgraph_type_erased_device_array_view_t{
            pointer, n_elems, n_elems * (rocgraph::c_api::data_type_sz[dtype]), dtype});
}

extern "C" void
    rocgraph_type_erased_device_array_view_free(rocgraph_type_erased_device_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t*>(p);
    delete internal_pointer;
}

extern "C" size_t
    rocgraph_type_erased_device_array_view_size(const rocgraph_type_erased_device_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(p);
    return internal_pointer->size_;
}

extern "C" rocgraph_data_type_id
    rocgraph_type_erased_device_array_view_type(const rocgraph_type_erased_device_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(p);
    return internal_pointer->type_;
}

extern "C" const void* rocgraph_type_erased_device_array_view_pointer(
    const rocgraph_type_erased_device_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(p);
    return internal_pointer->data_;
}

extern "C" rocgraph_status
    rocgraph_type_erased_host_array_create(const rocgraph_handle_t*            handle,
                                           size_t                              n_elems,
                                           rocgraph_data_type_id               dtype,
                                           rocgraph_type_erased_host_array_t** array,
                                           rocgraph_error_t**                  error)
{
    *array = nullptr;
    *error = nullptr;

    try
    {
        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        auto p_handle = handle;

        size_t n_bytes = n_elems * (rocgraph::c_api::data_type_sz[dtype]);

        *array = reinterpret_cast<rocgraph_type_erased_host_array_t*>(
            new rocgraph::c_api::rocgraph_type_erased_host_array_t{n_elems, n_bytes, dtype});

        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        auto tmp_error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
        *error         = reinterpret_cast<rocgraph_error_t*>(tmp_error);
        return rocgraph_status_unknown_error;
    }
}

extern "C" void rocgraph_type_erased_host_array_free(rocgraph_type_erased_host_array_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_t*>(p);
    delete internal_pointer;
}

#if 0
// Leaving this one out since we're not doing the more important device version
extern "C" void* rocgraph_type_erased_host_array_release(const rocgraph_type_erased_host_array_t* p)
{
  auto internal_pointer = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_t*>(p);
  return internal_pointer->data_.release();
}
#endif

extern "C" rocgraph_type_erased_host_array_view_t*
    rocgraph_type_erased_host_array_view(rocgraph_type_erased_host_array_t* array)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_t*>(array);
    return reinterpret_cast<rocgraph_type_erased_host_array_view_t*>(internal_pointer->view());
}

extern "C" rocgraph_type_erased_host_array_view_t* rocgraph_type_erased_host_array_view_create(
    void* pointer, size_t n_elems, rocgraph_data_type_id dtype)
{
    return reinterpret_cast<rocgraph_type_erased_host_array_view_t*>(
        new rocgraph::c_api::rocgraph_type_erased_host_array_view_t{
            static_cast<std::byte*>(pointer),
            n_elems,
            n_elems * (rocgraph::c_api::data_type_sz[dtype]),
            dtype});
}

extern "C" void rocgraph_type_erased_host_array_view_free(rocgraph_type_erased_host_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t*>(p);
    delete internal_pointer;
}

extern "C" size_t
    rocgraph_type_erased_host_array_size(const rocgraph_type_erased_host_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*>(p);
    return internal_pointer->size_;
}

extern "C" rocgraph_data_type_id
    rocgraph_type_erased_host_array_view_type(const rocgraph_type_erased_host_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*>(p);
    return internal_pointer->type_;
}

extern "C" void*
    rocgraph_type_erased_host_array_pointer(const rocgraph_type_erased_host_array_view_t* p)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*>(p);
    return internal_pointer->data_;
}

extern "C" rocgraph_status
    rocgraph_type_erased_host_array_view_copy(const rocgraph_handle_t*                      handle,
                                              rocgraph_type_erased_host_array_view_t*       dst,
                                              const rocgraph_type_erased_host_array_view_t* src,
                                              rocgraph_error_t**                            error)
{
    *error = nullptr;

    try
    {
        auto p_handle = handle;
        auto internal_pointer_dst
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t*>(dst);
        auto internal_pointer_src
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_host_array_view_t const*>(src);

        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        if(internal_pointer_src->num_bytes() != internal_pointer_dst->num_bytes())
        {
            *error = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{
                "source and destination arrays are different sizes"});
            return rocgraph_status_invalid_input;
        }

        raft::copy(reinterpret_cast<rocgraph_byte_t*>(internal_pointer_dst->data_),
                   reinterpret_cast<rocgraph_byte_t const*>(internal_pointer_src->data_),
                   internal_pointer_src->num_bytes(),
                   p_handle->get_raft_handle()->get_stream());

        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        auto tmp_error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
        *error         = reinterpret_cast<rocgraph_error_t*>(tmp_error);
        return rocgraph_status_unknown_error;
    }
}

extern "C" rocgraph_status rocgraph_type_erased_device_array_view_copy_from_host(
    const rocgraph_handle_t*                  handle,
    rocgraph_type_erased_device_array_view_t* dst,
    const rocgraph_byte_t*                    h_src,
    rocgraph_error_t**                        error)
{
    *error = nullptr;

    try
    {
        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        auto p_handle = handle;
        auto internal_pointer
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t*>(dst);

        raft::update_device(reinterpret_cast<rocgraph_byte_t*>(internal_pointer->data_),
                            h_src,
                            internal_pointer->num_bytes(),
                            p_handle->get_raft_handle()->get_stream());

        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        auto tmp_error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
        *error         = reinterpret_cast<rocgraph_error_t*>(tmp_error);
        return rocgraph_status_unknown_error;
    }
}

extern "C" rocgraph_status rocgraph_type_erased_device_array_view_copy_to_host(
    const rocgraph_handle_t*                        handle,
    rocgraph_byte_t*                                h_dst,
    const rocgraph_type_erased_device_array_view_t* src,
    rocgraph_error_t**                              error)
{
    *error = nullptr;

    try
    {
        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        auto p_handle = handle;
        auto internal_pointer
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                src);

        raft::update_host(h_dst,
                          reinterpret_cast<rocgraph_byte_t const*>(internal_pointer->data_),
                          internal_pointer->num_bytes(),
                          p_handle->get_raft_handle()->get_stream());
        p_handle->get_raft_handle()->sync_stream();

        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        auto tmp_error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
        *error         = reinterpret_cast<rocgraph_error_t*>(tmp_error);
        return rocgraph_status_unknown_error;
    }
}
extern "C" rocgraph_status
    rocgraph_type_erased_device_array_view_copy(const rocgraph_handle_t*                  handle,
                                                rocgraph_type_erased_device_array_view_t* dst,
                                                const rocgraph_type_erased_device_array_view_t* src,
                                                rocgraph_error_t** error)
{
    *error = nullptr;

    try
    {
        auto p_handle = handle;
        auto internal_pointer_dst
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t*>(dst);
        auto internal_pointer_src
            = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(
                src);

        if(!handle)
        {
            *error = reinterpret_cast<rocgraph_error_t*>(
                new rocgraph::c_api::rocgraph_error_t{"invalid resource handle"});
            return rocgraph_status_invalid_handle;
        }

        if(internal_pointer_src->num_bytes() != internal_pointer_dst->num_bytes())
        {
            *error = reinterpret_cast<rocgraph_error_t*>(new rocgraph::c_api::rocgraph_error_t{
                "source and destination arrays are different sizes"});
            return rocgraph_status_invalid_input;
        }

        raft::copy(reinterpret_cast<rocgraph_byte_t*>(internal_pointer_dst->data_),
                   reinterpret_cast<rocgraph_byte_t const*>(internal_pointer_src->data_),
                   internal_pointer_src->num_bytes(),
                   p_handle->get_raft_handle()->get_stream());

        return rocgraph_status_success;
    }
    catch(std::exception const& ex)
    {
        auto tmp_error = new rocgraph::c_api::rocgraph_error_t{ex.what()};
        *error         = reinterpret_cast<rocgraph_error_t*>(tmp_error);
        return rocgraph_status_unknown_error;
    }
}

extern "C" rocgraph_status rocgraph_type_erased_device_array_view_as_type(
    rocgraph_type_erased_device_array_t*       array,
    rocgraph_data_type_id                      dtype,
    rocgraph_type_erased_device_array_view_t** result_view,
    rocgraph_error_t**                         error)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_type_erased_device_array_t*>(array);

    if(rocgraph::c_api::data_type_sz[dtype]
       == rocgraph::c_api::data_type_sz[internal_pointer->type_])
    {
        *result_view = reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
            new rocgraph::c_api::rocgraph_type_erased_device_array_view_t{
                internal_pointer->data_.data(),
                internal_pointer->size_,
                internal_pointer->data_.size(),
                dtype});
        return rocgraph_status_success;
    }
    else
    {
        std::stringstream ss;
        ss << "Could not treat type " << internal_pointer->type_ << " as type " << dtype;
        auto tmp_error = new rocgraph::c_api::rocgraph_error_t{ss.str().c_str()};
        *error         = reinterpret_cast<rocgraph_error_t*>(tmp_error);
        return rocgraph_status_invalid_input;
    }
}
