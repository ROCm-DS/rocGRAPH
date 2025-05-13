// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "rocgraph_clients_matrices_dir.hpp"
#include "rocgraph_clients_envariables.hpp"
#include "utility.hpp"

//
//
//
struct clients_matrices_dir
{
private:
    std::string m_path{};
    std::string m_default_path{};
    bool        m_is_defined{};
    clients_matrices_dir()
    {
        m_is_defined
            = rocgraph_clients_envariables::is_defined(rocgraph_clients_envariables::MATRICES_DIR);
        if(m_is_defined)
        {
            m_path = rocgraph_clients_envariables::get(rocgraph_clients_envariables::MATRICES_DIR);
            if(m_path.size() > 0)
            {
                m_path += "/";
            }
        }
        m_default_path = rocgraph_exepath();
        m_default_path += "/../matrices/";
    }

public:
    static clients_matrices_dir& instance()
    {
        static clients_matrices_dir self;
        return self;
    }
    static const std::string& path(bool use_default)
    {
        const clients_matrices_dir& self = instance();
        if(self.m_is_defined)
        {
            return self.m_path;
        }
        else
        {
            if(use_default)
            {
                return self.m_default_path;
            }
            else
            {
                return self.m_path;
            }
        }
    }

    static const std::string& default_path()
    {
        const clients_matrices_dir& self = instance();
        return self.m_default_path;
    }

    static void set(const std::string& p)
    {
        clients_matrices_dir& self = instance();
        self.m_is_defined          = true;
        self.m_path                = p;
    }
};

//
//
//
const char* rocgraph_clients_matrices_dir_get(bool use_default_path)
{
    return clients_matrices_dir::path(use_default_path).c_str();
}

//
//
//
void rocgraph_clients_matrices_dir_set(const char* path)
{
    clients_matrices_dir::set(std::string(path) + "/");
}
