# Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Client packaging
include(CMakeParseArguments)

function(rocm_create_package_clients)
  set(options)
  set(oneValueArgs LIB_NAME DESCRIPTION SECTION MAINTAINER VERSION)
  set(multiValueArgs DEPENDS)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  string(CONCAT PACKAGE_NAME ${PARSE_LIB_NAME} "-clients-" ${PARSE_VERSION}
                "-Linux.deb")
  string(
    CONCAT DEB_CONTROL_FILE_CONTENT
           "Package: "
           ${PARSE_LIB_NAME}
           "-clients"
           "\nVersion: "
           ${PARSE_VERSION}
           "\nSection: "
           ${PARSE_SECTION}
           "\nPriority: optional"
           "\nArchitecture: amd64"
           "\nMaintainer: "
           ${PARSE_MAINTAINER}
           "\nDescription: "
           ${PARSE_DESCRIPTION}
           "\nDepends: "
           ${PARSE_LIB_NAME}
           "(>="
           ${PARSE_VERSION}
           ")\n\n")

  if(EXISTS "${PROJECT_BINARY_DIR}/package")
    file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/package")
  endif()
  file(MAKE_DIRECTORY
       "${PROJECT_BINARY_DIR}/package/${ROCM_PATH}/bin/${PARSE_LIB_NAME}")
  file(WRITE "${PROJECT_BINARY_DIR}/package/DEBIAN/control"
       ${DEB_CONTROL_FILE_CONTENT})

  add_custom_target(
    package_clients
    COMMAND ${CMAKE_COMMAND} -E remove -f
            "${PROJECT_BINARY_DIR}/package/${ROCM_PATH}/bin/${PARSE_LIB_NAME}/*"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/*"
            "${PROJECT_BINARY_DIR}/package/${ROCM_PATH}/bin/${PARSE_LIB_NAME}/"
    COMMAND dpkg -b "${PROJECT_BINARY_DIR}/package/" ${PACKAGE_NAME})
endfunction(rocm_create_package_clients)
