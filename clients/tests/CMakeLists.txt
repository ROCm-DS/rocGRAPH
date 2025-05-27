# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set(old_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

include(FetchContent)
set(GTEST_DEPENDENCIES)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

set(BUILD_SHARED_LIBS ${old_BUILD_SHARED_LIBS})

# MEMSTAT ?
if(BUILD_MEMSTAT)
  add_compile_options(-DROCGRAPH_WITH_MEMSTAT)
endif()

if(BUILD_ROCGRAPH_ILP64)
  add_compile_options(-Drocgraph_ILP64)
endif()

if(DOWNLOAD_CLIENTS_TEST_MATRICES)
  #
  # Client matrices.
  #
  if(NOT EXISTS "${CMAKE_MATRICES_DIR}")
    #
    # Download.
    #
    set(CMAKE_MATRICES_DIR
        ${PROJECT_BINARY_DIR}/matrices
        CACHE STRING "Matrices directory.")
    file(MAKE_DIRECTORY ${CMAKE_MATRICES_DIR})

    if(NOT TARGET rocgraph)
      set(CONVERT_SOURCE
          ${CMAKE_SOURCE_DIR}/../deps/convert.cpp
          CACHE STRING "Convert tool mtx2csr.")
      include(${CMAKE_SOURCE_DIR}/../cmake/ClientMatrices.cmake)
    else()
      set(CONVERT_SOURCE
          ${CMAKE_SOURCE_DIR}/deps/convert.cpp
          CACHE STRING "Convert tool mtx2csr.")
      include(${CMAKE_SOURCE_DIR}/cmake/ClientMatrices.cmake)
    endif()

  else()
    #
    # Copy.
    #
    if(NOT CMAKE_MATRICES_DIR STREQUAL "${PROJECT_BINARY_DIR}/matrices")
      message(
        "Copy matrix files from ${CMAKE_MATRICES_DIR} to ${PROJECT_BINARY_DIR}/matrices"
      )

      execute_process(
        COMMAND cp -r ${CMAKE_MATRICES_DIR} ${PROJECT_BINARY_DIR}/matrices
        RESULT_VARIABLE STATUS
        WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})

      if(STATUS AND NOT STATUS EQUAL 0)
        message(FATAL_ERROR "Failed to copy matrix .csr files, aborting.")
      endif()
    endif()

  endif()
endif()

set(ROCGRAPH_TEST_SOURCES
    test_rocgraph_balanced_cut_clustering.cpp
    test_rocgraph_betweenness_centrality.cpp
    test_rocgraph_bfs.cpp
    test_rocgraph_core_number.cpp
    test_rocgraph_ecg.cpp
    test_rocgraph_edge_betweenness_centrality.cpp
    test_rocgraph_eigenvector_centrality.cpp
    test_rocgraph_extract_ego.cpp
    test_rocgraph_extract_induced_subgraph.cpp
    test_rocgraph_extract_paths.cpp
    test_rocgraph_generate_rmat_edgelist.cpp
    test_rocgraph_hits.cpp
    test_rocgraph_jaccard_coefficients.cpp
    test_rocgraph_k_core.cpp
    test_rocgraph_k_truss_subgraph.cpp
    test_rocgraph_katz_centrality.cpp
    test_rocgraph_legacy_ecg.cpp
    test_rocgraph_leiden.cpp
    test_rocgraph_louvain.cpp
    test_rocgraph_node2vec.cpp
    test_rocgraph_out_degrees.cpp
    test_rocgraph_overlap_coefficients.cpp
    test_rocgraph_pagerank.cpp
    test_rocgraph_sg_graph_create.cpp
    test_rocgraph_sorensen_coefficients.cpp
    test_rocgraph_spectral_modularity_maximization.cpp
    test_rocgraph_sssp.cpp
    test_rocgraph_strongly_connected_components.cpp
    test_rocgraph_triangle_count.cpp
    test_rocgraph_two_hop_neighbors.cpp
    test_rocgraph_uniform_neighbor_sample.cpp
    test_rocgraph_uniform_random_walks.cpp
    test_rocgraph_weakly_connected_components.cpp)

set(ROCGRAPH_CLIENTS_TESTINGS
    ../testings/testing_rocgraph_balanced_cut_clustering.cpp
    ../testings/testing_rocgraph_betweenness_centrality.cpp
    ../testings/testing_rocgraph_bfs.cpp
    ../testings/testing_rocgraph_core_number.cpp
    ../testings/testing_rocgraph_ecg.cpp
    ../testings/testing_rocgraph_edge_betweenness_centrality.cpp
    ../testings/testing_rocgraph_eigenvector_centrality.cpp
    ../testings/testing_rocgraph_extract_ego.cpp
    ../testings/testing_rocgraph_extract_induced_subgraph.cpp
    ../testings/testing_rocgraph_extract_paths.cpp
    ../testings/testing_rocgraph_generate_rmat_edgelist.cpp
    ../testings/testing_rocgraph_hits.cpp
    ../testings/testing_rocgraph_jaccard_coefficients.cpp
    ../testings/testing_rocgraph_k_core.cpp
    ../testings/testing_rocgraph_k_truss_subgraph.cpp
    ../testings/testing_rocgraph_katz_centrality.cpp
    ../testings/testing_rocgraph_legacy_ecg.cpp
    ../testings/testing_rocgraph_leiden.cpp
    ../testings/testing_rocgraph_louvain.cpp
    ../testings/testing_rocgraph_node2vec.cpp
    ../testings/testing_rocgraph_out_degrees.cpp
    ../testings/testing_rocgraph_overlap_coefficients.cpp
    ../testings/testing_rocgraph_pagerank.cpp
    ../testings/testing_rocgraph_sg_graph_create.cpp
    ../testings/testing_rocgraph_sorensen_coefficients.cpp
    ../testings/testing_rocgraph_spectral_modularity_maximization.cpp
    ../testings/testing_rocgraph_sssp.cpp
    ../testings/testing_rocgraph_strongly_connected_components.cpp
    ../testings/testing_rocgraph_triangle_count.cpp
    ../testings/testing_rocgraph_two_hop_neighbors.cpp
    ../testings/testing_rocgraph_uniform_neighbor_sample.cpp
    ../testings/testing_rocgraph_uniform_random_walks.cpp
    ../testings/testing_rocgraph_weakly_connected_components.cpp)

set(ROCGRAPH_CLIENTS_COMMON
    ../common/utility.cpp
    ../common/rocgraph_random.cpp
    ../common/rocgraph_status_to_string.cpp
    ../common/rocgraph_clients_are_near_tolerance.cpp
    ../common/rocgraph_clients_expect_array_near_tolerance.cpp
    ../common/rocgraph_clients_expect_near_tolerance.cpp
    ../common/rocgraph_clients_expect_indirect_array_near_tolerance.cpp
    ../common/rocgraph_clients_are_eq.cpp
    ../common/rocgraph_clients_are_le.cpp
    ../common/rocgraph_clients_are_lt.cpp
    ../common/rocgraph_clients_are_gt.cpp
    ../common/rocgraph_clients_expect_array_eq.cpp
    ../common/rocgraph_clients_expect_indirect_array_eq.cpp
    ../common/rocgraph_clients_expect_eq.cpp
    ../common/rocgraph_clients_expect_ne.cpp
    ../common/rocgraph_clients_expect_le.cpp
    ../common/rocgraph_clients_expect_lt.cpp
    ../common/rocgraph_clients_expect_gt.cpp
    ../common/rocgraph_clients_expect_status.cpp
    ../common/rocgraph_clients_expect_success.cpp
    ../common/rocgraph_clients_expect_success_error.cpp
    ../common/rocgraph_clients_expect_hip_success.cpp
    ../common/rocgraph_clients_near_check.cpp
    ../common/rocgraph_clients_unit_check.cpp
    ../common/rocgraph_clients_create_test_graph.cpp
    ../common/rocgraph_clients_data_type_id_get.cpp
    ../common/rocgraph_parse_data.cpp
    ../common/rocgraph_enum.cpp
    ../common/rocgraph_init.cpp
    ../common/rocgraph_host.cpp
    ../common/rocgraph_vector_utils.cpp
    ../common/rocgraph_matrix_factory.cpp
    ../common/rocgraph_matrix_factory_laplace2d.cpp
    ../common/rocgraph_matrix_factory_laplace3d.cpp
    ../common/rocgraph_matrix_factory_zero.cpp
    ../common/rocgraph_matrix_factory_random.cpp
    ../common/rocgraph_matrix_factory_tridiagonal.cpp
    ../common/rocgraph_matrix_factory_pentadiagonal.cpp
    ../common/rocgraph_matrix_factory_file.cpp
    ../common/rocgraph_exporter_rocalution.cpp
    ../common/rocgraph_exporter_matrixmarket.cpp
    ../common/rocgraph_exporter_ascii.cpp
    ../common/rocgraph_type_conversion.cpp
    ../common/rocgraph_importer_rocalution.cpp
    ../common/rocgraph_importer_matrixmarket.cpp
    ../common/rocgraph_importer_mlcsr.cpp
    ../common/rocgraph_clients_envariables.cpp
    ../common/rocgraph_clients_matrices_dir.cpp)

add_executable(
  rocgraph-test rocgraph_test_main.cpp ${ROCGRAPH_TEST_SOURCES}
                ${ROCGRAPH_CLIENTS_COMMON} ${ROCGRAPH_CLIENTS_TESTINGS})

# Set GOOGLE_TEST definition
target_compile_definitions(rocgraph-test PRIVATE GOOGLE_TEST)

# Target compile options
target_compile_options(
  rocgraph-test PRIVATE -ffp-contract=on -mfma -Wno-deprecated
                        -Wno-unused-command-line-argument -Wall)

# Internal common header
target_include_directories(
  rocgraph-test
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# Target link libraries
target_link_libraries(rocgraph-test PRIVATE GTest::gtest_main roc::rocgraph
                                            hip::host hip::device)

# Add OpenMP if available
if(OPENMP_FOUND)
  target_link_libraries(rocgraph-test PRIVATE OpenMP::OpenMP_CXX
                                              -Wl,-rpath=${HIP_CLANG_ROOT}/lib)
endif()

# Set test output directory
set_target_properties(rocgraph-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS rocgraph-test COMPONENT tests)

string(REGEX REPLACE ".cpp" ".yaml" ROCGRAPH_TEST_YAMLS
                     "${ROCGRAPH_TEST_SOURCES}")

# Prepare testing data
set(ROCGRAPH_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocgraph_test.data")
add_custom_command(
  OUTPUT "${ROCGRAPH_TEST_DATA}"
  COMMAND ${python} ../common/rocgraph_gentest.py -I ../include
          rocgraph_test.yaml -o "${ROCGRAPH_TEST_DATA}"
  DEPENDS ../common/rocgraph_gentest.py rocgraph_test.yaml
          ../include/rocgraph_common.yaml known_bugs.yaml ${ROCGRAPH_TEST_YAMLS}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(rocgraph-test-data DEPENDS "${ROCGRAPH_TEST_DATA}")

add_dependencies(rocgraph-test rocgraph-test-data rocgraph-common)

if(BUILD_CODE_COVERAGE)
  add_custom_target(
    code_cov_tests
    DEPENDS rocgraph-test
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/rocgraph-coverage_%m.profraw" $<TARGET_FILE:rocgraph-test>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm-bin
    PATHS /opt/rocm/llvm/bin
  )

    find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/rocgraph-coverage_*.profraw -o ./coverage-report/rocgraph.profdata
    COMMAND ${LLVM_COV} report -object ./library/src/librocgraph.so -instr-profile=./coverage-report/rocgraph.profdata
    COMMAND ${LLVM_COV} show -object ./library/src/librocgraph.so -instr-profile=./coverage-report/rocgraph.profdata -format=html -output-dir=coverage-report
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

rocm_install(FILES ${ROCGRAPH_TEST_DATA} DESTINATION
             "${CMAKE_INSTALL_DATADIR}/rocgraph/test" COMPONENT tests)
