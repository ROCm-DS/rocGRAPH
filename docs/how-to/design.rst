.. meta::
  :description: rocGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _design:

********************
Design Documentation
********************

This document is intended for advanced developers that want to understand, modify or extend the functionality of the rocGRAPH library.

The rocGRAPH library is developed using the `Hourglass API` approach.
This provides a thin C89 API while still having all the convenience of C++.
As a side effect, ABI related binary compatibility issues can be avoided.
Furthermore, this approach allows rocGRAPH routines to be used by other programming languages.

In public API header files, rocGRAPH only relies on functions, pointers, forward declared structs, enumerations and type defs.
rocGRAPH introduces multiple library and object handles by using opaque types to hide layout and implementation details from the user.

Library Source Organization
===========================

The following is the structure of the rocGRAPH library in the GitHub repository.

``library/include/`` directory
------------------------------

The ``library/include`` directory contains all files that are exposed to the user.
The rocGRAPH API, is declared here.

=========================== ===========
File                        Description
=========================== ===========
``rocgraph_c.h``            Includes all other C API related rocGRAPH header files, including C API functions.
``rocgraph.h``              Includes all C API related rocGRAPH header files.
``rocgraph-auxiliary.h``
``rocgraph-functions.h``
``rocgraph-types.h``        Defines all data types used by rocGRAPH.
``rocgraph-version.h.in``   Provides the configured version and settings that is initially set by CMake during compilation.
=========================== ===========

The ``library/include/rocgraph_c`` directory contains all files related to various resource handling and specific graph algorithms

============================= ===========
File                          Description
============================= ===========
``algorithms.h``              Overarching include file that includes most of the following headers.
``array.h``                   Declares data structures for type erased host/device arrays.
``centrality_algorithms.h``   Defines data structures, functions, and algorithm interfaces for PageRank.
``community_algorithms.h``    Defines data structures, functions, and algorithm interfaces for triangle count, louvain, leiden, hierarchical clustering, and ecg algorithms.
``core_algorithms.h``         Defines data structures, functions, and algorithm interfaces for core number and k-core algorithms
``error.h``                   Defines error flag values.
``graph.h``                   Creation of graphs and masks.
``graph_functions.h``         Operations and queries on graphs.
``graph_generators.h``        Functions and data structures to generate random graphs.
``labeling_algorithms.h``     Defines data structures and functions for weakly and strongly connected components algorithms.
``random.h``                  Functions to create and destroy a random number generator state.
``resource_handle.h``         Defines data structures and functions for resource handles that are used to pass information to functions.
``sampling_algorithms.h``     Defines data structures, functions, and algorithm interfaces for uniform neighbor sampling.
``similarity_algorithms.h``   Defines data structures, functions, and algorithm interfaces for the Jaccard, Sorenson, and overlap algorithms.
``traversal_algorithms.h``    Defines data structures, functions, and algorithm interfaces for the BFS and SSSP algorithms.
============================= ===========

``library/src/`` directory
--------------------------

This directory contains all rocGRAPH library source files.
The root of the ``library/src/`` directory hosts the implementation of the library handle and auxiliary functions.
Each sub-directory is responsible for the specific types of graph algorithm or groups of utility/generic functions


``clients/`` directory
----------------------

This directory contains all clients, e.g. samples, unit tests and benchmarks.
Further details are given in :ref:`rocgraph_clients`.

.. _rocgraph_clients:

Clients
=======

rocGRAPH clients host a variety of different examples as well as a unit test and benchmarking package.
For detailed instructions on how to build rocGRAPH with clients, see :ref:`linux-install`.

Unit Tests
----------

Multiple unit tests are available to test for bad arguments, invalid parameters and graph routine functionality.
The unit tests are based on `GoogleTest <https://github.com/google/googletest>`_.
The tests cover all routines that are exposed by the API, including all available floating-point precision.
