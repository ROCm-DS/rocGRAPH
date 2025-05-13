.. meta::
  :description: rocGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _linux-install:

********************************************************************
Installation and building for Linux
********************************************************************

You can install rocGRAPH using the following instructions. There are some prerequisites
that should be installed prior to installing the rocGRAPH library, as described in the
following steps.

Prerequisites
=============

- A ROCm enabled platform. For more information, see `ROCm installation on Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_.

Building rocGRAPH from source
==============================

The following instructions can be used to build rocGRAPH from source files.

Requirements
------------

- `AMD ROCm 6.4.0 or later <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_
- `rocPRIM <https://github.com/ROCm/rocPRIM>`_
- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `GoogleTest <https://github.com/google/googletest>`_ (optional, for clients)

.. note::
   rocGRAPH does not require hipcc and is tested against other compilers like
   g++. It does require the libraries in a ROCm installation.

Download rocGRAPH
-------------------

The rocGRAPH source code is available at the `rocGRAPH GitHub page <https://github.com/ROCm-DS/rocGRAPH>`_.
Download the source code using the following commands:

.. code:: bash

  $ git clone https://github.com/ROCm-DS/rocGRAPH.git
  $ cd rocGRAPH

Using ``install.sh`` to build rocGRAPH with dependencies and clients
----------------------------------------------------------------------

It is recommended to install rocGRAPH using the ``install.sh`` script.
The following table lists common uses of ``install.sh`` to build the library, its dependencies, and clients.
Clients contain example code, unit tests and benchmarks.

.. list-table::
    :widths: 3, 9

    * - **Command**
      - **Description**

    * - ``./install.sh -h``
      - Print help information.

    * - ``./install.sh``
      - Build library in your local directory. It is assumed that all dependencies are available.

    * - ``./install.sh -d``
      - Build library and its dependencies in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.

    * - ``./install.sh -c``
      - Build library, and client in your local directory. It is assumed dependencies are available.

    * - ``./install.sh -dc``
      - Build library, dependencies, and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.

    * - ``./install.sh -i``
      - Build library, then build and install rocGRAPH package in `/opt/rocm/rocgraph`. You will be prompted for sudo access. This will install for all users.

    * - ``./install.sh -ic``
      - Build library, and client, then build and install rocGRAPH package in `opt/rocm/rocgraph`. You will be prompted for sudo access. This will install for all users.

    * - ``./install.sh -idc``
      - Build library, dependencies, and client, then build and install rocGRAPH package in `/opt/rocm/rocgraph`. You will be prompted for sudo access. This will install for all users.

.. note::
  You can also use the ``-a`` option on any of the preceding command to build the library for a supported architecture as listed in :ref:`supported-targets`. For example:

  ``./install.sh -i -a gfx908``

Building rocGRAPH using individual commands
--------------------------------------------

CMake 3.5 or later is required in order to build rocGRAPH.
The rocGRAPH library contains both host and device code, therefore the HIP compiler must be specified during the cmake configuration process.

.. code:: bash

  # Create and change to build directory
  $ mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ CXX=/opt/rocm/bin/hipcc cmake ../..

  # Compile rocGRAPH library
  $ make -j$(nproc)

  # Install rocGRAPH to /opt/rocm
  $ sudo make install

Building rocGRAPH with dependencies and clients
-----------------------------------------------

.. note::
  GoogleTest is required to build rocGRAPH clients.

.. code:: bash

  # Install GoogleTest
  $ mkdir -p build/release/deps ; cd build/release/deps
  $ cmake ../../../deps
  $ make -j$(nproc) install

  # Change to build directory
  $ cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ CXX=/opt/rocm/bin/hipcc cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                        -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocGRAPH library
  $ make -j$(nproc)

  # Install rocGRAPH to /opt/rocm
  $ sudo make install

Test the installation
---------------------

You can test the installation by running one of the rocGRAPH examples after successfully compiling the library with clients.

.. code:: bash

   # Navigate to clients binary directory
   $ cd rocGRAPH/build/release/clients/staging

   # Execute rocGRAPH example
   $ ./example_csrmv 1000

.. _supported-targets:

Supported Targets
=================

Currently, rocGRAPH is supported under the following operating systems

- `Ubuntu 16.04 <https://ubuntu.com/>`_
- `Ubuntu 18.04 <https://ubuntu.com/>`_
- `CentOS 7 <https://www.centos.org/>`_
- `SLES 15 <https://www.suse.com/solutions/enterprise-linux/>`_

To compile and run rocGRAPH, `AMD ROCm Platform <https://github.com/ROCm/ROCm>`_ is required.

The following HIP capable devices are currently supported

- gfx906 (e.g. Vega20, MI50, MI60)
- gfx908
- gfx90a (e.g. MI200)
- gfx940
- gfx941
- gfx942
- gfx1030 (e.g. Navi21)
- gfx1100 (e.g. Navi31)
- gfx1101 (e.g. Navi32)
- gfx1102 (e.g. Navi33)

Common build problems
=====================

#. **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names: ROCMConfig.cmake, rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/ROCm/rocm-cmake>`_
