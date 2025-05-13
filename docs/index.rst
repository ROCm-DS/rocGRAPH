.. meta::
  :description: rocGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _rocgraph:

********************************************************************
rocGRAPH documentation
********************************************************************

.. note::
  rocGRAPH is in an early access state. Running production workloads is not recommended.

The rocGRAPH library includes an extensive set of graph algorithms (e.g. centrality, community,
components, link analysis, link prediction, etc) for operations on graph matrices and vectors,
accessible through C-API functions. It is implemented on top of AMD's ROCm runtime and toolchains,
is created using the HIP programming language, and is optimized for AMD's latest discrete GPUs.

The code is open and hosted at: https://github.com/ROCm-DS/rocGRAPH

The rocGRAPH documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`linux-install`

  .. grid-item-card:: API reference

    * :ref:`rocgraph-reference`

To contribute to the documentation refer to `Contributing to ROCm-DS  <https://rocm.docs.amd.com/projects/ROCm-DS/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/projects/ROCm-DS/latest/about/license.html>`_ page.
