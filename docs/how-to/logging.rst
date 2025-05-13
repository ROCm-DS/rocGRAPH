.. meta::
  :description: rocGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, ROCm-DS

.. _rocgraph_logging:

********************************************************************
Activity Logging
********************************************************************

**Note that activity logging is not enabled by default in rocGRAPH**

Four different environment variables can be set to enable logging in rocGRAPH: ``ROCGRAPH_LAYER``, ``ROCGRAPH_LOG_TRACE_PATH``, ``ROCGRAPH_LOG_BENCH_PATH`` and ``ROCGRAPH_LOG_DEBUG_PATH``.

``ROCGRAPH_LAYER`` is a bit mask that enables logging, and where several logging modes can be specified as follows:

================================  =============================================================
``ROCGRAPH_LAYER`` unset          Logging is disabled.
``ROCGRAPH_LAYER`` set to ``1``   Enable trace logging.
``ROCGRAPH_LAYER`` set to ``2``   Enable bench logging.
``ROCGRAPH_LAYER`` set to ``3``   Enable trace and bench logging.
``ROCGRAPH_LAYER`` set to ``4``   Enable debug logging.
``ROCGRAPH_LAYER`` set to ``5``   Enable trace and debug logging.
``ROCGRAPH_LAYER`` set to ``6``   Enable bench and debug logging.
``ROCGRAPH_LAYER`` set to ``7``   Enable trace, bench, and debug logging.
================================  =============================================================

.. note::

    Performance will degrade when logging is enabled. By default, the environment variable ``ROCGRAPH_LAYER`` is unset and logging is disabled.

When logging is enabled, each rocGRAPH function call will write the function name and function arguments to the logging stream. The default logging output is streamed to ``stderr``.
To capture activity logging in a file, set the following environment variables as needed:

  * ``ROCGRAPH_LOG_TRACE_PATH`` specifies a path and file name to capture trace logging streamed to that file
  * ``ROCGRAPH_LOG_BENCH_PATH`` specifies a path and file name to capture bench logging
  * ``ROCGRAPH_LOG_DEBUG_PATH`` specifies a path and file name to capture debug logging

.. note::

    If the file cannot be opened, logging output is streamed to ``stderr``.
