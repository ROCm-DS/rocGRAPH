# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import re

from rocm_docs import ROCmDocs

# import rocgraph


with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r" VERSION\s+\"?([0-9.]+)[^0-9.]+", f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
# Temporary "fix" for Python version numbering
version_number = "1.0.0b1"
left_nav_title = f"rocGRAPH {version_number} documentation"

# for PDF output on Read the Docs
project = "rocGRAPH"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number
cpp_maximum_signature_line_length = 10
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-ds"}

extensions = [
    "rocm_docs",
    "breathe",
    "sphinx.ext.autodoc",
]

external_toc_path = "./sphinx/_toc.yml"
doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}

external_projects_current_project = "rocgraph"
