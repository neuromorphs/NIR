# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "NIR"
copyright = "2023, NIR team"
author = "NIR team"


# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_external_toc",
    "sphinxcontrib.mermaid",
    "autoapi.extension",
]

external_toc_path = "_toc.yml"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

autodoc_typehints = "both"
autoapi_type = "python"
autoapi_dirs = ["../../nir"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# MyST settings
nb_execution_mode = "off" # this can be turned to 'auto' once the package is stable
nb_execution_timeout = 300
nb_execution_show_tb = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "NIR documentation"
html_logo = "../logo_light.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "search_bar_text": "Search NIR docs...",
    "repository_url": "https://github.com/neuromorphs/nir",
    "repository_branch": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
}
