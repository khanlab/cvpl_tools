# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cvpl_tools'
copyright = '2024, KarlHanUW'
author = 'KarlHanUW'
release = '0.6.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']


# -- Options for autodoc -----------------------------------------------------


import sys
sys.path.insert(0, '../../src')
# autodoc requires actually importing the code
# This workaround is mentioned in
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
autodoc_mock_imports = [
    "matplotlib",
    "napari",
]
