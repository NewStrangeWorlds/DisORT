# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'DisORT++'
copyright = '2025, Daniel Kitzmann'
author = 'Daniel Kitzmann'
release = '2.1.3'
version = '2.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'titles_only': False,
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
}

latex_documents = [
    ('index', 'DisORTpp.tex', 'DisORT++ Documentation',
     'Daniel Kitzmann', 'manual'),
]
