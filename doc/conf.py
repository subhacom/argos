# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# NOTES:

# It is better to create latex ``sphinx-build -b latex . latex`` and
# then build PDF using something else, like ``pdflatex``, because
# direct pdf generation with ``sphinx-build -b pdf . pdf`` uses
# rst2pdf which truncates long lines of code. Still the wrapped code
# lines cannot be easily copied to command line because the PDF has a
# special character to indicate new line.

# Build docx : ``sphinx-build -b docx . docx`` This is pretty good as
# the code blocks are wrapped without any special character. However,
# the title, creator and subject properties for docx builder seem to
# be ineffective. Need to edit that manually.


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Argos'
copyright = 'Public Domain'
author = 'Subhasis Ray'

# The full version, including alpha/beta/rc tags
release = '0.1.0'
version = release  # Special for rst2pdf

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'rst2pdf.pdfbuilder',
    'docxbuilder',
]
pdf_documents = [('index', project, u'Argos documentation', u'Subhasis Ray')]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Enable figure numbering
numfig = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = 'images/argos_logo.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# -- Options for latex
latex_logo = 'images/argos_logo.png'

# -- Options for docxbuilder
docx_documents = [
    ('index',
     'Argos.docx',
     {
         'title': project,
         'creator': author,
         'subject': 'Tracking multiple animals in videos',
         'keywords': ['object tracking', 'video processing']
     },
     True),
]
