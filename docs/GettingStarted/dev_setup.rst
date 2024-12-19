.. _dev_setup:

Dev Setup
#########

This page include notes for continued development of the library, on aspects such as package
management, documentations and unit testing.

Poetry
******

Poetry is used for managing pip packages and the build and publish of :code:`cvpl_tools`
into PyPI. However, I find Poetry slow adding dependencies as it
needs to search over versions of packages in resolving dependencies and generating lock file.
In local development, I use an Anaconda environment to install the same list of
packages in :code:`pyproject.toml`

Building and publishing the package to PyPI is done with the appropriate setup followed by running
:code:`poetry build` and :code:`poetry publish`.

The steps is as follows:

1. Modify your codebase
2. Modify the version string in :code:`pyproject.toml`
3. Run :code:`poetry build` then :code:`poetry publish`

Documentations
**************

The documentation of :code:`cvpl_tools` has two aspects: within-source and on-line. In the source code, the
library uses the convention that every API function includes a Python
`docstring <https://peps.python.org/pep-0257/>`_ in
`Google Style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ and also include
comments when suitable at the start of a source code file.

Related to Numpy: Numpy has a :code:`numpy.typing` module which allows documenting the APIs with Numpy
typing hint such as :code:`np.typing.NDArray[np.float32]` representing an array of type float. This module
requires :code:`mypy` package, with configuration file at "cvpl_tools/mypy.ini".

The on-line documentation locates at "cvpl_tools/docs" folder, which is a sphinx documentation that can be
built using the command:

.. code-block:: Bash

    cd docs
    sphinx-build -M html . ./_build

The "docs" folder includes a "conf.py" file which imports the source code under "src" to support the
"autoclass" and "autofunction" directives used in the documentations.

The diagrams in "docs/assets" are generated using the diagram editor Dia, which is an open source software
good for drawing programming diagrams.

Unit Testing
************

:code:`pytest` is used for unit testing. Run unit tests with :code:`pytest test` command in the root
directory. Configuration file is at "cvpl_tools/pytest.ini".

Misc
****

- I use PyCharm as my development IDE
- The on-line documentation is hosted by Github Pages, which you can see the setting on GitHub by first go to
  the `main GitHub page <https://github.com/khanlab/cvpl_tools>`_ then visit Settings > Pages.
