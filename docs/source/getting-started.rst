Get Started with PyARPES
========================

Checking your installation
--------------------------

Some features in PyARPES require libraries that are not installed by
default, either because they are heavy dependencies we don’t want to
force on users, or there are possible issues of platform compatibility.

You can check whether your installation in a Python session or in
Jupyter

.. code:: python

   import arpes
   arpes.check()

You should see something like this depending on the state of your
optional dependencies:


.. only:: html or singlehtml
.. code:: text

   [x] Igor Pro Support:
       For Igor support, install igorpy with: 
       pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1
   [✔] qt_tool Support

Loading example data
--------------------

At this point, you should be able to load the example data, an ARPES
spectrum of the topological insulator bismuth selenide:

.. code:: python

   import arpes.config
   from arpes.io import load_example_data
   load_example_data()

Loading your own data
---------------------

If you have the path to a piece of data you want to load as well as the
data source it comes from (see the section on
:doc:`plugins </writing-plugins>` for more detail), you can load it with
``arpes.io.load_without_dataset``:

.. code:: python

   import arpes.config
   from arpes.io import load_data
   load_data('epath/to/my/data.h5', location='ALS-BL7')

What’s next?
------------

With the example data in hand, you can jump into the rest of the
examples on the site. If you’re a visual learner or are new to Jupyter
and are running into issues, have a look at the :doc:`tutorial videos </example-videos>`.
Another good place to start is on the
section for :doc:`exploration <notebooks/basic-data-exploration>` of ARPES data.
