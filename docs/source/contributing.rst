How to contribute to PyARPES
============================

We absolutely welcome the support and partnership of users that want to
contribute to PyARPES! If you just want to add a particular analysis
routine, provide a patch for a bug, or suggest a documentation change,
the best way to contribute is to submit a pull request. Even submitting
an issue is a substantial help as it lets us know what might be useful
for others to see changed in the software. Generally speaking, you
should verify or check the need for a new feature by first opening `an
issue <https://gitlab.com/lanzara-group/python-arpes/issues>`__.

If you are looking for low hanging fruit, we are acutely aware of some
current shortcomings:

1. Better/more complete documentation and docstrings
2. More example data for new users
3. More complete testing
4. Example analysis notebooks to help new users acclimate

Additionally, these all represent great ways to learn more about the
software as a user.

If it makes sense, we will consider adding users as developers on `the
repo <https://gitlab.com/lanzara-group/python-arpes>`__.

What you’ll need
----------------

Here’s a summary of what you’ll need to do, if you’are already familiar
with contributing to open source. If you are less familiar, much more
detail on this is described in the :doc:`developer’s guide </dev-guide>`.

1.  You will need a git client, if you don’t want to use a terminal,
    have a look at Github’s `GUI Client <https://desktop.github.com/>`__
2.  :doc:`Install an editable copy of PyARPES </dev-guide>`
3.  Write your new analysis code, bug patch, documentation, etc.
4.  Put it someplace reasonable in line with the project’s
    organizational principles
5.  Add convenience accessors on ``.T``, ``.S``, or ``.F`` if relelvant
6.  Make sure the new code is adequately documented with a
    `docstring <https://en.wikipedia.org/wiki/Docstring#Python>`__.
7.  Add documentation to this documentation site if relevant, see below
    for details
8.  Check that tests still pass and add new tests as necessary
9.  If you added new requirements, make sure they get added to ``pyproject.toml``/
    ``docs/requirements.txt``
10. Ensure you have the latest code by ``git pull``\ ing as necessary,
    to prevent any conflicts
11. ``git commit`` your change to a feature branch, and ``git push``
12. Open a merge request against master with your change
