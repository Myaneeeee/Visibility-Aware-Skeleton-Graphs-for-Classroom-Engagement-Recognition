"""Import side-effect: put the sibling ``lib/`` directory on sys.path.

``steps/*.py`` run as scripts (``python steps/foo.py``), so only ``steps/`` is on
sys.path by default. Importing this module first makes ``import common``,
``from model import ...``, ``from paths import ...`` resolve. During tests the
repo-root ``conftest.py`` provides the same paths.
"""
import os
import sys

_LIB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
