"""Make lib/ and steps/ importable by bare name during the test run.

pytest auto-imports this repo-root conftest before collecting tests, so
``import common``, ``from model import ...``, ``import train`` etc. resolve
regardless of where the module physically lives.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("lib", "steps"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
