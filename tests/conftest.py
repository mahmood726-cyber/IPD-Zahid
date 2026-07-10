"""Pytest configuration: make the ipd_qma_v2 package importable.

The analysis code lives in ``ipd_qma_v2/`` as a flat script directory (no
``__init__.py``), and the runner scripts import it via
``sys.path.insert(0, os.path.dirname(__file__)); from ipd_qma import ...``.
We reproduce that here so tests can ``import ipd_qma`` directly.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_DIR = os.path.join(_REPO_ROOT, "ipd_qma_v2")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
