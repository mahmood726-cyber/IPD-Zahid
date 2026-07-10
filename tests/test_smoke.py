"""Smoke tests: the module imports and exposes its public API.

Restores the two previously-recorded tests
(``test_module_imports`` / ``test_has_public_callable``) that were lost when
the source ``tests/test_smoke.py`` was deleted, leaving only a stale ``.pyc``.
"""


def test_module_imports():
    import ipd_qma  # noqa: F401

    assert ipd_qma is not None


def test_has_public_callable():
    import ipd_qma

    # The public surface: the estimator class, the simulator, and the
    # two result dataclasses.
    assert callable(ipd_qma.IPDQMA)
    assert callable(ipd_qma.simulate_location_scale)
    assert hasattr(ipd_qma, "PooledResult")
    assert hasattr(ipd_qma, "StudyResult")
