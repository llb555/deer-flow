"""Cleanup failures at the census sites must not replace the caller's outcome.

Each site stages a temporary file and discards it on an exit path that may also
be reporting a failure. Before this change a failing ``unlink`` replaced the
error the caller needed (and, in ``_write_bytes_atomically``, a failing
``fdopen`` leaked the descriptor).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def real_unlink():
    return Path.unlink


def _fail_unlink_on(monkeypatch, predicate, real_unlink):
    """Make ``Path.unlink`` raise for the paths a test owns."""

    def unlink(self, *args, **kwargs):
        if predicate(self):
            raise PermissionError("temporary unlink failed")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)


def test_extension_restore_keeps_the_replace_error_when_cleanup_fails(tmp_path, monkeypatch, real_unlink):
    from deerflow.extensions import manager

    target = tmp_path / "pyproject.toml"
    target.write_text("original", encoding="utf-8")

    def _failing_replace(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(manager.os, "replace", _failing_replace)
    _fail_unlink_on(monkeypatch, lambda path: path.suffix == ".tmp", real_unlink)

    with pytest.raises(OSError, match="replace failed"):
        manager._write_bytes_atomically(target, b"replacement", None)

    assert target.read_text(encoding="utf-8") == "original"


def test_managed_subagent_write_keeps_the_replace_error_when_cleanup_fails(tmp_path, monkeypatch, real_unlink):
    from deerflow.persistence.managed_subagents import file as store_module
    from deerflow.persistence.managed_subagents.base import ManagedSubagentDefinition

    definition = ManagedSubagentDefinition(
        name="planner",
        description="plans",
        system_prompt="plan things",
    )
    target = tmp_path / "planner.json"

    def _failing_replace(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(store_module.os, "replace", _failing_replace)
    _fail_unlink_on(monkeypatch, lambda path: path.suffix == ".tmp", real_unlink)

    with pytest.raises(OSError, match="replace failed"):
        store_module.FileManagedSubagentStore._atomic_write(target, definition)

    assert not target.exists()
