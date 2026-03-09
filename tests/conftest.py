"""Pytest session setup."""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from scripts.utils.tempfile_compat import ensure_tempfile_compat


def _prepare_pytest_temp_root() -> None:
    """
    Force pytest temp roots into a writable location with default ACLs.

    Pytest's default `%TEMP%/pytest-of-<user>` location can become inaccessible in
    some OneDrive/sandbox ACL combinations.
    """
    root = Path.cwd() / "tmp_check" / "pytest_root"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(root))

    user_root = root / f"pytest-of-{getpass.getuser()}"
    user_root.mkdir(parents=True, exist_ok=True)


def _patch_windows_mkdir_mode() -> None:
    """
    Drop restrictive mode bits on Windows mkdir calls during tests.

    Some sandboxed/OneDrive ACL combinations treat explicit mode values (for
    example pytest's 0o700 temp dirs) as non-writable for the running process.
    """
    if os.name != "nt":
        return

    _orig_mkdir = os.mkdir

    def _mkdir_default_acl(path, mode=0o777, *, dir_fd=None):  # type: ignore[override]
        if dir_fd is None:
            return _orig_mkdir(path)
        return _orig_mkdir(path, dir_fd=dir_fd)

    os.mkdir = _mkdir_default_acl  # type: ignore[assignment]


_prepare_pytest_temp_root()
_patch_windows_mkdir_mode()

# Apply once at collection time so tempfile-based tests work on Windows ACL edge cases.
ensure_tempfile_compat()
