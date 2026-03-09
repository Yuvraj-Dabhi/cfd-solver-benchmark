"""Compatibility helpers for tempfile behavior on restricted Windows setups."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


_PATCH_FLAG = "_cfd_tempfile_patch_active"


def _stdlib_tempfile_is_writable() -> bool:
    """Return True when stdlib mkdtemp creates writable directories."""
    temp_dir: Path | None = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        probe = temp_dir / ".tmp_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except PermissionError:
        return False
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_fallback_root() -> Path:
    root = Path.cwd() / "tmp_check" / "tempfile_compat"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_tempfile_compat(force: bool = False) -> bool:
    """
    Patch ``tempfile.mkdtemp`` when Windows ACL behavior makes temp dirs unwritable.

    Returns True when a patch is active after this call.
    """
    if os.name != "nt":
        return False
    if getattr(tempfile, _PATCH_FLAG, False):
        return True
    if not force and _stdlib_tempfile_is_writable():
        return False

    fallback_root = _resolve_fallback_root()

    def _patched_mkdtemp(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> str:
        if suffix is None:
            suffix = ""
        if prefix is None:
            prefix = tempfile.gettempprefix()

        roots = []
        if dir:
            roots.append(Path(dir))
        roots.append(fallback_root)

        for root in roots:
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue

            names = tempfile._get_candidate_names()
            for _ in range(tempfile.TMP_MAX):
                candidate = f"{prefix}{next(names)}{suffix}"
                path = root / candidate
                try:
                    # Intentionally omit mode to keep filesystem-default ACLs.
                    os.mkdir(path)
                    return str(path.resolve())
                except FileExistsError:
                    continue
                except PermissionError:
                    break

        raise PermissionError("Unable to create a writable temporary directory")

    tempfile.mkdtemp = _patched_mkdtemp
    tempfile.tempdir = str(fallback_root)
    setattr(tempfile, _PATCH_FLAG, True)
    return True
