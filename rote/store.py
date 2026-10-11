"""Where capabilities, app profiles and tenant overlays live on disk.

    library/
      apps/<product>/profile.yaml                 one per vendor product
      apps/<product>/overlays/<tenant>.yaml       per-tenant profile differences
      capabilities/<name>/<version>.yaml          immutable once approved
      capabilities/<name>/overlays/<tenant>.yaml  per-tenant step differences

Plain files, because a capability is meant to be reviewed like code: diffed,
commented on in a pull request, approved. A database would be the next step
behind the same interface (load/save/resolve), not a change to the schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schema import AppProfile, Capability, Overlay, apply_overlay, apply_profile_overlay

LIBRARY = Path(__file__).resolve().parent.parent / "library"


# Default values that are pure noise to a reviewer. Pruned by name rather than
# with exclude_defaults, which would also drop the discriminators (by:, kind:).
_NOISE = {"negate": False, "exact": True, "expect": [], "note": "", "description": "",
          "timeout_ms": 10_000, "required": True, "human_steps": 0}


def _prune(value):
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if not (k in _NOISE and v == _NOISE[k])}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def _dump(model) -> str:
    data = _prune(model.model_dump(mode="json", exclude_none=True))
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def _version_key(path: Path) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in path.stem.split("."))
    except ValueError:
        return (-1,)


class Store:
    def __init__(self, root: Path | str = LIBRARY) -> None:
        self.root = Path(root)

    # -- capabilities --

    def capability_dir(self, name: str) -> Path:
        return self.root / "capabilities" / name

    def save(self, capability: Capability, overwrite: bool = False) -> Path:
        path = self.capability_dir(capability.name) / f"{capability.version}.yaml"
        if path.exists() and not overwrite:
            existing = self.load_path(path)
            # An approved version is immutable: change means a new version.
            if existing.approved:
                raise FileExistsError(f"{path} is approved; bump the version instead")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(capability))
        return path

    @staticmethod
    def load_path(path: Path | str) -> Capability:
        raw = yaml.safe_load(Path(path).read_text())
        return Capability.model_validate(raw)

    def versions(self, name: str) -> list[str]:
        directory = self.capability_dir(name)
        return [p.stem for p in sorted(directory.glob("*.yaml"), key=_version_key)]

    def load(self, name: str, version: str | None = None) -> Capability:
        versions = self.versions(name)
        if not versions:
            raise FileNotFoundError(f"no capability named {name!r} in {self.root}")
        version = version or versions[-1]
        if version not in versions:
            raise FileNotFoundError(f"{name} has no version {version}; have {versions}")
        return self.load_path(self.capability_dir(name) / f"{version}.yaml")

    def catalog(self) -> list[Capability]:
        """The latest version of every capability."""
        found = []
        for directory in sorted((self.root / "capabilities").glob("*")):
            if directory.is_dir() and self.versions(directory.name):
                found.append(self.load(directory.name))
        return found

    def overlay(self, name: str, tenant: str | None) -> Overlay | None:
        if not tenant:
            return None
        path = self.capability_dir(name) / "overlays" / f"{tenant}.yaml"
        return Overlay.model_validate(yaml.safe_load(path.read_text())) if path.exists() else None

    def resolve(self, name: str, tenant: str | None = None, version: str | None = None) -> Capability:
        """The capability as it should run for this tenant: base + overlay."""
        return apply_overlay(self.load(name, version), self.overlay(name, tenant))

    def approve(self, name: str, version: str, by: str) -> Capability:
        capability = self.load(name, version)
        capability.approval.status = "approved"
        capability.approval.by = by
        capability.approval.at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        capability.approval.fingerprint = capability.fingerprint()
        self.save(capability, overwrite=True)
        return capability

    # -- app profiles --

    def profile(self, product: str, tenant: str | None = None) -> AppProfile:
        directory = self.root / "apps" / product
        path = directory / "profile.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no app profile for {product!r} at {path}")
        profile = AppProfile.model_validate(yaml.safe_load(path.read_text()))
        overlay_path = directory / "overlays" / f"{tenant}.yaml"
        if tenant and overlay_path.exists():
            overlay = Overlay.model_validate(yaml.safe_load(overlay_path.read_text()))
            profile = apply_profile_overlay(profile, overlay)
        return profile
