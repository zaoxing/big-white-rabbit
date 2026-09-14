"""Named settings bundles for models: per-model profiles and global templates.

A profile is a name plus a bag of settings. Two collections, deliberately:

  templates   global, reusable, not tied to a model. Some ship built-in and
              are read-only.
  profiles    per model id, optionally derived from a template.

Sampling versus engine settings
-------------------------------
The split matters and is enforced here rather than in the UI. Sampling
settings (`temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`) are
per-request, so a profile carrying only those can be applied to a LOADED
model with no reload -- which is what makes `expose_as_model` possible.
Engine settings (`n_ctx`, `n_batch`, `mlx_batch`, ...) are fixed when the
engine is constructed; changing one means reloading the model, so a profile
carrying them cannot be overlaid per request. `has_engine_fields` reports
that classification so the client never has to mirror the list -- and
`ENGINE_FIELDS` is the single place it is written down.

Storage
-------
One JSON file under the model directory (`.bwr/profiles.json`), written
atomically via a temp file and `os.replace`. That keeps profiles beside the
models they describe, so a model directory stays self-describing when it
moves between machines. There is no database: the whole file is small
(names and small dicts) and is read on demand.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Settings fixed when the engine is built. A profile touching any of these
# cannot be applied to a loaded model, only to the next load.
ENGINE_FIELDS = frozenset({
    "n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_threads", "n_threads_batch",
    "flash_attn", "kv_unified", "speculative", "spec_max_drafts",
    "mlx_batch", "mlx_prefix_cache", "mlx_prefix_cache_size", "mlx_kv_bits",
    "mlx_mtp", "mlx_mtp_depth", "prefix_cache", "prefix_cache_pins",
})

# Settings applied per request. Anything here can be overlaid on a loaded
# model, which is what `expose_as_model` does.
SAMPLING_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "max_tokens", "stop", "seed",
    "repetition_penalty", "presence_penalty", "frequency_penalty",
})

# Profile names become URL path segments and, when exposed, part of a model
# id after a colon. Keep them boring.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Separator between a base model id and an exposed profile name.
EXPOSE_SEP = ":"


class ProfileError(ValueError):
    """Bad name, duplicate, missing, or an attempt to edit a built-in."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Profile:
    name: str
    display_name: str = ""
    description: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    source_template: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    is_builtin: bool = False
    expose_as_model: bool = False

    def to_dict(self, *, base_model: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name or self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_template": self.source_template,
            "is_builtin": self.is_builtin,
            "settings": dict(self.settings),
            # The client renders a warning from this rather than keeping its
            # own copy of ENGINE_FIELDS, so the classification cannot drift.
            "has_engine_fields": self.has_engine_fields,
        }
        if base_model is not None:
            out["expose_as_model"] = self.expose_as_model
            out["model_id"] = exposed_id(base_model, self.name)
        return out

    @property
    def has_engine_fields(self) -> bool:
        return any(k in ENGINE_FIELDS for k in self.settings)

    @property
    def sampling_settings(self) -> dict[str, Any]:
        """Just the per-request half. What an overlay may touch."""
        return {k: v for k, v in self.settings.items() if k in SAMPLING_FIELDS}


def exposed_id(base_model: str, profile_name: str) -> str:
    return f"{base_model}{EXPOSE_SEP}{profile_name}"


def split_exposed(model_id: str) -> tuple[str, str | None]:
    """`"m:fast"` -> `("m", "fast")`; a plain id -> `(id, None)`.

    Splits on the LAST separator. Model ids can contain colons (a nested
    directory layout such as `repo/4-bit` does not, but a Hub id with a
    revision could), and the profile name is the part this module owns.
    """
    base, sep, name = model_id.rpartition(EXPOSE_SEP)
    if not sep or not name or not base:
        return model_id, None
    return base, name


# Shipped read-only starting points. Deliberately sampling-only so every one
# of them can be exposed as a model without a reload.
BUILTIN_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "precise",
        "display_name": "Precise",
        "description": "Greedy decoding. Deterministic, for code and extraction.",
        "settings": {"temperature": 0.0, "top_p": 1.0},
    },
    {
        "name": "balanced",
        "display_name": "Balanced",
        "description": "General-purpose sampling.",
        "settings": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
    },
    {
        "name": "creative",
        "display_name": "Creative",
        "description": "Looser sampling for drafting and brainstorming.",
        "settings": {"temperature": 1.0, "top_p": 0.98, "top_k": 80},
    },
)


class ProfileStore:
    """Thread-safe JSON-backed store. One instance per server."""

    FILENAME = "profiles.json"

    def __init__(self, model_dir: str | os.PathLike[str] | None) -> None:
        self._dir = Path(model_dir) / ".bwr" if model_dir is not None else None
        self._lock = threading.RLock()
        self._templates: dict[str, Profile] = {}
        self._profiles: dict[str, dict[str, Profile]] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    @property
    def path(self) -> Path | None:
        return (self._dir / self.FILENAME) if self._dir is not None else None

    def _load(self) -> None:
        raw: dict[str, Any] = {}
        p = self.path
        if p is not None and p.is_file():
            try:
                raw = json.loads(p.read_text())
            except (OSError, ValueError):
                # A corrupt file must not stop the server from serving. The
                # built-ins below still give the user a working set, and the
                # next write replaces the bad file.
                raw = {}
        self._templates = {
            t["name"]: Profile(is_builtin=True, **t) for t in BUILTIN_TEMPLATES
        }
        for name, body in (raw.get("templates") or {}).items():
            if name in self._templates:
                continue  # a built-in name is never overridden from disk
            self._templates[name] = Profile(**{**body, "name": name, "is_builtin": False})
        self._profiles = {}
        for model_id, bundle in (raw.get("models") or {}).items():
            self._profiles[model_id] = {
                name: Profile(**{**body, "name": name})
                for name, body in (bundle or {}).items()
            }

    def _save(self) -> None:
        p = self.path
        if p is None:
            return  # single-model server: nothing to anchor a file to
        payload = {
            "templates": {
                n: _body(t) for n, t in self._templates.items() if not t.is_builtin
            },
            "models": {
                m: {n: _body(pr) for n, pr in bundle.items()}
                for m, bundle in self._profiles.items()
                if bundle
            },
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a crash mid-write must not leave a half-file that the next
        # start would read as "no profiles".
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".profiles-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
            os.replace(tmp, p)
        except BaseException:  # noqa: BLE001 - cleanup then re-raise; a failed
            # write must not leave a stray .profiles-* temp file behind, and
            # BaseException is deliberate so a KeyboardInterrupt mid-write
            # cleans up too.
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- templates ---------------------------------------------------------

    def list_templates(self) -> list[Profile]:
        with self._lock:
            return sorted(self._templates.values(), key=lambda t: (not t.is_builtin, t.name))

    def create_template(
        self, name: str, *, display_name: str = "", description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Profile:
        with self._lock:
            _check_name(name)
            if name in self._templates:
                raise ProfileError(f"template {name!r} already exists")
            t = Profile(
                name=name, display_name=display_name or name, description=description,
                settings=dict(settings or {}), created_at=_now(), updated_at=_now(),
            )
            self._templates[name] = t
            self._save()
            return t

    def update_template(self, name: str, **changes: Any) -> Profile:
        with self._lock:
            t = self._templates.get(name)
            if t is None:
                raise ProfileError(f"unknown template {name!r}")
            if t.is_builtin:
                raise ProfileError(f"{name!r} is built in and cannot be edited")
            updated = _apply_changes(t, changes)
            if updated.name != name:
                _check_name(updated.name)
                if updated.name in self._templates:
                    raise ProfileError(f"template {updated.name!r} already exists")
                del self._templates[name]
            self._templates[updated.name] = updated
            self._save()
            return updated

    def delete_template(self, name: str) -> bool:
        with self._lock:
            t = self._templates.get(name)
            if t is None:
                return False
            if t.is_builtin:
                raise ProfileError(f"{name!r} is built in and cannot be deleted")
            del self._templates[name]
            self._save()
            return True

    # -- per-model profiles ------------------------------------------------

    def list_profiles(self, model_id: str) -> list[Profile]:
        with self._lock:
            return sorted(self._profiles.get(model_id, {}).values(), key=lambda p: p.name)

    def get_profile(self, model_id: str, name: str) -> Profile | None:
        with self._lock:
            return self._profiles.get(model_id, {}).get(name)

    def create_profile(
        self, model_id: str, name: str, *, display_name: str = "",
        description: str | None = None, settings: dict[str, Any] | None = None,
        source_template: str | None = None, also_save_as_template: bool = False,
        expose_as_model: bool = False,
    ) -> Profile:
        with self._lock:
            _check_name(name)
            bundle = self._profiles.setdefault(model_id, {})
            if name in bundle:
                raise ProfileError(f"profile {name!r} already exists for {model_id}")
            merged = dict(settings or {})
            if source_template:
                tmpl = self._templates.get(source_template)
                if tmpl is None:
                    raise ProfileError(f"unknown template {source_template!r}")
                # Explicit settings win over the template they came from.
                merged = {**tmpl.settings, **merged}
            p = Profile(
                name=name, display_name=display_name or name, description=description,
                settings=merged, source_template=source_template,
                created_at=_now(), updated_at=_now(), expose_as_model=expose_as_model,
            )
            bundle[name] = p
            if also_save_as_template and name not in self._templates:
                self._templates[name] = Profile(
                    name=name, display_name=p.display_name, description=description,
                    settings=dict(merged), created_at=_now(), updated_at=_now(),
                )
            self._save()
            return p

    def update_profile(self, model_id: str, name: str, **changes: Any) -> Profile:
        with self._lock:
            bundle = self._profiles.get(model_id, {})
            p = bundle.get(name)
            if p is None:
                raise ProfileError(f"unknown profile {name!r} for {model_id}")
            also_template = bool(changes.pop("also_save_as_template", False))
            updated = _apply_changes(p, changes)
            if updated.name != name:
                _check_name(updated.name)
                if updated.name in bundle:
                    raise ProfileError(f"profile {updated.name!r} already exists")
                del bundle[name]
            bundle[updated.name] = updated
            if also_template:
                self._templates[updated.name] = Profile(
                    name=updated.name, display_name=updated.display_name,
                    description=updated.description, settings=dict(updated.settings),
                    created_at=_now(), updated_at=_now(),
                )
            self._save()
            return updated

    def delete_profile(self, model_id: str, name: str) -> bool:
        with self._lock:
            bundle = self._profiles.get(model_id, {})
            if name not in bundle:
                return False
            del bundle[name]
            self._save()
            return True

    # -- exposure ----------------------------------------------------------

    def exposed(self) -> list[tuple[str, str, Profile]]:
        """`(exposed_id, base_model, profile)` for every exposed profile."""
        with self._lock:
            out = []
            for model_id, bundle in self._profiles.items():
                for p in bundle.values():
                    if p.expose_as_model:
                        out.append((exposed_id(model_id, p.name), model_id, p))
            return sorted(out)

    def resolve_overlay(self, model_id: str) -> tuple[str, dict[str, Any]]:
        """Split an exposed id into `(base model, sampling overlay)`.

        Returns the id unchanged with an empty overlay when it names no
        exposed profile -- so every caller can run it unconditionally.

        Only SAMPLING settings come back. An exposed profile that also sets
        engine fields keeps them for its next load; silently applying them
        per request is impossible, and failing the request over a setting the
        user cannot see would be worse than ignoring it.
        """
        base, name = split_exposed(model_id)
        if name is None:
            return model_id, {}
        with self._lock:
            p = self._profiles.get(base, {}).get(name)
            if p is None or not p.expose_as_model:
                return model_id, {}
            return base, p.sampling_settings


def _body(p: Profile) -> dict[str, Any]:
    """Serialised profile, minus the name (it is the dict key) and the
    built-in flag (only shipped templates carry it, and those are never
    written)."""
    return {
        "display_name": p.display_name,
        "description": p.description,
        "settings": p.settings,
        "source_template": p.source_template,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
        "expose_as_model": p.expose_as_model,
    }


def _check_name(name: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise ProfileError(
            f"invalid name {name!r}: use letters, digits, dot, dash or underscore "
            "(max 64, must start alphanumeric)"
        )


def _apply_changes(p: Profile, changes: dict[str, Any]) -> Profile:
    """Merge a patch. Absent keys and explicit `None` both mean "leave it".

    That is the contract the client encodes: UpdateProfileRequest omits nil
    fields, and a settings-only update must not clear the display name or
    reset exposure state set from the web dashboard.
    """
    out = Profile(**{**p.__dict__})
    if changes.get("new_name"):
        out.name = changes["new_name"]
    for key in ("display_name", "description"):
        if changes.get(key) is not None:
            setattr(out, key, changes[key])
    if changes.get("settings") is not None:
        out.settings = dict(changes["settings"])
    if changes.get("source_template") is not None:
        out.source_template = changes["source_template"]
    if changes.get("expose_as_model") is not None:
        out.expose_as_model = bool(changes["expose_as_model"])
    out.updated_at = _now()
    return out
