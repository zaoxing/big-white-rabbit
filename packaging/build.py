"""Driver for the macOS bundle's embedded Python layers.

`apps/bwr-mac/Scripts/build.sh` shells out to this with three interfaces, so
they are a contract, not conveniences:

  --print-fingerprint        hash of the inputs that invalidate the export
  --venvstacks-only          (re)build packaging/_export/ and stamp it
  --write-engine-commits DIR write DIR/_engine_commits.json

Why the spec is generated
=========================

bwr is not published to an index, so the layer has to reference this
checkout. It cannot reference it as a DIRECTORY: venvstacks installs with
`--only-binary :all:` and locks with hashes, and pip refuses to hash a
directory. So this script builds a wheel first and substitutes its path for
the committed spec's `@BWR_WHEEL@` placeholder, writing the resolved copy
into the build directory. The generated file is disposable; the template is
the source of truth, and it stays free of any absolute path so it does not
hard-code one developer's home directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGING = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING.parent
SPEC_TEMPLATE = PACKAGING / "venvstacks.toml"
GENERATED_SPEC = PACKAGING / "_build" / "venvstacks.resolved.toml"
EXPORT_DIR = PACKAGING / "_export"

# Inputs whose contents decide whether the export is stale. build.sh compares
# this against packaging/_export/.fingerprint before deciding to rebuild.
FINGERPRINT_FILES = (
    REPO_ROOT / "pyproject.toml",
    SPEC_TEMPLATE,
    REPO_ROOT / "uv.lock",
)
# bwr's own sources MUST be in here. They were not at first, and the result
# was a bundle that built happily around a stale wheel: the app spawned
# `bwr serve --preload first` and the embedded (older) bwr answered
# "unrecognized arguments: --preload first". Dependency metadata alone does
# not change when the code does.
FINGERPRINT_TREES = (
    (REPO_ROOT / "python" / "bwr", ("*.py",)),
    (REPO_ROOT / "csrc", ("*.cpp", "*.h", "*.metal", "*.mm")),
)


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in FINGERPRINT_FILES:
        h.update(path.name.encode())
        h.update(path.read_bytes() if path.is_file() else b"<missing>")
    for root, patterns in FINGERPRINT_TREES:
        for pattern in patterns:
            for src in sorted(root.rglob(pattern)):
                if "__pycache__" in src.parts:
                    continue
                h.update(str(src.relative_to(REPO_ROOT)).encode())
                h.update(src.read_bytes())
    return h.hexdigest()


WHEELHOUSE = PACKAGING / "_build" / "wheels"


def build_wheel() -> Path:
    """Build a bwr wheel and return its path, under a content-addressed dir.

    venvstacks installs with `--only-binary :all:` and locks with hashes, so
    the layer cannot reference this checkout as a directory -- pip refuses
    ("Can't verify hashes for these file:// requirements because they point
    to directories"). Building the wheel up front satisfies both, and pins
    the compiled `_bwr_metal` extension to a single deterministic build.

    Why the wheel lives in `wheels/<sha256[:12]>/` instead of `wheels/`:
    the project version does not move between builds, so a rebuilt wheel has
    a byte-identical FILENAME. The locked requirement string was therefore
    identical too, `--lock-if-needed` saw nothing to redo, and the install
    then died with "THESE PACKAGES DO NOT MATCH THE HASHES FROM THE
    REQUIREMENTS FILE" -- the lock still carried the previous wheel's hash.
    Hashing the content into the directory makes the requirement change
    exactly when the code does, so the lock goes stale on its own.

    This does NOT make pip reinstall the wheel; see purge_installed_bwr().
    """
    staging = WHEELHOUSE / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    # Prefer the development venv: it already has the scikit-build-core /
    # cmake / pybind11 toolchain this project needs to compile csrc/.
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    print(f"  building bwr wheel with {python}", flush=True)
    subprocess.run([python, "-m", "pip", "wheel", ".", "--no-deps",
                    "--wheel-dir", str(staging)], cwd=REPO_ROOT, check=True)
    built = sorted(staging.glob("big_white_rabbit-*.whl"))
    if not built:
        raise SystemExit(f"no bwr wheel produced in {staging}")
    wheel = built[-1]
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()[:12]
    final_dir = WHEELHOUSE / digest
    final = final_dir / wheel.name
    if not final.exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(wheel), str(final))
    shutil.rmtree(staging)
    # Keep only this build's wheel; the others can never be referenced again.
    for other in WHEELHOUSE.iterdir():
        if other != final_dir:
            shutil.rmtree(other) if other.is_dir() else other.unlink()
    return final


LAYER_ROOT = PACKAGING / "_build" / "framework-mlx-base"
LAYER_SITE = LAYER_ROOT / "lib" / "python3.11" / "site-packages"


def purge_installed_bwr() -> None:
    """Delete bwr from the framework layer so pip installs the new wheel.

    pip decides a wheel is redundant by VERSION alone:

        big-white-rabbit is already installed with the same version as the
        provided wheel. Use --force-reinstall to force an installation.

    bwr's version does not move between builds and venvstacks exposes no
    --force-reinstall, so an unchanged version number was enough to make the
    layer keep whatever bwr it had. That is how the bundle came to ship a
    `bwr` that answered `unrecognized arguments: --preload first` while the
    freshly built wheel sitting beside it had the flag. Removing the install
    first is the only lever this side of the venvstacks CLI. It costs one
    small reinstall per export; the export itself is already gated on the
    fingerprint, so this runs only when something actually changed.
    """
    if not LAYER_SITE.is_dir():
        return
    purged = False
    for dist_info in LAYER_SITE.glob("big_white_rabbit-*.dist-info"):
        record = dist_info / "RECORD"
        if record.is_file():
            for line in record.read_text().splitlines():
                rel = line.split(",", 1)[0].strip()
                if not rel:
                    continue
                target = LAYER_SITE / rel
                # RECORD reaches outside site-packages for console scripts
                # (`../../../bin/bwr`). Follow those, but never outside the
                # layer -- a malformed RECORD must not delete anything else.
                try:
                    resolved = target.resolve()
                    resolved.relative_to(LAYER_ROOT.resolve())
                except (OSError, ValueError):
                    continue
                if resolved.is_file() or resolved.is_symlink():
                    resolved.unlink()
        shutil.rmtree(dist_info, ignore_errors=True)
        purged = True
    shutil.rmtree(LAYER_SITE / "bwr", ignore_errors=True)
    if purged:
        print("  purged the previously installed bwr from framework-mlx-base",
              flush=True)


def render_spec() -> Path:
    """Write the spec with @BWR_WHEEL@ resolved to a freshly built wheel."""
    GENERATED_SPEC.parent.mkdir(parents=True, exist_ok=True)
    wheel = build_wheel()
    text = SPEC_TEMPLATE.read_text().replace("@BWR_WHEEL@", wheel.as_uri())
    GENERATED_SPEC.write_text(text)
    return GENERATED_SPEC


def venvstacks(*args: str) -> None:
    """Run the venvstacks CLI, preferring an installed console script."""
    exe = None
    for candidate in ("venvstacks", str(Path.home() / ".local/bin/venvstacks")):
        if subprocess.run(["command", "-v", candidate], shell=False,
                          capture_output=True).returncode == 0 or Path(candidate).exists():
            exe = candidate
            break
    cmd = [exe, *args] if exe else [sys.executable, "-m", "venvstacks", *args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_export() -> None:
    spec = render_spec()
    # lock + build + export in one pass. --clean would discard the layer
    # caches that make an incremental rebuild bearable; the fingerprint above
    # is what decides when a rebuild is needed at all.
    # --lock-if-needed suffices because build_wheel() puts the wheel behind a
    # content-addressed path: when bwr changes, the requirement string in the
    # spec changes with it, so the lock is genuinely outdated and gets redone.
    # Third-party versions in the layer therefore only move when their own
    # constraints do, not on every bwr edit.
    purge_installed_bwr()
    venvstacks("build", str(spec), "--lock-if-needed",
               "--build-dir", str(PACKAGING / "_build"),
               "--output-dir", str(PACKAGING / "_artifacts"))
    venvstacks("local-export", str(spec),
               "--build-dir", str(PACKAGING / "_build"),
               "--output-dir", str(EXPORT_DIR))
    _flatten_export()
    (EXPORT_DIR / ".fingerprint").write_text(fingerprint())


def _flatten_export() -> None:
    """Ensure the layer dirs sit directly under _export/.

    build.sh looks for `_export/cpython-3.11` and `_export/framework-mlx-base`
    (and copies them into Contents/Resources/Python/ verbatim). venvstacks may
    nest its export one level deeper depending on version; hoist if so, rather
    than teaching build.sh two layouts.
    """
    wanted = ("cpython-3.11", "framework-mlx-base")
    if all((EXPORT_DIR / name).is_dir() for name in wanted):
        return
    for child in EXPORT_DIR.iterdir():
        if child.is_dir() and all((child / name).is_dir() for name in wanted):
            for name in wanted:
                (child / name).rename(EXPORT_DIR / name)
            return
    found = sorted(p.name for p in EXPORT_DIR.iterdir()) if EXPORT_DIR.is_dir() else []
    raise SystemExit(
        f"venvstacks export is missing {wanted}; found {found}. "
        "build.sh cannot stage a bundle from this tree."
    )


def write_engine_commits(target: Path) -> None:
    """Stamp the packaged tree with the commit it was built from."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - metadata only; never fail a build
            return ""

    target.mkdir(parents=True, exist_ok=True)
    (target / "_engine_commits.json").write_text(json.dumps({
        "bwr": git("rev-parse", "HEAD"),
        "describe": git("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(git("status", "--porcelain")),
    }, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-fingerprint", action="store_true")
    ap.add_argument("--venvstacks-only", action="store_true")
    ap.add_argument("--write-engine-commits", metavar="DIR")
    args = ap.parse_args()

    if args.print_fingerprint:
        print(fingerprint())
        return 0
    if args.write_engine_commits:
        write_engine_commits(Path(args.write_engine_commits))
        return 0
    if args.venvstacks_only:
        build_export()
        return 0
    ap.error("nothing to do: pass --print-fingerprint, --venvstacks-only "
             "or --write-engine-commits")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
