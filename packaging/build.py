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
FINGERPRINT_INPUTS = (
    REPO_ROOT / "pyproject.toml",
    SPEC_TEMPLATE,
    REPO_ROOT / "uv.lock",
)


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in FINGERPRINT_INPUTS:
        h.update(path.name.encode())
        h.update(path.read_bytes() if path.is_file() else b"<missing>")
    return h.hexdigest()


WHEELHOUSE = PACKAGING / "_build" / "wheels"


def build_wheel() -> Path:
    """Build a bwr wheel and return its path.

    venvstacks installs with `--only-binary :all:` and locks with hashes, so
    the layer cannot reference this checkout as a directory -- pip refuses
    ("Can't verify hashes for these file:// requirements because they point
    to directories"). Building the wheel up front satisfies both, and pins
    the compiled `_bwr_metal` extension to a single deterministic build.
    """
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    for stale in WHEELHOUSE.glob("big_white_rabbit-*.whl"):
        stale.unlink()
    # Prefer the development venv: it already has the scikit-build-core /
    # cmake / pybind11 toolchain this project needs to compile csrc/.
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    print(f"  building bwr wheel with {python}", flush=True)
    subprocess.run([python, "-m", "pip", "wheel", ".", "--no-deps",
                    "--wheel-dir", str(WHEELHOUSE)], cwd=REPO_ROOT, check=True)
    wheels = sorted(WHEELHOUSE.glob("big_white_rabbit-*.whl"))
    if not wheels:
        raise SystemExit(f"no bwr wheel produced in {WHEELHOUSE}")
    return wheels[-1]


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
