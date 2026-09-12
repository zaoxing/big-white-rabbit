"""RED test for BLE001 marker on metal_engine.py:260 BaseException.

The existing noqa_markers criterion (`! grep -rqE 'except Exception:$'`) cannot
see `except BaseException:` (see .loop/backlog.md:32). The fix is to add the
same `# noqa: BLE001 - <reason>` marker that line 263 already carries.
This test must FAIL before the fix (line 260 bare) and PASS after.
"""

import pathlib
import re


def test_no_bare_broad_except_without_noqa() -> None:
    root = pathlib.Path("python/bwr")
    # Matches `except Exception:` / `except BaseException:` plus alias form
    # (`except Exception as e:`) and tuple form (`except (Exception, ValueError):`),
    # without trailing `# noqa: BLE001`. The old criterion missed all three.
    pattern = re.compile(
        r"except\s+(?:"
        r"(?:Exception|BaseException)(?:\s+as\s+\w+)?"
        r"|\([^)]*(?:Exception|BaseException)[^)]*\)(?:\s+as\s+\w+)?"
        r")\s*:\s*(?:#.*)?$"
    )
    noqa = re.compile(r"#\s*noqa:\s*BLE001")

    violations: list[str] = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line) and not noqa.search(line):
                violations.append(f"{py}:{i}: {line.strip()}")

    assert not violations, (
        "Bare broad except without BLE001 marker:\n" + "\n".join(violations)
    )
