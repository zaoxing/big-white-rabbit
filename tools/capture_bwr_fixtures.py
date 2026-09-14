#!/usr/bin/env python3
"""Capture the macOS app's JSON contract from a live bwr server.

The Swift suite decodes these through the app's real DTOs
(`apps/bwr-mac/Tests/BigWhiteRabbitTests/BWRBackendContractTests.swift`), so
a shape change on either side fails a test instead of silently blanking a
screen at runtime.

    .venv/bin/python -m bwr.cli serve --model-dir ./models --port 1919
    python3 tools/capture_bwr_fixtures.py [--port 1919] [--warm]

`--warm` sends one short completion first so the stats fixture carries real
counters rather than a zeroed one -- worth doing, since "all zeros" is also
what a broken accumulator looks like.
"""

from __future__ import annotations

import argparse
import getpass
import json
import pathlib
import re
import socket
import sys
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = REPO / "apps/bwr-mac/Tests/BigWhiteRabbitTests/Fixtures"

# name -> path. The name is what the Swift test asks for.
ENDPOINTS = {
    "bwr-server-info": "/admin/api/server-info",
    "bwr-stats": "/admin/api/stats",
    "bwr-models": "/admin/api/models",
    "bwr-global-settings": "/admin/api/global-settings",
    "bwr-device-info": "/admin/api/device-info",
    "bwr-activity": "/admin/api/activity",
    "bwr-update-check": "/admin/api/update-check",
    "bwr-cluster-deployments": "/admin/api/cluster/deployments",
    "bwr-logs": "/admin/api/logs",
    "bwr-usage": "/admin/api/usage",
    "bwr-api-status": "/api/status",
}
# Needs a concrete model id, filled in from /v1/models at capture time.
MODEL_SCOPED = {"bwr-model-settings": "/admin/api/models/{model}/settings"}


def sanitize(text: str) -> str:
    """Strip this machine out of the fixture.

    These files are committed, and a capture carries the operator's home
    directory and the Mac's Bonjour name. Doing it here rather than by hand
    also keeps re-capture idempotent -- a hand-sanitised fixture would come
    back dirty on the next run and show up as noise in every diff.
    """
    text = re.sub(r"/Users/[^/\"]+", "/Users/test", text)
    for real, fake in (
        (socket.gethostname(), "test-mac"),
        (getpass.getuser(), "test"),
    ):
        if real:
            text = text.replace(real, fake)
    return text


def fetch(base: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--warm", action="store_true",
                    help="send one completion first so stats are non-zero")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    status, models = fetch(base, "/v1/models")
    if status != 200:
        print(f"no server at {base} (/v1/models -> {status})", file=sys.stderr)
        return 1
    ids = [m["id"] for m in models.get("data", [])]
    if not ids:
        print(f"server at {base} lists no models; capture would be empty",
              file=sys.stderr)
        return 1
    model = ids[0]

    if args.warm:
        print(f"warming {model} (first load reads the weights off disk)...")
        code, _ = fetch(base, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
        })
        print(f"  completion -> {code}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, path in {**ENDPOINTS,
                       **{k: v.format(model=model) for k, v in MODEL_SCOPED.items()}
                       }.items():
        code, body = fetch(base, path)
        (FIXTURES / f"{name}.json").write_text(
            sanitize(json.dumps(body, indent=1, sort_keys=True)) + "\n"
        )
        print(f"  {code}  {name:28} {path}")
    print(f"\nwrote {len(ENDPOINTS) + len(MODEL_SCOPED)} fixtures to {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
