"""AsyncEngine: an asyncio face over the synchronous MetalEngine.

Why a thread at all. ``llama_decode`` is a blocking C call. The binding releases the GIL
around it, but the step loop still cannot run *on* the event loop: a decode holding the
loop would stall ``/health``, SSE flushes, and every other request for its duration. So
the engine lives on one dedicated worker thread and the loop talks to it over queues.

Why ALL engine calls are marshalled, not just step(). ``MetalEngine`` and the underlying
``llama_context`` are not thread-safe -- concurrent ``add_request`` and ``step`` would
race on the KV cache and the seq_id pool. Calling ``add_request`` straight from a request
handler would put engine mutation on the event-loop thread while the worker is mid-decode.
So submit and cancel are *commands* posted to the worker, and the worker is the only
thread that ever touches the engine. That single-owner rule is the whole design.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from .config import RequestParams
from .metal_engine import MetalEngine, StepOutput

# Sentinel pushed into a request's output queue to close its stream.
_END = object()


@dataclass
class _Command:
    """Work for the engine thread. `future` is a plain concurrent future, resolved on the
    worker and awaited (via asyncio.wrap_future) by the caller."""

    kind: str  # "submit" | "cancel"
    future: "asyncio.Future | None" = None
    prompt: "str | Sequence[int] | None" = None
    params: RequestParams | None = None
    request_id: int | None = None
    add_special: bool = True


@dataclass
class _Stream:
    """One request's delivery channel, owned by the event loop."""

    queue: "asyncio.Queue[object]" = field(default_factory=asyncio.Queue)
    done: bool = False
    # True only when the ENGINE retired the request itself (a StepOutput with
    # finished=True reached _deliver). `done` is weaker: it is also set when the stream
    # is closed early or failed, which is exactly the case where the engine still owns
    # the seq_id. release() needs to tell those two apart -- see its docstring.
    retired: bool = False


class AsyncEngine:
    """Serve many concurrent requests from one MetalEngine on one worker thread."""

    def __init__(self, engine: MetalEngine, *, idle_poll_s: float = 0.05) -> None:
        self.engine = engine
        self._idle_poll_s = idle_poll_s
        self._commands: "queue.Queue[_Command | None]" = queue.Queue()
        self._streams: dict[int, _Stream] = {}
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()
        self._worker_error: BaseException | None = None

    # --- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, name="freetoken-mac-engine", daemon=True
        )
        self._thread.start()

    async def stop(self, *, close_context: bool = True) -> None:
        if self._thread is None:
            return
        self._stopping.set()
        self._commands.put(None)  # wake the worker out of its idle wait
        thread, self._thread = self._thread, None
        await asyncio.get_running_loop().run_in_executor(None, thread.join, 10.0)
        # Close any stream still open so its consumer is not left awaiting forever.
        for rid in list(self._streams):
            self._close_stream(rid)
        self._streams.clear()
        if close_context:
            # Release the llama_context deterministically. Waiting for Python to collect
            # it is not good enough: ggml frees the Metal device from a static destructor
            # at process exit and asserts its residency sets are empty, so a context that
            # is still alive then abort()s the process AFTER a clean shutdown. The worker
            # is joined, so this is the only thread touching the engine now.
            self.engine.ctx.close()

    # --- submission --------------------------------------------------------------

    async def submit(
        self,
        prompt: str | Sequence[int],
        params: RequestParams | None = None,
        *,
        add_special: bool = True,
    ) -> int:
        """Admit a request and return its id. Raises whatever the engine raises (an
        over-long prompt, seq_id exhaustion) -- the worker survives either way."""
        self._check_worker()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._commands.put(
            _Command(
                kind="submit",
                future=fut,
                prompt=prompt,
                params=params,
                add_special=add_special,
            )
        )
        return await fut

    async def cancel(self, request_id: int) -> None:
        """Ask the worker to drop a request. Returns once the command is queued; the
        stream closes when the worker acts on it."""
        self._post_cancel(request_id)

    def _post_cancel(self, request_id: int) -> None:
        """Queue a cancel. Safe from either thread and safe to repeat: the worker's
        handler tolerates an unknown or already-finished request."""
        self._commands.put(_Command(kind="cancel", request_id=request_id))

    async def stream(self, request_id: int) -> AsyncIterator[StepOutput]:
        """Yield this request's tokens until it finishes or is cancelled."""
        st = self._streams.get(request_id)
        if st is None:
            raise KeyError(f"unknown request {request_id}")
        while True:
            item = await st.queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, StepOutput)
            yield item
            if item.finished:
                return

    # --- worker ------------------------------------------------------------------

    def _run(self) -> None:
        """The engine thread. The ONLY thread that touches self.engine."""
        try:
            while not self._stopping.is_set():
                self._drain_commands()
                if self._stopping.is_set():
                    break
                if not self._has_work():
                    # Nothing in flight: block on the command queue instead of spinning.
                    try:
                        cmd = self._commands.get(timeout=self._idle_poll_s)
                    except queue.Empty:
                        continue
                    if cmd is None:
                        break
                    self._apply(cmd)
                    continue
                # A step can still raise (a policy overrun, a KV-slot failure). One bad
                # step must not end the worker, so failures are reported to whichever
                # requests are open and the loop carries on.
                try:
                    outputs = self.engine.step()
                except BaseException as exc:  # noqa: BLE001 - deliberately broad
                    self._fail_open_streams(exc)
                    continue
                for out in outputs:
                    self._dispatch(out)
        except BaseException as exc:  # noqa: BLE001 - the thread must never die silently
            self._worker_error = exc
            self._fail_open_streams(exc)

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            if cmd is None:
                self._stopping.set()
                return
            self._apply(cmd)

    def _apply(self, cmd: _Command) -> None:
        if cmd.kind == "submit":
            try:
                rid = self.engine.add_request(
                    cmd.prompt, cmd.params, add_special=cmd.add_special
                )
            except BaseException as exc:  # noqa: BLE001 - hand it back, stay alive
                self._resolve(cmd.future, exc=exc)
                return
            # Register the stream BEFORE resolving, so `stream(rid)` cannot race ahead
            # of the entry existing.
            self._call_soon(self._register_stream, rid)
            self._resolve(cmd.future, value=rid)
        elif cmd.kind == "cancel":
            try:
                self.engine.cancel(cmd.request_id)
            except (KeyError, ValueError):
                pass  # already finished or never existed
            self._call_soon(self._close_stream, cmd.request_id)

    def _has_work(self) -> bool:
        return self.engine.has_work

    def _dispatch(self, out: StepOutput) -> None:
        self._call_soon(self._deliver, out)

    def _fail_open_streams(self, exc: BaseException) -> None:
        for rid in list(self._streams):
            self._call_soon(self._deliver_error, rid, exc)

    # --- event-loop side ---------------------------------------------------------
    # These run ON the loop thread via call_soon_threadsafe, so they are the only code
    # allowed to touch self._streams' asyncio primitives.

    def _register_stream(self, request_id: int) -> None:
        self._streams.setdefault(request_id, _Stream())

    def _deliver(self, out: StepOutput) -> None:
        st = self._streams.get(out.request_id)
        if st is None or st.done:
            return
        st.queue.put_nowait(out)
        if out.finished:
            st.done = True
            st.retired = True  # the engine freed the slot in MetalEngine._retire
            st.queue.put_nowait(_END)

    def _deliver_error(self, request_id: int, exc: BaseException) -> None:
        st = self._streams.get(request_id)
        if st is None or st.done:
            return
        st.done = True
        st.queue.put_nowait(exc)
        st.queue.put_nowait(_END)

    def _close_stream(self, request_id: int) -> None:
        st = self._streams.get(request_id)
        if st is None or st.done:
            return
        st.done = True
        st.queue.put_nowait(_END)

    def release(self, request_id: int) -> None:
        """Give up a request's channel, CANCELLING it unless the engine already retired it.

        Handlers call this from a finally block, so it runs on the abnormal exits too: a
        stream that ended in an error (the worker reported a failed step through
        `_deliver_error`), or a handler killed by CancelledError because the client hung
        up. Popping the channel is not enough on those paths -- `_streams` is only the
        delivery side, and `MetalEngine._retire` is the ONLY thing that returns a seq_id
        to the pool. Dropping the channel alone therefore stranded the request inside the
        engine: its slot was never freed, and since `MetalEngine.has_work` is "any request
        not finished", the worker kept re-running the same failing step forever, at one
        core and tens of thousands of wasted decodes.

        The cancel lives here rather than in each handler's finally block because the two
        surfaces have four such blocks (streaming and non-streaming x OpenAI and
        Anthropic) and any route added later would need a fifth; "release" is exactly the
        point where the caller stops being able to consume the request, which is what
        makes it the right place to decide the request must die.

        Normal completion does NOT cancel: `_deliver` marks the channel `retired` when it
        sees the engine's finished StepOutput, i.e. after `_retire` has already freed the
        slot. A cancel then would be a no-op anyway (`MetalEngine.cancel` returns False
        for a finished request), but not posting it keeps the worker's queue honest.
        """
        st = self._streams.pop(request_id, None)
        if st is not None and not st.retired:
            self._post_cancel(request_id)

    # --- plumbing ----------------------------------------------------------------

    def _call_soon(self, fn, *args) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass  # loop shut down under us during teardown

    def _resolve(self, fut, *, value=None, exc: BaseException | None = None) -> None:
        if fut is None:
            return

        def _set() -> None:
            if fut.done():
                return
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(value)

        self._call_soon(_set)

    def _check_worker(self) -> None:
        if self._thread is None:
            raise RuntimeError("AsyncEngine.start() has not been called")
        if self._worker_error is not None:
            raise RuntimeError(f"engine thread died: {self._worker_error!r}")
