"""Run the Skyview app on one or two listeners in a single process.

* HTTP  on ``SKYVIEW_HTTP_HOST:SKYVIEW_HTTP_PORT`` (default ``0.0.0.0:8501``):
  the long-standing listener. Local tooling (deploy health check, cron, QA
  scripts) talks to it on 127.0.0.1. Once the CDN talks HTTPS to the origin it
  is bound to 127.0.0.1 only.
* HTTPS on ``SKYVIEW_TLS_HOST:SKYVIEW_TLS_PORT`` (default ``0.0.0.0:8443``),
  only when ``SKYVIEW_TLS_CERT`` and ``SKYVIEW_TLS_KEY`` point to readable
  files: the listener the CDN64 edge servers use, so the edge-to-origin hop is
  encrypted.

Both listeners share one app instance, so caches are not duplicated and the
app's lifespan (startup/shutdown) runs once. A missing certificate never takes
the site down: the HTTPS listener is skipped with a warning.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from dataclasses import dataclass
from typing import Optional

import uvicorn

log = logging.getLogger("skyview.serve")


@dataclass(frozen=True)
class Listener:
    host: str
    port: int
    certfile: Optional[str] = None
    keyfile: Optional[str] = None

    @property
    def tls(self) -> bool:
        return bool(self.certfile and self.keyfile)

    def describe(self) -> str:
        return f"{'https' if self.tls else 'http'}://{self.host}:{self.port}"


def _port(env: dict, name: str, default: int) -> int:
    try:
        value = int(str(env.get(name, default)).strip())
    except ValueError:
        log.warning("%s is not a port number; using %s", name, default)
        return default
    return value if 0 < value < 65536 else default


def listeners_from_env(env: Optional[dict] = None) -> list[Listener]:
    """HTTP listener always; HTTPS listener when a readable cert + key are configured."""
    env = os.environ if env is None else env
    out = [
        Listener(
            host=str(env.get("SKYVIEW_HTTP_HOST") or "0.0.0.0").strip(),
            port=_port(env, "SKYVIEW_HTTP_PORT", 8501),
        )
    ]
    cert = str(env.get("SKYVIEW_TLS_CERT") or "").strip()
    key = str(env.get("SKYVIEW_TLS_KEY") or "").strip()
    if cert or key:
        readable = all(p and os.path.isfile(p) and os.access(p, os.R_OK) for p in (cert, key))
        if readable:
            out.append(
                Listener(
                    host=str(env.get("SKYVIEW_TLS_HOST") or "0.0.0.0").strip(),
                    port=_port(env, "SKYVIEW_TLS_PORT", 8443),
                    certfile=cert,
                    keyfile=key,
                )
            )
        else:
            log.warning(
                "TLS listener disabled: certificate %r or key %r missing/unreadable", cert, key
            )
    if len(out) == 2 and out[0].port == out[1].port:
        log.warning("HTTP and HTTPS both configured on port %s; HTTPS listener disabled", out[0].port)
        out.pop()
    return out


class _GroupMember(uvicorn.Server):
    """A uvicorn server that leaves signal handling to the group (see run)."""

    @contextlib.contextmanager
    def capture_signals(self):  # noqa: D401 - uvicorn hook
        yield


def build_servers(app, listeners: list[Listener], **common) -> list[uvicorn.Server]:
    """One uvicorn server per listener; only the first runs the app lifespan."""
    servers = []
    for i, lst in enumerate(listeners):
        config = uvicorn.Config(
            app,
            host=lst.host,
            port=lst.port,
            ssl_certfile=lst.certfile,
            ssl_keyfile=lst.keyfile,
            lifespan="auto" if i == 0 else "off",
            **common,
        )
        servers.append(_GroupMember(config))
    return servers


async def serve_group(servers: list[uvicorn.Server]) -> None:
    """Run all servers; stop all of them on SIGINT/SIGTERM or when any one stops."""

    def request_exit(sig=None, _frame=None):
        for s in servers:
            if s.should_exit and sig == signal.SIGINT:
                s.force_exit = True
            s.should_exit = True

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError):  # not in the main thread (tests)
            previous[sig] = signal.signal(sig, request_exit)
    try:
        # Start the lifespan owner first so startup (cache seeding) completes
        # before the second listener accepts traffic.
        tasks = [asyncio.create_task(servers[0].serve())]
        while not servers[0].started and not tasks[0].done():
            await asyncio.sleep(0.05)
        if not tasks[0].done():
            tasks += [asyncio.create_task(s.serve()) for s in servers[1:]]
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # One listener stopped (signal, bind failure, lifespan error): stop all,
        # so systemd restarts the whole service instead of running half of it.
        request_exit()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(app, listeners: list[Listener], **common) -> None:
    for lst in listeners:
        log.info("Listening on %s", lst.describe())
    asyncio.run(serve_group(build_servers(app, listeners, **common)))
