"""
FileToLink – api.py  (production-grade rewrite)

Key improvements vs original:
 • Connection pool per DC — senders are reused, not created per request
 • No DC-level lock — concurrent streams work in parallel
 • Fast seek — offset jumps directly to the requested byte range
 • Multi-browser / multi-file support via asyncio.Semaphore per pool slot
 • Proper generator cleanup with try/finally + shield
 • FloodWait handled with exponential back-off per sender
 • Vercel: NOT supported (needs persistent process) — deploy on
   HuggingFace Spaces / Railway / Render / Fly.io instead
"""

import os
import asyncio
import math
import socket
import urllib.parse
from collections import defaultdict
from contextlib import asynccontextmanager
from math import ceil, floor
from mimetypes import guess_type
from datetime import datetime
from typing import Optional, List, AsyncGenerator, Union, Awaitable, DefaultDict, Dict

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from telethon import utils, TelegramClient
from telethon.errors import FloodWaitError
from telethon.network import MTProtoSender
from telethon.sessions import MemorySession
from telethon.tl.alltlobjects import LAYER
from telethon.tl.custom import Message
from telethon.tl.functions import InvokeWithLayerRequest
from telethon.tl.functions.auth import ExportAuthorizationRequest, ImportAuthorizationRequest
from telethon.tl.functions.upload import GetFileRequest
from telethon.tl.types import (
    Document,
    InputFileLocation,
    InputDocumentFileLocation,
    InputPhotoFileLocation,
    InputPeerPhotoFileLocation,
)

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
try:
    from config import API_ID, API_HASH, BOT_TOKEN, LOG_CHANNEL_ID
    from utils import LOGGER
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER = logging.getLogger(__name__)
    API_ID         = int(os.getenv("API_ID", 0))
    API_HASH       = os.getenv("API_HASH", "")
    BOT_TOKEN      = os.getenv("BOT_TOKEN", "")
    LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID", "0"))

CUSTOM_DOMAIN         = os.getenv("CUSTOM_DOMAIN")
HEROKU_APP_NAME       = os.getenv("HEROKU_APP_NAME")
RENDER_EXTERNAL_URL   = os.getenv("RENDER_EXTERNAL_URL")
RAILWAY_PUBLIC_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN")
RAILWAY_STATIC_URL    = os.getenv("RAILWAY_STATIC_URL")
FLY_APP_NAME          = os.getenv("FLY_APP_NAME")
VERCEL_URL            = os.getenv("VERCEL_URL")

# ─────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────
TypeLocation = Union[
    Document,
    InputDocumentFileLocation,
    InputPeerPhotoFileLocation,
    InputFileLocation,
    InputPhotoFileLocation,
]

# ─────────────────────────────────────────────────────
# SERVER CONFIG
# ─────────────────────────────────────────────────────
class Telegram:
    API_ID     = API_ID
    API_HASH   = API_HASH
    BOT_TOKEN  = BOT_TOKEN
    CHANNEL_ID = LOG_CHANNEL_ID

class Server:
    BIND_ADDRESS = "0.0.0.0"
    PORT         = int(os.getenv("PORT", 7860))
    BASE_URL     = None

templates = Jinja2Templates(directory="templates")

error_messages = {
    400: "Invalid request.",
    401: "File code is required to download the file.",
    403: "Invalid file code.",
    404: "File not found.",
    416: "Invalid range.",
    500: "Internal server error.",
    503: "Service temporarily unavailable.",
}

# ─────────────────────────────────────────────────────
# CHUNK / POOL SETTINGS  (tune here)
# ─────────────────────────────────────────────────────
CHUNK_SIZE        = 512 * 1024   # 512 KB per Telegram GetFile call
POOL_SIZE_PER_DC  = 5            # reusable senders kept alive per DC
MAX_CONCURRENT    = 20           # total simultaneous HTTP stream requests
PREFETCH_CHUNKS   = 3            # how many chunks to prefetch ahead


# ─────────────────────────────────────────────────────
# SENDER POOL  — one pool per DC, shared across requests
# ─────────────────────────────────────────────────────
class SenderPool:
    """
    Keeps a fixed number of MTProtoSender connections alive per DC.
    Requests borrow a sender, use it, then return it.
    This eliminates the ~2 s connect overhead on every range request.
    """

    def __init__(self, client: "FileToLinkAPI", dc_id: int, size: int = POOL_SIZE_PER_DC):
        self.client   = client
        self.dc_id    = dc_id
        self.size     = size
        self._queue: asyncio.Queue[MTProtoSender] = asyncio.Queue()
        self._auth_key = None
        self._ready    = False

    async def _make_sender(self) -> MTProtoSender:
        dc     = await self.client._get_dc(self.dc_id)
        sender = MTProtoSender(self._auth_key, loggers=self.client._log)
        await sender.connect(
            self.client._connection(
                dc.ip_address, dc.port, dc.id,
                loggers=self.client._log,
                proxy=self.client._proxy,
            )
        )
        if not self._auth_key:
            auth = await self.client(ExportAuthorizationRequest(self.dc_id))
            self.client._init_request.query = ImportAuthorizationRequest(
                id=auth.id, bytes=auth.bytes
            )
            req = InvokeWithLayerRequest(LAYER, self.client._init_request)
            await sender.send(req)
            self._auth_key = sender.auth_key
        return sender

    async def warm_up(self):
        if self._ready:
            return
        # Check if same DC as bot — reuse session auth_key
        if self.client.session.dc_id == self.dc_id:
            self._auth_key = self.client.session.auth_key
        senders = await asyncio.gather(
            *[self._make_sender() for _ in range(self.size)],
            return_exceptions=True,
        )
        for s in senders:
            if isinstance(s, Exception):
                LOGGER.warning("Pool sender init error for DC %s: %s", self.dc_id, s)
            else:
                await self._queue.put(s)
        self._ready = True
        LOGGER.info("SenderPool DC%s ready with %s senders", self.dc_id, self._queue.qsize())

    async def acquire(self) -> MTProtoSender:
        """Get a sender from the pool (waits if all are busy)."""
        return await self._queue.get()

    async def release(self, sender: MTProtoSender):
        """Return a sender to the pool; replace it if disconnected."""
        if not sender.is_connected():
            LOGGER.debug("Sender DC%s disconnected, replacing", self.dc_id)
            try:
                sender = await self._make_sender()
            except Exception as e:
                LOGGER.error("Failed to replace sender DC%s: %s", self.dc_id, e)
                return  # drop it — pool shrinks but won't deadlock
        await self._queue.put(sender)

    async def close(self):
        while not self._queue.empty():
            sender = self._queue.get_nowait()
            try:
                await sender.disconnect()
            except Exception:
                pass


# Global pool registry  {dc_id: SenderPool}
_sender_pools: Dict[int, SenderPool] = {}


async def get_pool(client: "FileToLinkAPI", dc_id: int) -> SenderPool:
    if dc_id not in _sender_pools:
        pool = SenderPool(client, dc_id)
        _sender_pools[dc_id] = pool
        await pool.warm_up()
    elif not _sender_pools[dc_id]._ready:
        await _sender_pools[dc_id].warm_up()
    return _sender_pools[dc_id]


# ─────────────────────────────────────────────────────
# CHUNK FETCHER  — single GetFile call with FloodWait retry
# ─────────────────────────────────────────────────────
async def fetch_chunk(
    client: "FileToLinkAPI",
    sender: MTProtoSender,
    location: TypeLocation,
    offset: int,
    limit: int,
    max_retries: int = 5,
) -> bytes:
    request = GetFileRequest(location, offset=offset, limit=limit)
    for attempt in range(max_retries):
        try:
            result = await client._call(sender, request)
            return result.bytes
        except FloodWaitError as e:
            wait = e.seconds + 1
            LOGGER.warning("FloodWait %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
            await asyncio.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                LOGGER.warning("Chunk error attempt %d: %s", attempt + 1, e)
            else:
                raise
    raise RuntimeError(f"Failed to fetch chunk at offset {offset} after {max_retries} retries")


# ─────────────────────────────────────────────────────
# STREAMING GENERATOR
# Handles seek (arbitrary from_bytes), prefetch, multi-sender
# ─────────────────────────────────────────────────────
async def stream_file_chunks(
    client: "FileToLinkAPI",
    location: TypeLocation,
    dc_id: int,
    from_bytes: int,
    until_bytes: int,
) -> AsyncGenerator[bytes, None]:
    """
    Streams bytes [from_bytes, until_bytes] inclusive.
    Uses multiple pool senders in parallel for throughput.
    Handles arbitrary seek offsets correctly.
    """
    pool = await get_pool(client, dc_id)

    # Align start offset to chunk boundary
    offset          = from_bytes - (from_bytes % CHUNK_SIZE)
    first_part_cut  = from_bytes - offset
    req_length      = until_bytes - from_bytes + 1
    total_parts     = ceil((until_bytes + 1) / CHUNK_SIZE) - floor(offset / CHUNK_SIZE)

    bytes_sent = 0
    current    = 0

    # We'll pipeline PREFETCH_CHUNKS futures at a time
    # Each future = (sender, asyncio.Task<chunk>)
    pending: list = []

    def next_offset(part_idx: int) -> int:
        return offset + part_idx * CHUNK_SIZE

    async def launch_task(part_idx: int):
        s = await pool.acquire()
        off = next_offset(part_idx)
        task = asyncio.ensure_future(
            fetch_chunk(client, s, location, off, CHUNK_SIZE)
        )
        return s, task

    try:
        # Pre-fill pipeline
        for i in range(min(PREFETCH_CHUNKS, total_parts)):
            pending.append(await launch_task(i))
        next_to_launch = PREFETCH_CHUNKS

        while pending:
            sender, task = pending.pop(0)

            try:
                chunk = await asyncio.wait_for(asyncio.shield(task), timeout=90.0)
            except asyncio.TimeoutError:
                LOGGER.warning("Chunk timeout at part %s", current)
                await pool.release(sender)
                current += 1
                continue
            except Exception as e:
                LOGGER.error("Chunk error at part %s: %s", current, e)
                await pool.release(sender)
                raise

            await pool.release(sender)

            # Trim first / last chunks
            if total_parts == 1:
                data = chunk[first_part_cut : first_part_cut + req_length]
            elif current == 0:
                data = chunk[first_part_cut:]
            elif current == total_parts - 1:
                last_cut = (until_bytes % CHUNK_SIZE) + 1
                data = chunk[:last_cut]
            else:
                data = chunk

            # Don't overshoot
            remaining = req_length - bytes_sent
            if len(data) > remaining:
                data = data[:remaining]

            if data:
                yield data
                bytes_sent += len(data)

            current += 1

            if bytes_sent >= req_length:
                break

            # Launch next prefetch
            if next_to_launch < total_parts:
                pending.append(await launch_task(next_to_launch))
                next_to_launch += 1

    except GeneratorExit:
        # Client disconnected — cancel remaining tasks cleanly
        for s, t in pending:
            t.cancel()
            await pool.release(s)
        LOGGER.debug("Client disconnected, stream cancelled")
    except Exception as e:
        for s, t in pending:
            t.cancel()
            await pool.release(s)
        LOGGER.error("Stream error: %s", e)
        raise
    else:
        # Drain unused prefetched tasks back to pool
        for s, t in pending:
            t.cancel()
            await pool.release(s)


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def get_base_url_from_request(request: Request) -> str:
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host  = request.headers.get("x-forwarded-host")
    host            = request.headers.get("host")
    if forwarded_host:
        scheme = forwarded_proto or "https"
        return f"{scheme}://{forwarded_host}"
    elif host:
        scheme = "https" if forwarded_proto == "https" else "http"
        return f"{scheme}://{host}"
    return Server.BASE_URL or f"http://localhost:{Server.PORT}"


def abort(status_code: int = 500, description: str = None):
    raise HTTPException(
        status_code=status_code,
        detail=description or error_messages.get(status_code),
    )


def sanitize_filename(filename: str) -> str:
    try:
        filename.encode("latin-1")
        return filename
    except UnicodeEncodeError:
        return urllib.parse.quote(filename, safe="")


def get_file_properties(message: Message):
    file_name = message.file.name
    file_size = message.file.size or 0
    mime_type = message.file.mime_type
    if not file_name:
        attributes = {
            "video":      "mp4",
            "audio":      "mp3",
            "voice":      "ogg",
            "photo":      "jpg",
            "video_note": "mp4",
        }
        for attribute, extension in attributes.items():
            media = getattr(message, attribute, None)
            if media:
                file_type   = attribute
                file_format = extension
                break
        else:
            abort(400, "Invalid media type.")
        date      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{file_type}-{date}.{file_format}"
    if not mime_type:
        mime_type = guess_type(file_name)[0] or "application/octet-stream"
    return file_name, file_size, mime_type


# ─────────────────────────────────────────────────────
# TELEGRAM CLIENT
# ─────────────────────────────────────────────────────
class FileToLinkAPI(TelegramClient):
    def __init__(self, api_id, api_hash, bot_token):
        LOGGER.info("Creating Telethon FileToLink Client with MemorySession")
        super().__init__(
            MemorySession(),
            api_id,
            api_hash,
            connection_retries=-1,
            timeout=120,
            flood_sleep_threshold=60,
            request_retries=5,
            auto_reconnect=True,
        )
        self.bot_token      = bot_token
        self.max_concurrent = MAX_CONCURRENT
        self.semaphore      = asyncio.Semaphore(self.max_concurrent)
        # Message cache: avoid re-fetching the same message for every range request
        self._msg_cache: Dict[int, tuple] = {}   # file_id → (message, timestamp)
        self._cache_ttl = 300                     # 5 minutes

    async def start_api(self):
        while True:
            try:
                await self.start(bot_token=self.bot_token)
                LOGGER.info("Telethon FileToLink Client started successfully!")
                LOGGER.info("Max concurrent requests: %s", self.max_concurrent)
                # Warm up pools for the bot's own DC immediately
                own_dc = self.session.dc_id
                await get_pool(self, own_dc)
                return
            except FloodWaitError as e:
                LOGGER.warning("FloodWait during startup: waiting %d seconds", e.seconds)
                await asyncio.sleep(e.seconds)

    async def get_cached_message(self, file_id: int):
        """Cache messages to avoid Telegram round-trip on every byte-range request."""
        now = asyncio.get_event_loop().time()
        if file_id in self._msg_cache:
            msg, ts = self._msg_cache[file_id]
            if now - ts < self._cache_ttl:
                return msg
        msg = await asyncio.wait_for(
            self.get_messages(Telegram.CHANNEL_ID, ids=file_id),
            timeout=15.0,
        )
        if msg:
            self._msg_cache[file_id] = (msg, now)
        return msg


# ─────────────────────────────────────────────────────
# BASE URL DETECTION
# ─────────────────────────────────────────────────────
async def get_local_ip() -> str:
    loop = asyncio.get_event_loop()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    try:
        await loop.sock_connect(s, ("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


async def detect_base_url() -> str:
    if CUSTOM_DOMAIN:
        base_url = (
            f"https://{CUSTOM_DOMAIN}"
            if not CUSTOM_DOMAIN.startswith("http")
            else CUSTOM_DOMAIN
        )
        LOGGER.info("Using CUSTOM_DOMAIN: %s", base_url)
        return base_url
    for env, label, fmt in [
        (HEROKU_APP_NAME,       "Heroku",          "https://{}.herokuapp.com"),
        (FLY_APP_NAME,          "Fly.io",           "https://{}.fly.dev"),
        (RAILWAY_PUBLIC_DOMAIN, "Railway (domain)", "https://{}"),
        (VERCEL_URL,            "Vercel",           "https://{}"),
    ]:
        if env:
            url = fmt.format(env)
            LOGGER.info("Detected %s: %s", label, url)
            return url
    for env, label in [
        (RENDER_EXTERNAL_URL, "Render"),
        (RAILWAY_STATIC_URL,  "Railway (static)"),
    ]:
        if env:
            url = env.rstrip("/")
            LOGGER.info("Detected %s: %s", label, url)
            return url
    ip = await get_local_ip()
    url = f"http://{ip}:{Server.PORT}"
    LOGGER.info("No platform detected, using local IP: %s", url)
    return url


# ─────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    Server.BASE_URL = await detect_base_url()
    await api_instance.start_api()
    LOGGER.info("API running on: %s", Server.BASE_URL)
    yield
    LOGGER.info("Shutting down — closing sender pools")
    for pool in _sender_pools.values():
        await pool.close()
    await api_instance.disconnect()


app = FastAPI(lifespan=lifespan, title="FileToLink")


# ─────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>FileToLink API</h1><p>Status: Running</p><p>Error loading template: {e}</p>"
        )


async def _resolve_file_and_validate(file_id: int, code: str):
    """Fetch (cached) and validate a message."""
    try:
        file = await api_instance.get_cached_message(file_id)
    except asyncio.TimeoutError:
        abort(500, "Request timeout")
    except Exception as e:
        LOGGER.error("Failed to retrieve message %s: %s", file_id, e)
        abort(500)
    if not file:
        abort(404)
    if code != file.raw_text:
        abort(403)
    return file


@app.get("/stream/{file_id}", response_class=HTMLResponse)
async def stream_page(file_id: int, request: Request):
    code = request.query_params.get("code") or abort(401)
    file = await _resolve_file_and_validate(file_id, code)
    file_name, file_size, mime_type = get_file_properties(file)
    base_url     = get_base_url_from_request(request)
    file_url     = f"{base_url}/dl/{file_id}?code={urllib.parse.quote(code)}"
    file_size_mb = f"{file_size / (1024 * 1024):.2f} MB"
    try:
        return templates.TemplateResponse(
            request=request, name="player.html",
            context={
                "file_name":    file_name,
                "file_size_mb": file_size_mb,
                "file_url":     file_url,
                "mime_type":    mime_type,
            },
        )
    except Exception:
        return HTMLResponse(
            content=f"<h1>{file_name}</h1><p>Size: {file_size_mb}</p>"
                    f"<a href='{file_url}'>Download</a>"
        )


@app.get("/dl/{file_id}")
async def transmit_file(file_id: int, request: Request):
    code = request.query_params.get("code") or abort(401)

    # =stream suffix → redirect to player page
    if code.endswith("=stream"):
        code_clean = code[:-7]
        base_url   = get_base_url_from_request(request)
        file       = await _resolve_file_and_validate(file_id, code_clean)
        file_name, file_size, mime_type = get_file_properties(file)
        file_url     = f"{base_url}/dl/{file_id}?code={urllib.parse.quote(code_clean)}"
        file_size_mb = f"{file_size / (1024 * 1024):.2f} MB"
        try:
            return templates.TemplateResponse(
                request=request, name="player.html",
                context={
                    "file_name":    file_name,
                    "file_size_mb": file_size_mb,
                    "file_url":     file_url,
                    "mime_type":    mime_type,
                },
            )
        except Exception:
            return HTMLResponse(
                content=f"<h1>{file_name}</h1><p>Size: {file_size_mb}</p>"
                        f"<a href='{file_url}'>Download</a>"
            )

    # ── Normal / range download ──────────────────────
    # Use semaphore ONLY to throttle concurrent metadata lookups,
    # NOT the streaming itself (streaming is I/O-bound, not CPU-bound).
    async with api_instance.semaphore:
        file = await _resolve_file_and_validate(file_id, code)
        file_name, file_size, mime_type = get_file_properties(file)
        dc_id, location = utils.get_input_location(file.media)

    me_username = (await api_instance.get_me()).username
    LOGGER.info(
        "Download request - File ID: %s API: @%s | File: %s  Size: %s  Type: %s",
        file_id, me_username, file_name, file_size, mime_type,
    )

    # Parse Range header
    range_header = request.headers.get("Range", "")
    if range_header:
        range_str   = range_header.replace("bytes=", "")
        parts       = range_str.split("-")
        from_bytes  = int(parts[0]) if parts[0] else 0
        until_bytes = int(parts[1]) if parts[1] else file_size - 1
        LOGGER.info("Range request: %s-%s/%s", from_bytes, until_bytes, file_size)
    else:
        from_bytes  = 0
        until_bytes = file_size - 1
        LOGGER.info("Full file request: %s bytes", file_size)

    if until_bytes >= file_size or from_bytes < 0 or until_bytes < from_bytes:
        abort(416, "Invalid range.")

    req_length         = until_bytes - from_bytes + 1
    sanitized_filename = sanitize_filename(file_name)

    headers = {
        "Content-Type":        mime_type,
        "Content-Range":       f"bytes {from_bytes}-{until_bytes}/{file_size}",
        "Content-Length":      str(req_length),
        "Content-Disposition": f"attachment; filename*=UTF-8''{sanitized_filename}",
        "Accept-Ranges":       "bytes",
        # Allow browsers/CDN to cache chunks — safe since file content never changes
        "Cache-Control":       "public, max-age=3600",
        "Connection":          "keep-alive",
    }

    # Warm up pool for this file's DC proactively (no-op if already warm)
    asyncio.ensure_future(get_pool(api_instance, dc_id))

    return StreamingResponse(
        stream_file_chunks(api_instance, location, dc_id, from_bytes, until_bytes),
        headers=headers,
        status_code=206 if range_header else 200,
        media_type=mime_type,
    )


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return Response(
        content=exc.detail or error_messages.get(exc.status_code),
        status_code=exc.status_code,
    )


# ─────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────
api_instance = FileToLinkAPI(
    api_id   = Telegram.API_ID,
    api_hash = Telegram.API_HASH,
    bot_token= Telegram.BOT_TOKEN,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "__main__:app",
        host               = Server.BIND_ADDRESS,
        port               = Server.PORT,
        workers            = 1,          # must be 1 — asyncio state is process-local
        loop               = "uvloop",
        limit_concurrency  = 500,
        backlog            = 1024,
        timeout_keep_alive = 300,        # keep HTTP connections alive between range requests
        h11_max_incomplete_event_size = 16_777_216,
    )
