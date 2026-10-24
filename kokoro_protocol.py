"""Shared file locations and message framing for the Kokoro server and client.

The server listens on a Unix domain socket. Every message in both directions
is one frame:

    4 bytes     big-endian length of the JSON header
    N bytes     JSON header (UTF-8)
    M bytes     raw payload; M is the header's "audio_bytes" value (default 0)

Requests (client -> server), one frame per connection:

    {"command": "speak", "text": "...", "speed": 1.0}
    {"command": "stop"}

Responses (server -> client) to "speak":

    {"type": "hello", "device": "mps"}
    {"type": "chunk", "index": 0, "graphemes": "...", "phonemes": "...",
     "audio_bytes": M} followed by M bytes of mono float32 PCM at SAMPLERATE
    ... more chunks ...
    {"type": "done"}

Responses to "stop": {"type": "done"}, then the server exits.
"""

import io
import json
import os
import socket
import struct
from pathlib import Path
from typing import Any

SAMPLERATE = 24000

HEADER_LENGTH = struct.Struct(">I")


def _state_dir() -> Path:
    """Return the per-user directory for the socket, lock, and log file.

    Uses $XDG_RUNTIME_DIR on Linux (set by systemd for login sessions and user
    services) and ~/.cache elsewhere, e.g. on macOS.
    """
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime_dir) if runtime_dir else Path.home() / ".cache"
    return base / "kokoro-server"


STATE_DIR = _state_dir()
SOCKET_PATH = STATE_DIR / "kokoro-server.sock"
LOCK_PATH = STATE_DIR / "kokoro-server.lock"
LOG_PATH = STATE_DIR / "kokoro-server.log"


def ensure_state_dir() -> None:
    """Create the state directory, readable by the current user only."""
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)


def send_frame(sock: socket.socket, header: dict[str, Any], audio: bytes = b"") -> None:
    """Send one frame with the given JSON header and optional audio payload."""
    if audio:
        header = header | {"audio_bytes": len(audio)}
    header_bytes = json.dumps(header).encode()
    sock.sendall(HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + audio)


def _exactly(data: bytes, expected_length: int) -> bytes:
    """Return data, or raise if the connection ended in the middle of a frame."""
    if len(data) != expected_length:
        msg = f"connection closed mid-frame ({len(data)} of {expected_length} bytes)"
        raise ConnectionError(msg)
    return data


def read_frame(rfile: io.BufferedIOBase) -> tuple[dict[str, Any], bytes] | None:
    """Read one frame; return (header, audio), or None if the peer closed the
    connection cleanly before the frame started.
    """
    length_bytes = rfile.read(HEADER_LENGTH.size)
    if not length_bytes:
        return None
    (header_length,) = HEADER_LENGTH.unpack(_exactly(length_bytes, HEADER_LENGTH.size))
    header = json.loads(_exactly(rfile.read(header_length), header_length))
    audio_length = header.get("audio_bytes", 0)
    audio = _exactly(rfile.read(audio_length), audio_length) if audio_length else b""
    return header, audio
