#!/usr/bin/env python3
"""Drop-in replacement for kokoro-hello's kokoro-speak.py that talks to a
running kokoro_server.py instead of loading the Kokoro model itself.

Reads text from a file, a pipe, or the clipboard, sends it to the server, and
plays the returned speech directly (no file is written). Starts the server in
the background if it is not running; the server then keeps running after this
client exits. The server is never restarted automatically after a failure.

Usage:

    kokoro_client.py FILE              read the text file and speak it
    kokoro_client.py --clipboard       speak the text from the clipboard
    echo "hi" | kokoro_client.py       speak text piped to stdin
    kokoro_client.py FILE --out-file x.wav
        write the synthesized speech from FILE to x.wav instead of speaking
    kokoro_client.py FILE --speed 1.25
        speak the text file at 1.25x speed
    kokoro_client.py --stop-server     stop the running server, freeing its memory

    kokoro_client.py --help        show help
"""

import argparse
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from kokoro_protocol import (
    LOG_PATH,
    SAMPLERATE,
    SOCKET_PATH,
    ensure_state_dir,
    read_frame,
    send_frame,
)

DESCRIPTION = (
    "Kokoro TTS client: read text from a file, a pipe, or the clipboard and "
    "play it back as speech directly (no file is written), or write it to a "
    "file with --out-file. Synthesis runs in a background kokoro server, "
    "which is started on first use and keeps the model loaded."
)
SERVER_SCRIPT = Path(__file__).resolve().parent / "kokoro_server.py"
# First start downloads the model and voice if they are not cached yet.
STARTUP_TIMEOUT_S = 300
STARTUP_POLL_INTERVAL_S = 0.1


def _clipboard_read_command() -> list[str]:
    """Return the OS-appropriate command that prints the clipboard to stdout."""
    if sys.platform == "darwin":
        return ["pbpaste"]
    if shutil.which("wl-paste"):
        return ["wl-paste", "--no-newline"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard", "-o"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--output"]
    msg = "No clipboard utility found (need pbpaste, wl-paste, xclip, or xsel)"
    raise RuntimeError(msg)


def read_clipboard() -> str:
    """Return the text currently stored in the clipboard."""
    return subprocess.check_output(_clipboard_read_command(), text=True)


def _connect() -> socket.socket:
    """Connect to the server socket; raise OSError if no server listens."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(SOCKET_PATH))
    except OSError:
        sock.close()
        raise
    return sock


def _start_server() -> subprocess.Popen[bytes]:
    """Start the server detached from this terminal, logging to LOG_PATH."""
    ensure_state_dir()
    print(f"kokoro: starting server, log: {LOG_PATH}", file=sys.stderr)
    with LOG_PATH.open("ab") as log_file:
        return subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,  # Ctrl-C in this terminal must not reach the server
        )


def connect_or_start_server() -> socket.socket:
    """Connect to the running server, starting it first if necessary."""
    with suppress(FileNotFoundError, ConnectionRefusedError):
        return _connect()

    server_process = _start_server()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        with suppress(FileNotFoundError, ConnectionRefusedError):
            return _connect()
        # Exit code 0 means another server won the start race; keep waiting for it.
        returncode = server_process.poll()
        if returncode not in {None, 0}:
            sys.exit(f"kokoro: server exited with code {returncode}, see {LOG_PATH}")
        if time.monotonic() > deadline:
            sys.exit(
                f"kokoro: server not ready in {STARTUP_TIMEOUT_S} s, see {LOG_PATH}"
            )
        time.sleep(STARTUP_POLL_INTERVAL_S)


def stop_server() -> None:
    """Ask the running server to exit; do nothing if none is running."""
    try:
        sock = _connect()
    except (FileNotFoundError, ConnectionRefusedError):
        print("kokoro: server is not running", file=sys.stderr)
        return
    with sock, sock.makefile("rb") as rfile:
        send_frame(sock, {"command": "stop"})
        read_frame(rfile)
    print("kokoro: server stopped", file=sys.stderr)


def synthesize(text: str, speed: float) -> Iterator[tuple[dict[str, Any], bytes]]:
    """Yield (chunk header, float32 PCM audio) pairs as the server sends them."""
    with connect_or_start_server() as sock, sock.makefile("rb") as rfile:
        send_frame(sock, {"command": "speak", "text": text, "speed": speed})
        while True:
            frame = read_frame(rfile)
            if frame is None:
                sys.exit(f"kokoro: server hung up mid-request, see {LOG_PATH}")
            header, audio = frame
            if header["type"] == "done":
                return
            if header["type"] == "hello":
                print(f"device: {header['device']}")
                continue
            yield header, audio


def _print_chunk(header: dict[str, Any]) -> None:
    """Print chunk details in the same format as kokoro-speak.py."""
    print(header["index"])  # index
    print(header["graphemes"])  # graphemes/text
    print(header["phonemes"])  # phonemes


def write_file(chunks: Iterator[tuple[dict[str, Any], bytes]], out_file: str) -> None:
    """Concatenate all audio chunks into one sound file."""
    import numpy as np  # noqa: PLC0415 - lazy: only --out-file needs it
    import soundfile as sf  # noqa: PLC0415 - see above

    with sf.SoundFile(out_file, "w", channels=1, samplerate=SAMPLERATE) as outfile:
        for header, audio in chunks:
            _print_chunk(header)
            outfile.write(np.frombuffer(audio, dtype=np.float32))


def play(chunks: Iterator[tuple[dict[str, Any], bytes]]) -> None:
    """Play audio chunks back to back on one output stream, without gaps."""
    import sounddevice as sd  # noqa: PLC0415 - lazy: loads PortAudio

    with sd.RawOutputStream(
        samplerate=SAMPLERATE, channels=1, dtype="float32"
    ) as stream:
        for header, audio in chunks:
            _print_chunk(header)
            stream.write(audio)


def main() -> None:
    """Send the text from a file, clipboard, or stdin pipe to the server and play it."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("file", nargs="?", help="text file to speak")
    parser.add_argument(
        "--clipboard",
        "-c",
        action="store_true",
        help="speak text from the clipboard",
    )
    parser.add_argument(
        "--out-file",
        "--out",
        "-o",
        metavar="FILE",
        help="write the synthesized speech to FILE instead of playing it",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1,
        help="narration speed multiplier (default: %(default)s)",
    )
    parser.add_argument(
        "--stop-server",
        action="store_true",
        help="stop the running kokoro server and exit",
    )
    args = parser.parse_args()

    if args.stop_server:
        stop_server()
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.clipboard:
        text = read_clipboard()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_usage()
        sys.exit(2)

    chunks = synthesize(text, args.speed)
    if args.out_file:
        write_file(chunks, args.out_file)
    else:
        play(chunks)


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
