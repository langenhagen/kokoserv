#!/usr/bin/env python3
"""Keep the Kokoro TTS model loaded and synthesize speech for kokoro_client.py.

Loads the model once, then serves requests on a Unix domain socket, one at a
time. See kokoro_protocol.py for the socket location and the message format.

A lock file ensures that at most one server per user runs. A second server
exits with code 0 right away, without loading the model.

Usage:

    kokoro_server.py      load the model, then serve until stopped

Stop the server with `kokoro_client.py --stop-server`, SIGTERM, or Ctrl-C.
"""

import fcntl
import logging
import signal
import socketserver
from contextlib import suppress
from typing import IO, Any, cast

from kokoro_protocol import (
    DEFAULT_VOICE,
    LOCK_PATH,
    LOG_PATH,
    SOCKET_PATH,
    ensure_state_dir,
    read_frame,
    send_frame,
)

logger = logging.getLogger("kokoro-server")


def _release_cached_device_memory(device: str) -> None:
    """Return the GPU allocator cache to the system after each request.

    On Apple silicon the GPU shares the system RAM, so an idle server should not
    hold on to buffers from the last long text.
    """
    import torch  # noqa: PLC0415 - loaded by main() already; local for fast import

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


class SpeakHandler(socketserver.StreamRequestHandler):
    """Handle one request: synthesize text and stream audio back, or stop."""

    def _speak(
        self, server: "KokoroServer", text: str, speed: float, voice: str
    ) -> None:
        """Synthesize text chunk by chunk and send each chunk once it is ready."""
        logger.info(
            "speak: %d characters at speed %s with voice %s", len(text), speed, voice
        )
        pipeline = server.pipeline_for(voice)
        send_frame(self.request, {"type": "hello", "device": server.device})
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        for index, (graphemes, phonemes, audio) in enumerate(generator):
            header = {
                "type": "chunk",
                "index": index,
                "graphemes": graphemes,
                "phonemes": phonemes,
            }
            send_frame(self.request, header, audio.numpy().tobytes())
        send_frame(self.request, {"type": "done"})
        logger.info("speak: done")

    def handle(self) -> None:
        """Read the request frame and dispatch on its command."""
        server = cast("KokoroServer", self.server)
        frame = read_frame(self.rfile)
        if frame is None:
            return
        request, _ = frame
        if request["command"] == "stop":
            logger.info("stop requested")
            server.stop_requested = True
            send_frame(self.request, {"type": "done"})
            return
        try:
            self._speak(server, request["text"], request["speed"], request["voice"])
        except (BrokenPipeError, ConnectionResetError):
            logger.info("client disconnected, synthesis aborted")
        finally:
            _release_cached_device_memory(server.device)


class KokoroServer(socketserver.UnixStreamServer):
    """Unix socket server that holds the loaded Kokoro pipelines, one per language."""

    # pipeline is a kokoro.KPipeline; annotating it would import torch at load time.
    def __init__(self, pipeline: Any, device: str) -> None:  # noqa: ANN401 - see above
        """Bind the socket and keep the pipeline for the request handler."""
        super().__init__(str(SOCKET_PATH), SpeakHandler)
        self.pipelines = {pipeline.lang_code: pipeline}
        self.device = device
        self.stop_requested = False

    def pipeline_for(self, voice: str) -> Any:  # noqa: ANN401 - see __init__
        """Return the pipeline for the voice's language, creating it on first use.

        All pipelines share the model of the first one, so a new language costs
        no second model load.
        """
        lang_code = voice[0]
        if lang_code not in self.pipelines:
            from kokoro import KPipeline  # noqa: PLC0415 - loaded by main() already

            logger.info("creating pipeline for language code %s", lang_code)
            model = next(iter(self.pipelines.values())).model
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code, model=model)
        return self.pipelines[lang_code]


def _acquire_single_instance_lock() -> IO[str] | None:
    """Return the locked lock file, or None if another server holds the lock.

    The caller must keep the returned file object open for the server's lifetime.
    """
    lock_file = LOCK_PATH.open("w")  # stays open to hold the lock
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def _load_pipeline() -> tuple[Any, str]:
    """Load the Kokoro pipeline on the best available device and warm it up."""
    # Importing torch and the Kokoro model machinery takes seconds to load.
    import torch  # noqa: PLC0415 - see comment above
    from kokoro import KPipeline  # noqa: PLC0415 - see comment above

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("device: %s", device)
    pipeline = KPipeline(lang_code=DEFAULT_VOICE[0], device=device)
    # Load the default voice and run the device kernels once, so that the first
    # real request is as fast as all later ones.
    for _ in pipeline("Hello.", voice=DEFAULT_VOICE):
        pass
    _release_cached_device_memory(device)
    return pipeline, device


def _raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    """Turn SIGTERM into the same clean shutdown path as Ctrl-C."""
    raise KeyboardInterrupt


def main() -> None:
    """Load the model once, then serve requests until stopped."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_state_dir()
    lock_file = _acquire_single_instance_lock()
    if lock_file is None:
        logger.info("another kokoro server is already running; exiting")
        return

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    logger.info("loading model (log file for autostarted servers: %s)", LOG_PATH)
    pipeline, device = _load_pipeline()

    # A socket file left over from a crashed server blocks bind(). Holding the
    # lock guarantees that no live server uses it.
    SOCKET_PATH.unlink(missing_ok=True)
    server = KokoroServer(pipeline, device)
    logger.info("ready on %s", SOCKET_PATH)
    try:
        while not server.stop_requested:
            server.handle_request()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        server.server_close()
        SOCKET_PATH.unlink(missing_ok=True)
        lock_file.close()
        logger.info("stopped")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
