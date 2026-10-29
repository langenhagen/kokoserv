# Kokoserv

A Kokoro TTS server and CLI client for quick and responsive TTS on Linux and macOS.

Keeps the Kokoro TTS model loaded in a background process, so speech starts
without the multi-second model load if used as standalone.

- `kokoro_server.py` loads the model once and serves requests, one at a time.
- `kokoro_client.py` is a drop-in replacement for `kokoro-speak.py` (same
  arguments and output). It plays the audio itself, so Ctrl-C stops playback
  and synthesis immediately.
- `kokoro.sh` runs the client through the project venv; link it into `PATH`
  as `kokoro` in place of kokoro-hello's `kokoro.sh`.
- `kokoro_protocol.py` holds the socket location and the message format.

If not already running, the first CLI invocation starts the server.

## Setup

```bash
uv sync
sudo ln -sf "$PWD/kokoro.sh" /usr/local/bin/kokoro  # link the CLI so you can use it everywhere
```

## Usage

```bash
kokoro FILE                    # speak a text file
kokoro --clipboard             # speak the clipboard
echo "hello" | kokoro          # speak piped text
kokoro FILE --out-file x.wav   # write to a WAV file instead
kokoro FILE --speed 1.25       # speak faster
kokoro FILE --voice bf_emma    # speak with another voice
kokoro --stop-server           # stop the server and free its memory
```

## How It Works

Client and server talk over a Unix domain socket: a file-system path that only
the current user can open, with no TCP port and no network exposure. The
socket, the lock file, and the log file live in `$XDG_RUNTIME_DIR/kokoro-server/`
on Linux and in `~/.cache/kokoro-server/` on macOS.

If no server listens, the client starts one detached from the terminal and
waits until the model is loaded. A lock file guarantees at most one server per
user, so concurrent clients never load the model twice. Nothing restarts the
server after a crash; the next client call starts a fresh one.

## Autostart On macOS

Optional; without it, the first `kokoro` call starts the server. The launchd
agent in `autostart/` starts the server once per login and never restarts it.
Run `kokoro --stop-server` once first, so that the log directory exists.

```bash
cp autostart/com.andreasl.kokoro-server.plist ~/Library/LaunchAgents/
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.andreasl.kokoro-server.plist
```

Remove it again:

```bash
launchctl bootout "gui/$(id -u)/com.andreasl.kokoro-server"
rm ~/Library/LaunchAgents/com.andreasl.kokoro-server.plist
```

## Autostart On Linux

Optional; same behavior as on macOS, via a systemd user service.

```bash
mkdir --parents ~/.config/systemd/user
cp autostart/kokoro-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kokoro-server.service
journalctl --user --unit kokoro-server.service   # view the log
```

Remove it again:

```bash
systemctl --user disable --now kokoro-server.service
rm ~/.config/systemd/user/kokoro-server.service
```

## License

See [LICENSE](LICENSE) file.
