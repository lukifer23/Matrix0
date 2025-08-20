# Matrix0 WebUI

The WebUI provides a browser interface for playing and evaluating games against
Matrix0 and Stockfish.

## Running

```
uvicorn webui.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in your browser. Each new game uses a dedicated
WebSocket (`/ws/{game_id}`) that streams the current FEN, game result and
evaluation data. The REST endpoints (`/move`, `/engine-move`, `/eval`) remain
available for fallbacks but all updates are also pushed over the socket.

Multiple games can be played simultaneously. Every game opened creates a new
tab; switching tabs restores the board, move list and evaluation chart for that
game.
