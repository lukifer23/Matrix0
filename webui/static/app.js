let games = {}; // gameId -> { chess, evalData, socket, log }
let currentGameId = null;
let board = null;
let chess = null;
let evalData = null;
let evalChart = null;

function log(gid, msg) {
  const g = games[gid];
  if (!g) return;
  if (!g.log) g.log = [];
  g.log.unshift(`[${new Date().toLocaleTimeString()}] ${msg}`);
  if (gid === currentGameId) {
    document.getElementById('log').textContent = g.log.join('\n');
  }
}

function updateState(fen, turn, finished, result) {
  if (!chess) return;
  chess.load(fen);
  document.getElementById('fen').textContent = fen;
  document.getElementById('turn').textContent = turn;
  if (finished) log(currentGameId, `Game over: ${result}`);
  board.set({ fen, turnColor: (turn === 'w' ? 'white' : 'black') });
  renderMoves();
  renderChart();
}

function renderMoves() {
  if (!chess) return;
  const el = document.getElementById('movelist');
  const hist = chess.history({ verbose: true });
  const out = [];
  for (let i = 0; i < hist.length; i++) {
    const m = hist[i];
    const num = Math.floor(i / 2) + 1;
    if (i % 2 === 0) out.push(`${num}. ${m.san}`);
    else out[out.length - 1] += `  ${m.san}`;
  }
  el.textContent = out.join('\n');
}

async function api(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function addTab(gid) {
  let tabs = document.getElementById('tabs');
  if (!tabs) {
    tabs = document.createElement('div');
    tabs.id = 'tabs';
    tabs.style.display = 'flex';
    tabs.style.gap = '8px';
    const header = document.querySelector('header');
    header.appendChild(tabs);
  }
  const btn = document.createElement('button');
  btn.textContent = gid;
  btn.addEventListener('click', () => setActiveGame(gid));
  tabs.appendChild(btn);
}

function setActiveGame(gid) {
  currentGameId = gid;
  const g = games[gid];
  chess = g.chess;
  evalData = g.evalData;
  document.getElementById('gid').textContent = gid;
  const finished = g.chess.game_over();
  let result = null;
  if (finished) {
    if (g.chess.in_draw()) result = '1/2-1/2';
    else result = (g.chess.turn() === 'w') ? '0-1' : '1-0';
  }
  updateState(g.chess.fen(), g.chess.turn(), finished, result);
  if (g.log) document.getElementById('log').textContent = g.log.join('\n');
  document.getElementById('mval').textContent = (g.lastMat === undefined || g.lastMat === null ? '—' : g.lastMat.toFixed(3));
  document.getElementById('scpf').textContent = (g.lastCp === undefined || g.lastCp === null ? '—' : g.lastCp);
}

function onMessage(gid, ev) {
  const msg = JSON.parse(ev.data);
  const g = games[gid];
  if (!g) return;
  if (msg.type === 'state') {
    g.chess.load(msg.fen);
    if (gid === currentGameId) {
      updateState(msg.fen, msg.turn, msg.game_over, msg.result);
    }
  } else if (msg.type === 'eval') {
    const ply = g.chess.history().length;
    g.evalData.ply.push(ply);
    g.evalData.matrix0.push(msg.matrix0_value);
    g.evalData.stockfish.push(msg.stockfish_cp !== null ? msg.stockfish_cp / 100.0 : null);
    g.lastMat = msg.matrix0_value;
    g.lastCp = msg.stockfish_cp;
    if (gid === currentGameId) {
      renderChart();
      document.getElementById('mval').textContent = (g.lastMat === null ? '—' : g.lastMat.toFixed(3));
      document.getElementById('scpf').textContent = (g.lastCp === null ? '—' : g.lastCp);
    }
  }
}

async function newGame() {
  const white = document.getElementById('white').value;
  const black = document.getElementById('black').value;
  const tc = parseInt(document.getElementById('tc').value || '100', 10);
  const out = await api('/new', { white, black, engine_tc_ms: tc });
  const gid = out.game_id;
  games[gid] = {
    chess: new Chess(out.fen),
    evalData: { ply: [], matrix0: [], stockfish: [] },
    socket: null,
    log: [],
  };
  document.getElementById('tclabel').textContent = tc;
  addTab(gid);
  setActiveGame(gid);
  log(gid, `New game ${gid}: ${white} vs ${black}`);
  const proto = (location.protocol === 'https:' ? 'wss' : 'ws');
  const ws = new WebSocket(`${proto}://${location.host}/ws/${gid}`);
  ws.onmessage = (ev) => onMessage(gid, ev);
  games[gid].socket = ws;
  loadAnalytics();
}

async function pushMove(uci) {
  const gid = currentGameId;
  if (!gid) return;
  await api('/move', { game_id: gid, uci });
  log(gid, `Human plays ${uci}`);
}

async function engineMove() {
  if (!currentGameId) return;
  const turn = chess.turn();
  const color = (turn === 'w' ? 'white' : 'black');
  const engine = document.getElementById(color).value; // who plays this color
  const out = await api('/engine-move', { game_id: currentGameId, engine });
  log(currentGameId, `${engine} plays ${out.uci}`);
  try { await evalPos(); } catch {}
}

async function evalPos() {
  if (!currentGameId) return;
  await api('/eval', { game_id: currentGameId, include_stockfish: true });
}

function initBoard() {
  const startFen = new Chess().fen();
  board = Chessground(document.getElementById('board'), {
    fen: startFen,
    orientation: 'white',
    movable: {
      free: false,
      color: () => 'both',
      events: {
        after: async (orig, dest, meta) => {
          if (!currentGameId) return;
          const move = { from: orig, to: dest, promotion: (meta.promotion || 'q') };
          const uci = move.from + move.to + (move.promotion ? move.promotion : '');
          try {
            await pushMove(uci);
          } catch (e) {
            log(currentGameId, 'Illegal move');
            board.set({ fen: chess.fen() });
          }
          try { await evalPos(); } catch {}
        }
      }
    }
  });
}

function renderChart() {
  if (!evalData) return;
  const ctx = document.getElementById('evalChart').getContext('2d');
  const data = {
    labels: evalData.ply,
    datasets: [
      {
        label: 'Matrix0 value',
        data: evalData.matrix0,
        borderColor: '#5ac8fa',
        backgroundColor: 'rgba(90,200,250,0.15)',
        tension: 0.2,
      },
      {
        label: 'Stockfish eval (pawns)',
        data: evalData.stockfish,
        borderColor: '#a0fa5a',
        backgroundColor: 'rgba(160,250,90,0.15)',
        tension: 0.2,
      }
    ]
  };
  const options = {
    responsive: true,
    plugins: { legend: { labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--fg') || '#fff' } } },
    scales: {
      x: { ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted') } },
      y: { ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted') } }
    }
  };
  if (!evalChart) {
    evalChart = new Chart(ctx, { type: 'line', data, options });
  } else {
    evalChart.data = data;
    evalChart.update();
  }
}

async function loadAnalytics() {
  try {
    const res = await fetch('/analytics/summary');
    if (!res.ok) return;
    const d = await res.json();
    const el = document.getElementById('analytics');
    el.textContent = `Recent matches: ${d.total} | WR=${(d.win_rate*100).toFixed(1)}% | W/L/D=${d.wins}/${d.losses}/${d.draws}`;
  } catch {}
}

window.addEventListener('DOMContentLoaded', () => {
  initBoard();
  document.getElementById('new').addEventListener('click', newGame);
  document.getElementById('engineMove').addEventListener('click', engineMove);
  document.getElementById('evalBtn').addEventListener('click', evalPos);
  loadAnalytics();
});
