let gameId = null;
let board = null;
let chess = new Chess();
let evalData = { ply: [], matrix0: [], stockfish: [] };
let evalChart = null;

function log(msg) {
  const el = document.getElementById('log');
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + el.textContent;
}

function updateState(fen, turn, finished, result) {
  chess.load(fen);
  document.getElementById('fen').textContent = fen;
  document.getElementById('turn').textContent = turn;
  if (finished) log(`Game over: ${result}`);
  board.set({ fen, turnColor: (turn === 'w' ? 'white' : 'black') });
  renderMoves();
  renderChart();
}

function renderMoves() {
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

async function newGame() {
  const white = document.getElementById('white').value;
  const black = document.getElementById('black').value;
  const tc = parseInt(document.getElementById('tc').value || '100', 10);
  const out = await api('/new', { white, black, engine_tc_ms: tc });
  gameId = out.game_id;
  document.getElementById('gid').textContent = gameId;
  document.getElementById('tclabel').textContent = tc;
  log(`New game ${gameId}: ${white} vs ${black}`);
  evalData = { ply: [], matrix0: [], stockfish: [] };
  updateState(out.fen, out.turn, false, null);
  loadAnalytics();
}

async function pushMove(uci) {
  const out = await api('/move', { game_id: gameId, uci });
  updateState(out.fen, out.turn, out.game_over, out.result);
}

async function engineMove() {
  if (!gameId) return;
  const turn = chess.turn();
  const color = (turn === 'w' ? 'white' : 'black');
  const engine = document.getElementById(color).value; // who plays this color
  const out = await api('/engine-move', { game_id: gameId, engine });
  log(`${engine} plays ${out.uci}`);
  updateState(out.fen, out.turn, out.game_over, out.result);
  // collect eval snapshot
  try { await evalPos(); } catch {}
}

async function evalPos() {
  if (!gameId) return;
  const out = await api('/eval', { game_id: gameId, include_stockfish: true });
  document.getElementById('mval').textContent = (out.matrix0_value === null ? '—' : out.matrix0_value.toFixed(3));
  document.getElementById('scpf').textContent = (out.stockfish_cp === null ? '—' : out.stockfish_cp);
  // push evals for chart
  const ply = chess.history().length;
  evalData.ply.push(ply);
  evalData.matrix0.push(out.matrix0_value);
  evalData.stockfish.push(out.stockfish_cp !== null ? out.stockfish_cp / 100.0 : null);
  renderChart();
}

function initBoard() {
  board = Chessground(document.getElementById('board'), {
    fen: chess.fen(),
    orientation: 'white',
    movable: {
      free: false,
      color: () => 'both',
      events: {
        after: async (orig, dest, meta) => {
          if (!gameId) return;
          // promotion
          let promo = '';
          const move = { from: orig, to: dest, promotion: (meta.promotion || 'q') };
          const uci = move.from + move.to + (move.promotion ? move.promotion : '');
          try {
            await pushMove(uci);
          } catch (e) {
            log('Illegal move');
            board.set({ fen: chess.fen() });
          }
          try { await evalPos(); } catch {}
        }
      }
    }
  });
}

function renderChart() {
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
