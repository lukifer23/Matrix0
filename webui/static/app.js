let gameId = null;
let board = null;
let chess = new Chess();
let evalData = { ply: [], matrix0: [], stockfish: [] };
let evalChart = null;
let trainingChart = null;
let currentView = 'game';
let tournamentPollInterval = null;

// Debug logging
function debugLog(message, data = null) {
  console.log(`[WebUI Debug] ${message}`, data || '');
}

// View switching
function switchView(viewName) {
  debugLog(`Switching to view: ${viewName}`);

  // Update buttons
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  document.getElementById(`show${viewName.charAt(0).toUpperCase() + viewName.slice(1)}`).classList.add('active');

  // Update content
  document.querySelectorAll('.view-content').forEach(content => {
    content.classList.remove('active');
  });
  document.getElementById(`${viewName}View`).classList.add('active');

  currentView = viewName;

  // Load view-specific data
  if (viewName === 'training') {
    debugLog('Loading training status');
    loadTrainingStatus();
  } else if (viewName === 'ssl') {
    debugLog('Loading SSL status');
    loadSSLStatus();
  } else if (viewName === 'tournament') {
    debugLog('Loading tournament data');
    loadTournamentData();
  } else if (viewName === 'analysis') {
    debugLog('Loading model analysis');
    loadModelAnalysis();
  }

  // Start/stop tournament polling
  if (viewName === 'tournament') {
    startTournamentPolling();
  } else {
    stopTournamentPolling();
  }
}

// Enhanced training monitoring with statistics
async function loadTrainingStatus() {
  debugLog('Fetching training status from /training/status');
  try {
    const response = await fetch('/training/status');
    debugLog('Training status response received', response.status);
    const data = await response.json();

    const statusEl = document.getElementById('trainingStatus');
    const stepEl = document.getElementById('trainStep');
    const progressEl = document.getElementById('trainProgress');
    const lossEl = document.getElementById('trainLoss');
    const policyLossEl = document.getElementById('trainPolicyLoss');
    const valueLossEl = document.getElementById('trainValueLoss');
    const sslLossEl = document.getElementById('trainSSLLoss');
    const lrEl = document.getElementById('trainLR');
    const historyEl = document.getElementById('trainingHistory');

    if (data.is_training) {
      statusEl.textContent = 'Active';
      statusEl.className = 'status-indicator training';

      stepEl.textContent = `${data.current_step}/${data.total_steps}`;
      progressEl.textContent = `${data.progress.toFixed(1)}%`;
      addProgressBar(data.progress);

      const metrics = data.latest_metrics;
      lossEl.textContent = metrics.loss.toFixed(4);
      policyLossEl.textContent = metrics.policy_loss.toFixed(4);
      valueLossEl.textContent = metrics.value_loss.toFixed(4);
      sslLossEl.textContent = metrics.ssl_loss.toFixed(4);
      lrEl.textContent = metrics.learning_rate.toFixed(6);

      // Display enhanced recent history with statistics
      const history = data.recent_history;
      const stats = data.statistics;

      let historyText = `Training Statistics:\n`;
      historyText += `Average Loss: ${stats.avg_loss.toFixed(4)}\n`;
      historyText += `Average SSL Loss: ${stats.avg_ssl_loss.toFixed(4)}\n`;
      historyText += `Loss Trend: ${stats.loss_trend}\n`;
      historyText += `SSL Trend: ${stats.ssl_trend}\n`;
      historyText += `LR Range: [${stats.lr_range[0].toFixed(6)}, ${stats.lr_range[1].toFixed(6)}]\n\n`;

      historyText += `Recent Steps:\n`;
      historyText += history.map(h =>
        `Step ${h.step}/${h.total_steps}: Loss=${h.loss.toFixed(4)}, Policy=${h.policy_loss.toFixed(4)}, Value=${h.value_loss.toFixed(4)}, SSL=${h.ssl_loss.toFixed(4)}, LR=${h.learning_rate.toFixed(6)}`
      ).join('\n');

      historyEl.textContent = historyText;

      // Update training chart with enhanced data
      updateTrainingChart(history, stats);
    } else {
      statusEl.textContent = 'Not Training';
      statusEl.className = 'status-indicator not-training';
      stepEl.textContent = '—';
      progressEl.textContent = '—';
      lossEl.textContent = '—';
      policyLossEl.textContent = '—';
      valueLossEl.textContent = '—';
      sslLossEl.textContent = '—';
      lrEl.textContent = '—';
      historyEl.textContent = data.message || 'No training data available';
    }
  } catch (error) {
    log('Failed to load training status: ' + error.message);
  }
}

// Enhanced SSL status monitoring with performance metrics
async function loadSSLStatus() {
  try {
    const response = await fetch('/ssl/status');
    const data = await response.json();

    const statusEl = document.getElementById('sslStatus');
    const enabledEl = document.getElementById('sslEnabled');
    const tasksEl = document.getElementById('sslTasks');
    const headCountEl = document.getElementById('sslHeadCount');
    const weightEl = document.getElementById('sslWeight');
    const paramsEl = document.getElementById('sslParams');
    const headsEl = document.getElementById('sslHeads');

    if (data.enabled) {
      statusEl.textContent = 'Enabled';
      statusEl.className = 'status-indicator ssl-enabled';
    } else {
      statusEl.textContent = 'Disabled';
      statusEl.className = 'status-indicator ssl-disabled';
    }

    enabledEl.textContent = data.enabled ? 'Yes' : 'No';
    tasksEl.textContent = data.tasks.join(', ') || 'None';
    headCountEl.textContent = data.ssl_head_count;
    weightEl.textContent = data.config.ssl_weight;
    paramsEl.textContent = data.total_ssl_params ? data.total_ssl_params.toLocaleString() : '—';

    // Display enhanced SSL heads with weights and analysis
    if (data.head_analysis && data.task_weights) {
      const headsHtml = Object.entries(data.head_analysis)
        .map(([name, analysis]) => {
          const weight = data.task_weights[name] || 1.0;
          return `
            <div class="ssl-head">
              <div class="ssl-head-name">${name} (weight: ${weight})</div>
              <div class="ssl-head-params">${analysis.parameters.toLocaleString()} parameters</div>
              <div class="ssl-head-structure">${analysis.structure}</div>
            </div>
          `;
        }).join('');
      headsEl.innerHTML = headsHtml;
    } else if (data.head_parameters) {
      const headsHtml = Object.entries(data.head_parameters)
        .map(([name, params]) => `
          <div class="ssl-head">
            <div class="ssl-head-name">${name}</div>
            <div class="ssl-head-params">${params.toLocaleString()} parameters</div>
          </div>
        `).join('');
      headsEl.innerHTML = headsHtml;
    } else {
      headsEl.textContent = 'SSL heads not available';
    }

    // Load SSL performance metrics
    await loadSSLPerformance();

  } catch (error) {
    log('Failed to load SSL status: ' + error.message);
  }
}

// SSL performance monitoring
async function loadSSLPerformance() {
  try {
    const response = await fetch('/ssl/performance');
    const data = await response.json();

    // Update SSL performance visualization if we have data
    if (data.statistics) {
      const perfEl = document.getElementById('sslPerformance');
      if (perfEl) {
        perfEl.innerHTML = `
          <div class="ssl-metric">
            <strong>Current SSL Loss:</strong> ${data.statistics.current_ssl_loss.toFixed(4)}
          </div>
          <div class="ssl-metric">
            <strong>Average SSL Loss:</strong> ${data.statistics.avg_ssl_loss.toFixed(4)}
          </div>
          <div class="ssl-metric">
            <strong>SSL Loss Trend:</strong> ${data.statistics.ssl_loss_trend}
          </div>
        `;
      }
    }
  } catch (error) {
    log('Failed to load SSL performance: ' + error.message);
  }
}

// Model analysis
async function loadModelAnalysis() {
  try {
    const response = await fetch('/model/analysis');
    const data = await response.json();

    const totalParamsEl = document.getElementById('totalParams');
    const channelsEl = document.getElementById('modelChannels');
    const blocksEl = document.getElementById('modelBlocks');
    const attnHeadsEl = document.getElementById('modelAttnHeads');
    const sslEnabledEl = document.getElementById('modelSSLEnabled');
    const breakdownEl = document.getElementById('paramBreakdown');

    totalParamsEl.textContent = data.total_parameters.toLocaleString();
    channelsEl.textContent = data.architecture.channels;
    blocksEl.textContent = data.architecture.blocks;
    attnHeadsEl.textContent = data.architecture.attention_heads;
    sslEnabledEl.textContent = data.architecture.ssl_enabled ? 'Yes' : 'No';

    // Parameter breakdown
    if (data.parameter_breakdown) {
      const breakdownHtml = Object.entries(data.parameter_breakdown)
        .sort(([,a], [,b]) => b - a) // Sort by parameter count
        .map(([type, count]) => `
          <div class="param-item">
            <span class="param-type">${type}</span>
            <span class="param-count">${count.toLocaleString()}</span>
          </div>
        `).join('');
      breakdownEl.innerHTML = breakdownHtml;
    }

  } catch (error) {
    log('Failed to load model analysis: ' + error.message);
  }
}

// Enhanced training chart with multiple metrics
function updateTrainingChart(history, statistics) {
  const ctx = document.getElementById('trainingChart').getContext('2d');

  // Create datasets for all metrics
  const datasets = [
    {
      label: 'Total Loss',
      data: history.map(h => ({ x: h.step, y: h.loss })),
      borderColor: '#ef4444',
      backgroundColor: 'rgba(239, 68, 68, 0.15)',
      tension: 0.2,
      pointRadius: 2,
      pointHoverRadius: 4,
    },
    {
      label: 'Policy Loss',
      data: history.map(h => ({ x: h.step, y: h.policy_loss })),
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245, 158, 11, 0.15)',
      tension: 0.2,
      pointRadius: 2,
      pointHoverRadius: 4,
    },
    {
      label: 'Value Loss',
      data: history.map(h => ({ x: h.step, y: h.value_loss })),
      borderColor: '#8b5cf6',
      backgroundColor: 'rgba(139, 92, 246, 0.15)',
      tension: 0.2,
      pointRadius: 2,
      pointHoverRadius: 4,
    },
    {
      label: 'SSL Loss',
      data: history.map(h => ({ x: h.step, y: h.ssl_loss })),
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.15)',
      tension: 0.2,
      pointRadius: 2,
      pointHoverRadius: 4,
    },
    {
      label: 'Learning Rate',
      data: history.map(h => ({ x: h.step, y: h.learning_rate })),
      borderColor: '#10b981',
      backgroundColor: 'rgba(16, 185, 129, 0.15)',
      yAxisID: 'y1',
      tension: 0.2,
      pointRadius: 2,
      pointHoverRadius: 4,
    }
  ];

  const data = { datasets };

  const options = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        labels: {
          color: getComputedStyle(document.documentElement).getPropertyValue('--fg') || '#fff',
          usePointStyle: true,
          pointStyle: 'line'
        }
      },
      tooltip: {
        callbacks: {
          title: function(context) {
            return `Step ${context[0].parsed.x}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        ticks: {
          color: getComputedStyle(document.documentElement).getPropertyValue('--muted'),
          callback: function(value) { return value; }
        },
        title: {
          display: true,
          text: 'Training Step',
          color: getComputedStyle(document.documentElement).getPropertyValue('--muted')
        }
      },
      y: {
        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted') },
        title: {
          display: true,
          text: 'Loss',
          color: getComputedStyle(document.documentElement).getPropertyValue('--muted')
        }
      },
      y1: {
        position: 'right',
        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted') },
        title: {
          display: true,
          text: 'Learning Rate',
          color: getComputedStyle(document.documentElement).getPropertyValue('--muted')
        }
      }
    }
  };

  if (!trainingChart) {
    trainingChart = new Chart(ctx, { type: 'line', data, options });
  } else {
    trainingChart.data = data;
    trainingChart.options = options;
    trainingChart.update();
  }
}

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
  debugLog(`API call: ${path}`, body);
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  debugLog(`API response: ${path} -> ${res.status}`);
  if (!res.ok) {
    const errorText = await res.text();
    debugLog(`API error: ${path} -> ${res.status}: ${errorText}`);
    throw new Error(errorText);
  }
  const result = await res.json();
  debugLog(`API success: ${path}`, result);
  return result;
}

async function newGame() {
  debugLog('New game button clicked');
  const white = document.getElementById('white').value;
  const black = document.getElementById('black').value;
  const tc = parseInt(document.getElementById('tc').value || '100', 10);
  debugLog(`Starting new game: ${white} vs ${black}, tc=${tc}`);
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

// Periodic updates for real-time monitoring
let updateIntervals = {};

function startPeriodicUpdates() {
  // Update training status every 30 seconds
  if (updateIntervals.training) clearInterval(updateIntervals.training);
  updateIntervals.training = setInterval(() => {
    if (currentView === 'training') {
      loadTrainingStatus();
    }
  }, 30000);

  // Update SSL status every 60 seconds
  if (updateIntervals.ssl) clearInterval(updateIntervals.ssl);
  updateIntervals.ssl = setInterval(() => {
    if (currentView === 'ssl') {
      loadSSLStatus();
    }
  }, 60000);
}

function stopPeriodicUpdates() {
  Object.values(updateIntervals).forEach(interval => clearInterval(interval));
  updateIntervals = {};
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

// Add progress bar to training view
function addProgressBar(progressPercent) {
  const progressEl = document.getElementById('trainProgress');
  const existingBar = progressEl.querySelector('.progress-bar');
  if (existingBar) existingBar.remove();

  const progressBar = document.createElement('div');
  progressBar.className = 'progress-bar';
  progressBar.innerHTML = `<div class="progress-fill" style="width: ${progressPercent}%"></div>`;
  progressEl.appendChild(progressBar);
}

window.addEventListener('DOMContentLoaded', () => {
  initBoard();

  // Game controls
  document.getElementById('new').addEventListener('click', newGame);
  document.getElementById('engineMove').addEventListener('click', engineMove);
  document.getElementById('evalBtn').addEventListener('click', evalPos);

  // View switching controls
  document.getElementById('showGame').addEventListener('click', () => switchView('game'));
  document.getElementById('showTraining').addEventListener('click', () => switchView('training'));
  document.getElementById('showSSL').addEventListener('click', () => switchView('ssl'));
  document.getElementById('showTournament').addEventListener('click', () => switchView('tournament'));
  document.getElementById('showAnalysis').addEventListener('click', () => switchView('analysis'));

  // Tournament controls
  document.getElementById('createTournament').addEventListener('click', createTournament);

  // Load initial data
  loadAnalytics();

  // Start periodic updates
  startPeriodicUpdates();

  // Load initial view data
  loadTrainingStatus();
  loadSSLStatus();
  loadTournamentData();
  loadModelAnalysis();
});

// Tournament Management Functions

async function loadTournamentData() {
  await Promise.all([
    loadActiveTournaments(),
    loadEngineRatings(),
    loadTournamentHistory()
  ]);
}

async function createTournament() {
  try {
    const name = document.getElementById('tournamentName').value.trim();
    if (!name) {
      alert('Please enter a tournament name');
      return;
    }

    // Get selected engines
    const selectedEngines = Array.from(document.querySelectorAll('#engineSelection input:checked'))
      .map(cb => cb.value);

    if (selectedEngines.length < 2) {
      alert('Please select at least 2 engines');
      return;
    }

    const config = {
      name: name,
      engines: selectedEngines,
      format: document.getElementById('tournamentFormat').value,
      num_games_per_pairing: parseInt(document.getElementById('gamesPerPairing').value),
      time_control_ms: parseInt(document.getElementById('timeControl').value),
      max_concurrency: parseInt(document.getElementById('maxConcurrency').value)
    };

    const response = await fetch('/tournament/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });

    if (response.ok) {
      const result = await response.json();
      log(`Tournament created: ${result.tournament_id}`);
      document.getElementById('tournamentName').value = '';
      await loadTournamentData();
    } else {
      const error = await response.json();
      alert(`Failed to create tournament: ${error.detail}`);
    }
  } catch (error) {
    log(`Tournament creation failed: ${error.message}`);
  }
}

async function loadActiveTournaments() {
  try {
    const response = await fetch('/tournament/list');
    const data = await response.json();

    const container = document.getElementById('activeTournaments');

    if (data.active_tournaments.length === 0) {
      container.innerHTML = '<p>No active tournaments</p>';
      return;
    }

    const html = data.active_tournaments.map(tournament => `
      <div class="tournament-item">
        <h4>${tournament.name}</h4>
        <div class="tournament-stats">
          <div class="tournament-stat">
            <div class="tournament-stat-label">Status</div>
            <div class="tournament-stat-value">${tournament.status}</div>
          </div>
          <div class="tournament-stat">
            <div class="tournament-stat-label">Format</div>
            <div class="tournament-stat-value">${tournament.format.replace('_', ' ')}</div>
          </div>
          <div class="tournament-stat">
            <div class="tournament-stat-label">Engines</div>
            <div class="tournament-stat-value">${tournament.engines.length}</div>
          </div>
          <div class="tournament-stat">
            <div class="tournament-stat-label">Duration</div>
            <div class="tournament-stat-value">${Math.round(tournament.duration)}s</div>
          </div>
        </div>
      </div>
    `).join('');

    container.innerHTML = html;
  } catch (error) {
    log(`Failed to load active tournaments: ${error.message}`);
  }
}

async function loadEngineRatings() {
  try {
    const response = await fetch('/ratings');
    const data = await response.json();

    const container = document.getElementById('engineRatings');

    if (!data.ratings || Object.keys(data.ratings).length === 0) {
      container.innerHTML = '<p>No ratings available</p>';
      return;
    }

    const html = Object.entries(data.ratings).map(([engine, ratings]) => `
      <div class="rating-item">
        <div class="rating-engine">${engine}</div>
        <div class="rating-values">
          <div class="rating-elo">ELO: ${ratings.elo}</div>
          <div class="rating-glicko">Glicko: ${ratings.glicko_rating.toFixed(0)}</div>
        </div>
      </div>
    `).join('');

    container.innerHTML = html;
  } catch (error) {
    log(`Failed to load engine ratings: ${error.message}`);
  }
}

async function loadTournamentHistory() {
  try {
    const response = await fetch('/tournament/list');
    const data = await response.json();

    const container = document.getElementById('tournamentHistory');

    if (data.completed_tournaments.length === 0) {
      container.innerHTML = '<p>No completed tournaments</p>';
      return;
    }

    const html = data.completed_tournaments.slice(0, 5).map(tournament => `
      <div class="tournament-item">
        <h4>${tournament.name}</h4>
        <div class="tournament-stats">
          <div class="tournament-stat">
            <div class="tournament-stat-label">Format</div>
            <div class="tournament-stat-value">${tournament.format.replace('_', ' ')}</div>
          </div>
          <div class="tournament-stat">
            <div class="tournament-stat-label">Games</div>
            <div class="tournament-stat-value">${tournament.total_games}</div>
          </div>
          <div class="tournament-stat">
            <div class="tournament-stat-label">Duration</div>
            <div class="tournament-stat-value">${Math.round(tournament.duration)}s</div>
          </div>
        </div>
      </div>
    `).join('');

    container.innerHTML = html;
  } catch (error) {
    log(`Failed to load tournament history: ${error.message}`);
  }
}

function startTournamentPolling() {
  if (tournamentPollInterval) {
    clearInterval(tournamentPollInterval);
  }
  tournamentPollInterval = setInterval(() => {
    if (currentView === 'tournament') {
      loadActiveTournaments();
    }
  }, 5000); // Poll every 5 seconds
}

function stopTournamentPolling() {
  if (tournamentPollInterval) {
    clearInterval(tournamentPollInterval);
    tournamentPollInterval = null;
  }
}
