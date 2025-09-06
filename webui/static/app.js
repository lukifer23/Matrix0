let gameId = null;
let board = null;
let chess = null;
let boardInitTries = 0;
let moveHistory = [];
let evalData = { ply: [], matrix0: [], stockfish: [] };
let evalChart = null;
let trainingChart = null;
let currentView = 'game';
let tournamentPollInterval = null;
let orchPollInterval = null;
let orchEventSource = null;
let connectionStatus = 'unknown'; // 'connected', 'disconnected', 'unknown'

// Debug logging
function debugLog(message, data = null) {
  console.log(`[WebUI Debug] ${message}`, data || '');
}

// Connection status monitoring
function updateConnectionStatus(status) {
  connectionStatus = status;
  const statusIndicator = document.getElementById('connectionStatus');
  if (statusIndicator) {
    statusIndicator.className = `status-indicator ${status === 'connected' ? 'ssl-enabled' : 'ssl-disabled'}`;
    statusIndicator.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
  }

  if (status === 'disconnected') {
    showWarning('Lost connection to server. Some features may not work.', { duration: 0, dismissible: false });
  }
}

async function checkServerConnection() {
  try {
    const response = await fetch('/health', { timeout: 5000 });
    const wasDisconnected = connectionStatus === 'disconnected';
    updateConnectionStatus('connected');

    if (wasDisconnected) {
      showSuccess('Reconnected to server');
    }

    return true;
  } catch (error) {
    updateConnectionStatus('disconnected');
    return false;
  }
}

// Global error handler
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
  if (event.error && event.error.message) {
    showError(`Application error: ${event.error.message}`, { duration: 10000 });
  } else if (event.error) {
    showError(`Application error: ${String(event.error)}`, { duration: 10000 });
  }
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  if (event.reason && event.reason.message) {
    showError(`Unhandled error: ${event.reason.message}`, { duration: 10000 });
  } else if (event.reason) {
    showError(`Unhandled error: ${String(event.reason)}`, { duration: 10000 });
  }
});

// View switching
function switchView(viewName) {
  debugLog(`Switching to view: ${viewName}`);

  // Update buttons
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const btn = document.getElementById(`show${viewName.charAt(0).toUpperCase() + viewName.slice(1)}`);
  if (btn) btn.classList.add('active');

  // Update content
  document.querySelectorAll('.view-content').forEach(content => {
    content.classList.remove('active');
  });
  const container = document.getElementById(`${viewName}View`);
  if (container) container.classList.add('active');

  currentView = viewName;

  // Load view-specific data
  if (viewName === 'game') {
    debugLog('Loading game view');
    // Ensure board is initialized
    if (typeof Chess !== 'undefined' && !chess) {
      chess = new Chess();
    }
    if (chess) {
      const boardElement = document.getElementById('board');
      if (boardElement && !boardElement.querySelector('.square')) {
        boardElement.classList.remove('loading');
        createChessBoard(boardElement);
      }
      updateBoardDisplay();
    }
  } else if (viewName === 'training') {
    debugLog('Loading training status');
    loadTrainingStatus();
  } else if (viewName === 'orchestrator') {
    debugLog('Loading orchestrator status');
    loadOrchestratorStatus();
    loadOrchestratorLogs();
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
  if (viewName === 'orchestrator') {
    startOrchestratorPolling();
    startOrchestratorSSE();
  } else {
    stopOrchestratorPolling();
    stopOrchestratorSSE();
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
    const sslWeightEl = document.getElementById('trainSSLWeight');
    const historyEl = document.getElementById('trainingHistory');
    const controlStatusEl = document.getElementById('trainingControlStatus');
    const lastUpdateEl = document.getElementById('trainingLastUpdate');

    if (data.is_training) {
      statusEl.textContent = 'Active';
      statusEl.className = 'status-indicator training';
      controlStatusEl.textContent = 'Training Active';
      controlStatusEl.className = 'status-indicator training';
      lastUpdateEl.textContent = new Date().toLocaleTimeString();

      stepEl.textContent = `${data.current_step}/${data.total_steps}`;
      progressEl.textContent = `${data.progress.toFixed(1)}%`;
      addProgressBar(data.progress);

      const metrics = data.latest_metrics;
      lossEl.textContent = metrics.loss.toFixed(4);
      policyLossEl.textContent = metrics.policy_loss.toFixed(4);
      valueLossEl.textContent = metrics.value_loss.toFixed(4);
      sslLossEl.textContent = metrics.ssl_loss.toFixed(4);
      lrEl.textContent = metrics.learning_rate.toFixed(6);
      sslWeightEl.textContent = '0.15'; // This could be fetched from config if available

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
      controlStatusEl.textContent = 'No Active Training';
      controlStatusEl.className = 'status-indicator not-training';
      lastUpdateEl.textContent = new Date().toLocaleTimeString();
      stepEl.textContent = '—';
      progressEl.textContent = '—';
      lossEl.textContent = '—';
      policyLossEl.textContent = '—';
      valueLossEl.textContent = '—';
      sslLossEl.textContent = '—';
      lrEl.textContent = '—';
      sslWeightEl.textContent = '—';
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

// System metrics dashboard
async function loadSystemMetrics() {
  try {
    const startTime = Date.now();

    // Load various system metrics in parallel
    const [healthRes, trainingRes, sslRes, tournamentRes] = await Promise.allSettled([
      fetch('/health'),
      fetch('/training/status'),
      fetch('/ssl/status'),
      fetch('/tournament/list')
    ]);

    const responseTime = Date.now() - startTime;

    // Update metrics
    const activeGamesEl = document.getElementById('metricActiveGames');
    const trainingStatusEl = document.getElementById('metricTrainingStatus');
    const sslTasksEl = document.getElementById('metricSSLTasks');
    const tournamentsEl = document.getElementById('metricTournaments');
    const memoryEl = document.getElementById('metricMemoryUsage');
    const responseTimeEl = document.getElementById('metricResponseTime');

    // Active games (simplified - could be enhanced with actual game count)
    activeGamesEl.textContent = gameId ? '1' : '0';

    // Training status
    if (trainingRes.status === 'fulfilled') {
      const training = await trainingRes.value.json();
      trainingStatusEl.textContent = training.is_training ? 'Active' : 'Idle';
      trainingStatusEl.style.color = training.is_training ? 'var(--success)' : 'var(--muted)';
    }

    // SSL tasks
    if (sslRes.status === 'fulfilled') {
      const ssl = await sslRes.value.json();
      sslTasksEl.textContent = ssl.tasks ? ssl.tasks.length : '0';
    }

    // Tournaments
    if (tournamentRes.status === 'fulfilled') {
      const tournaments = await tournamentRes.value.json();
      const activeCount = tournaments.active_tournaments?.length || 0;
      tournamentsEl.textContent = activeCount;
    }

    // Memory usage (placeholder - would need backend support)
    memoryEl.textContent = '~11GB';

    // Response time
    responseTimeEl.textContent = `${responseTime}ms`;
    responseTimeEl.style.color = responseTime < 500 ? 'var(--success)' : responseTime < 1000 ? 'var(--warning)' : 'var(--error)';

  } catch (error) {
    debugLog('Failed to load system metrics', error);
  }
}

// Model analysis
async function loadModelAnalysis() {
  try {
    // Load system metrics first
    await loadSystemMetrics();

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
  try {
    if (chess) {
      chess.load(fen);
      debugLog(`Board state updated: ${fen}`);
    }
  } catch (e) {
    debugLog(`Failed to load FEN: ${fen}`, e);
  }

  currentFen = fen;
  currentTurn = turn;

  document.getElementById('fen').textContent = fen;
  document.getElementById('turn').textContent = turn;

  if (finished) {
    log(`Game over: ${result}`);
    showSuccess(`Game finished: ${result}`);
  }

  try {
    // Update chess instance with new FEN
    if (chess) {
      chess.load(fen);
      updateBoardDisplay();
      debugLog('Board UI updated successfully');
    } else {
      debugLog('Chess instance not available for board update');
    }
  } catch (e) {
    debugLog('Board update failed', e);
  }

  renderMoves();
  renderChart();
}

function getLegalMoves() {
  if (!chess) return {};

  const dests = {};
  chess.moves({ verbose: true }).forEach(move => {
    if (!dests[move.from]) dests[move.from] = [];
    dests[move.from].push(move.to);
  });

  return dests;
}

function isInCheck() {
  return chess ? chess.inCheck() : false;
}

function renderMoves() {
  const el = document.getElementById('movelist');
  let out = [];
  if (chess) {
    const hist = chess.history({ verbose: true });
    for (let i = 0; i < hist.length; i++) {
      const m = hist[i]; const num = Math.floor(i / 2) + 1;
      if (i % 2 === 0) out.push(`${num}. ${m.san}`); else out[out.length - 1] += `  ${m.san}`;
    }
  } else {
    for (let i = 0; i < moveHistory.length; i++) {
      const num = Math.floor(i / 2) + 1;
      if (i % 2 === 0) out.push(`${num}. ${moveHistory[i]}`); else out[out.length - 1] += `  ${moveHistory[i]}`;
    }
  }
  el.textContent = out.join('\n');
}

async function api(path, body, options = {}) {
  const { showLoading = false, loadingElement = null } = options;

  debugLog(`API call: ${path}`, body);

  if (showLoading && loadingElement) {
    loadingElement.classList.add('loading');
  }

  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    debugLog(`API response: ${path} -> ${res.status}`);

    if (!res.ok) {
      let errorText;
      try {
        const errorData = await res.json();
        errorText = errorData.detail || errorData.message || `HTTP ${res.status}`;
      } catch {
        try {
          errorText = await res.text() || `HTTP ${res.status}`;
        } catch {
          errorText = `HTTP ${res.status}`;
        }
      }
      debugLog(`API error: ${path} -> ${res.status}: ${errorText}`);
      throw new Error(errorText);
    }

    const result = await res.json();
    debugLog(`API success: ${path}`, result);
    return result;

  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error: Unable to connect to server');
    }
    throw error;
  } finally {
    if (showLoading && loadingElement) {
      loadingElement.classList.remove('loading');
    }
  }
}

async function newGame() {
  debugLog('New game button clicked');

  const white = document.getElementById('white').value;
  const black = document.getElementById('black').value;
  const tc = parseInt(document.getElementById('tc').value || '100', 10);

  try {
    debugLog(`Starting new game: ${white} vs ${black}, tc=${tc}`);
    const out = await api('/new', { white, black, engine_tc_ms: tc });

    gameId = out.game_id;
    document.getElementById('gid').textContent = gameId;
    document.getElementById('tclabel').textContent = tc;

    log(`New game ${gameId}: ${white} vs ${black}`);
    showSuccess(`New game started: ${white} vs ${black}`);

    // Hide game status indicator for regular games
    const statusEl = document.getElementById('gameStatus');
    statusEl.style.display = 'none';

    evalData = { ply: [], matrix0: [], stockfish: [] };

    // Initialize or reinitialize board
    if (!board || !chess) {
      initBoard();
    }

    updateState(out.fen, out.turn, false, null);
    loadAnalytics();

    // Auto-make engine move if it's engine's turn
    const currentPlayer = out.turn === 'w' ? white : black;
    if (currentPlayer !== 'human') {
      setTimeout(() => engineMove(), 500);
    }

  } catch (error) {
    debugLog('Failed to start new game', error);
    showError('Failed to start new game: ' + error.message);
  }
}

async function startModelVsModel() {
  debugLog('Model vs Model game started');

  const tc = parseInt(document.getElementById('tc').value || '100', 10);

  try {
    debugLog(`Starting Matrix0 vs Matrix0 game, tc=${tc}`);
    const out = await api('/new', { white: 'matrix0', black: 'matrix0', engine_tc_ms: tc });

    gameId = out.game_id;
    document.getElementById('gid').textContent = gameId;
    document.getElementById('tclabel').textContent = tc;

    // Update player selects to show Matrix0 vs Matrix0
    document.getElementById('white').value = 'matrix0';
    document.getElementById('black').value = 'matrix0';

    log(`Model vs Model game ${gameId}: Matrix0 vs Matrix0`);
    showSuccess('Model vs Model game started - AI vs AI battle!');

    // Show game status
    const statusEl = document.getElementById('gameStatus');
    statusEl.textContent = 'AI vs AI';
    statusEl.style.display = 'inline-flex';
    statusEl.className = 'badge status-indicator training';

    evalData = { ply: [], matrix0: [], stockfish: [] };

    // Initialize or reinitialize board
    if (!board || !chess) {
      initBoard();
    }

    updateState(out.fen, out.turn, false, null);
    loadAnalytics();

    // Start the AI battle - first move will be made automatically
    showSuccess('AI battle commencing...');
    setTimeout(() => engineMove(), 1000);

  } catch (error) {
    debugLog('Failed to start model vs model game', error);
    showError('Failed to start AI vs AI game: ' + error.message);
  }
}

async function startMatrixVsStockfish() {
  debugLog('Matrix0 vs Stockfish game started');

  const tc = parseInt(document.getElementById('tc').value || '100', 10);

  try {
    debugLog(`Starting Matrix0 vs Stockfish game, tc=${tc}`);
    const out = await api('/new', { white: 'matrix0', black: 'stockfish', engine_tc_ms: tc });

    gameId = out.game_id;
    document.getElementById('gid').textContent = gameId;
    document.getElementById('tclabel').textContent = tc;

    // Update player selects to show Matrix0 vs Stockfish
    document.getElementById('white').value = 'matrix0';
    document.getElementById('black').value = 'stockfish';

    log(`Matrix0 vs Stockfish game ${gameId}: Matrix0 vs Stockfish`);
    showSuccess('Matrix0 vs Stockfish challenge started!');

    // Show game status
    const statusEl = document.getElementById('gameStatus');
    statusEl.textContent = 'Matrix0 vs SF';
    statusEl.style.display = 'inline-flex';
    statusEl.className = 'badge status-indicator ssl-enabled';

    evalData = { ply: [], matrix0: [], stockfish: [] };

    // Initialize or reinitialize board
    if (!board || !chess) {
      initBoard();
    }

    updateState(out.fen, out.turn, false, null);
    loadAnalytics();

    // Start the challenge - Matrix0 moves first
    showSuccess('Challenge commencing...');
    setTimeout(() => engineMove(), 1000);

  } catch (error) {
    debugLog('Failed to start Matrix0 vs Stockfish game', error);
    showError('Failed to start challenge: ' + error.message);
  }
}

async function pushMove(uci) {
  const out = await api('/move', { game_id: gameId, uci });
  updateState(out.fen, out.turn, out.game_over, out.result);
}

async function engineMove() {
  if (!gameId) {
    showError('No active game');
    return;
  }

  if (!chess) {
    showError('Chess engine not initialized');
    return;
  }

  const turn = chess.turn();
  const color = (turn === 'w' ? 'white' : 'black');
  const engine = document.getElementById(color).value; // who plays this color

  // Show loading state
  const engineMoveBtn = document.getElementById('engineMove');
  if (engineMoveBtn) {
    engineMoveBtn.disabled = true;
    engineMoveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Thinking...';
  }

  try {
    debugLog(`Requesting engine move from ${engine} for ${color}`);
    const out = await api('/engine-move', { game_id: gameId, engine });

    const ms = (out.ms != null) ? ` (${out.ms.toFixed(1)} ms)` : '';
    log(`${engine} plays ${out.uci}${ms}`);
    showSuccess(`${engine} played ${out.uci}${ms}`);

    updateState(out.fen, out.turn, out.game_over, out.result);

    // Hide status indicator if game is over
    if (out.game_over) {
      const statusEl = document.getElementById('gameStatus');
      statusEl.style.display = 'none';
    }

    // collect eval snapshot
    try { await evalPos(); } catch {}

    // Auto-make next engine move if it's still engine's turn and game not over
    if (!out.game_over) {
      const nextTurn = chess.turn();
      const nextColor = (nextTurn === 'w' ? 'white' : 'black');
      const nextEngine = document.getElementById(nextColor).value;
      if (nextEngine !== 'human') {
        setTimeout(() => engineMove(), 1000); // 1 second delay for better UX
      }
    }

  } catch (error) {
    debugLog('Engine move failed', error);
    showError(`Engine move failed: ${error.message}`);
  } finally {
    // Reset button state
    if (engineMoveBtn) {
      engineMoveBtn.disabled = false;
      engineMoveBtn.innerHTML = '<i class="fas fa-play"></i> Engine Move';
    }
  }
}

async function evalPos() {
  if (!gameId) return;
  const out = await api('/eval', { game_id: gameId, include_stockfish: true });
  document.getElementById('mval').textContent = (out.matrix0_value === null ? '—' : out.matrix0_value.toFixed(3));
  document.getElementById('scpf').textContent = (out.stockfish_cp === null ? '—' : out.stockfish_cp);

  // Calculate evaluation difference
  if (out.matrix0_value !== null && out.stockfish_cp !== null) {
    const matrix0Pawn = out.matrix0_value;
    const stockfishPawn = out.stockfish_cp / 100.0;
    const diff = matrix0Pawn - stockfishPawn;
    document.getElementById('evalDiff').textContent = diff.toFixed(2) + ' pawns';
    document.getElementById('evalDiff').style.color = diff > 0.2 ? 'var(--success)' : diff < -0.2 ? 'var(--error)' : 'var(--muted)';
  } else {
    document.getElementById('evalDiff').textContent = '—';
  }

  // push evals for chart
  const ply = chess ? chess.history().length : moveHistory.length;
  evalData.ply.push(ply);
  evalData.matrix0.push(out.matrix0_value);
  evalData.stockfish.push(out.stockfish_cp !== null ? out.stockfish_cp / 100.0 : null);
  renderChart();
}

async function performDeepAnalysis() {
  if (!gameId) {
    showError('No active game to analyze');
    return;
  }

  try {
    showSuccess('Performing deep analysis...');
    // Perform multiple evaluations with different time controls for deeper analysis
    const analyses = [];
    for (let i = 0; i < 3; i++) {
      const out = await api('/eval', { game_id: gameId, include_stockfish: true });
      analyses.push(out);
      await new Promise(resolve => setTimeout(resolve, 500)); // Small delay between analyses
    }

    // Calculate average evaluations
    const avgMatrix0 = analyses.reduce((sum, a) => sum + (a.matrix0_value || 0), 0) / analyses.length;
    const avgStockfish = analyses.reduce((sum, a) => sum + ((a.stockfish_cp || 0) / 100.0), 0) / analyses.length;

    const diff = avgMatrix0 - avgStockfish;
    const analysis = `Deep Analysis Results:
Matrix0: ${avgMatrix0.toFixed(3)}
Stockfish: ${avgStockfish.toFixed(3)}
Difference: ${diff.toFixed(2)} pawns

${diff > 0.1 ? 'Matrix0 has advantage' : diff < -0.1 ? 'Stockfish has advantage' : 'Position is equal'}`;

    log(analysis);
    showSuccess('Deep analysis completed!');

  } catch (error) {
    debugLog('Deep analysis failed', error);
    showError('Deep analysis failed: ' + error.message);
  }
}

function clearEvaluation() {
  evalData = { ply: [], matrix0: [], stockfish: [] };
  document.getElementById('mval').textContent = '—';
  document.getElementById('scpf').textContent = '—';
  document.getElementById('evalDiff').textContent = '—';
  renderChart();
  showSuccess('Evaluation data cleared');
}

function createChessBoard(container) {
  debugLog('Creating chess board...');
  
  // Clear existing content
  container.innerHTML = '';

  // Create 8x8 board
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const square = document.createElement('div');
      square.className = 'square';
      square.dataset.square = String.fromCharCode(97 + col) + (8 - row);
      square.style.gridColumn = col + 1;
      square.style.gridRow = row + 1;

      // Add click handler for moves
      square.addEventListener('click', handleSquareClick);

      container.appendChild(square);
    }
  }
  
  debugLog(`Created ${container.children.length} squares`);

  // Add CSS for the board
  if (!document.getElementById('chess-board-styles')) {
    const style = document.createElement('style');
    style.id = 'chess-board-styles';
    style.textContent = `
      #board {
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        grid-template-rows: repeat(8, 1fr);
        gap: 0;
        border: 3px solid var(--border);
        background: var(--card);
        width: 100%;
        height: 100%;
        border-radius: 8px;
        overflow: hidden;
        min-height: 400px;
      }
      .square {
        aspect-ratio: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        cursor: pointer;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      .square:nth-child(16n+1),
      .square:nth-child(16n+3),
      .square:nth-child(16n+5),
      .square:nth-child(16n+7),
      .square:nth-child(16n+10),
      .square:nth-child(16n+12),
      .square:nth-child(16n+14),
      .square:nth-child(16n+16) {
        background: #f0d9b5;
      }
      .square:nth-child(16n+2),
      .square:nth-child(16n+4),
      .square:nth-child(16n+6),
      .square:nth-child(16n+8),
      .square:nth-child(16n+9),
      .square:nth-child(16n+11),
      .square:nth-child(16n+13),
      .square:nth-child(16n+15) {
        background: #b58863;
      }
      .square.highlight {
        box-shadow: inset 0 0 0 4px var(--accent);
        background: rgba(59, 130, 246, 0.3) !important;
      }
      .square.selected {
        box-shadow: inset 0 0 0 4px var(--success);
        background: rgba(16, 185, 129, 0.3) !important;
      }
      .square:hover {
        background: rgba(59, 130, 246, 0.2) !important;
      }
    `;
    document.head.appendChild(style);
  }

  updateBoardDisplay();
  
  // Force a reflow to ensure the board is visible
  container.offsetHeight;
  
  // Ensure board is properly sized
  setTimeout(() => {
    const boardWrap = document.getElementById('boardWrap');
    if (boardWrap) {
      const size = Math.min(boardWrap.offsetWidth - 40, boardWrap.offsetHeight - 40);
      container.style.width = `${size}px`;
      container.style.height = `${size}px`;
      debugLog(`Board resized to: ${size}x${size}`);
    }
  }, 100);
  
  debugLog('Chess board creation completed');
  debugLog(`Board container size: ${container.offsetWidth}x${container.offsetHeight}`);
  debugLog(`Squares created: ${container.children.length}`);
}

function updateBoardDisplay() {
  if (!chess) {
    debugLog('No chess instance available for board update');
    return;
  }

  const board = chess.board();
  const squares = document.querySelectorAll('.square');
  
  debugLog(`Updating board display: ${squares.length} squares found`);

  squares.forEach(square => {
    const squareNotation = square.dataset.square;
    const file = squareNotation.charCodeAt(0) - 97;
    const rank = 8 - parseInt(squareNotation[1]);

    const piece = board[rank][file];
    square.textContent = piece ? getPieceSymbol(piece) : '';
    square.classList.remove('highlight', 'selected');
  });
  
  debugLog('Board display updated');
}

function getPieceSymbol(piece) {
  const symbols = {
    'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚',
    'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔'
  };
  return symbols[piece.type] || '';
}

let selectedSquare = null;

function handleSquareClick(event) {
  const square = event.target.closest('.square');
  if (!square) return;

  const squareNotation = square.dataset.square;

  if (selectedSquare) {
    // Check if clicking on the same square (deselect)
    if (selectedSquare === squareNotation) {
      selectedSquare = null;
      document.querySelectorAll('.square').forEach(sq => sq.classList.remove('selected', 'highlight'));
      return;
    }

    // Try to make a move
    const move = chess.move({
      from: selectedSquare,
      to: squareNotation,
      promotion: 'q'
    });

    if (move) {
      // Move was successful
      moveHistory.push(move.san);
      updateBoardDisplay();
      pushMove(move.from + move.to + (move.promotion ? move.promotion : '')).then(() => {
        showSuccess('Move played successfully');
        updateState();
        evalPos();
      }).catch(e => {
        showError('Move failed: ' + e.message);
        chess.undo();
        moveHistory.pop();
        updateBoardDisplay();
      });
    } else {
      // Invalid move, just clear selection
      showError('Invalid move');
    }

    // Clear selection
    selectedSquare = null;
    document.querySelectorAll('.square').forEach(sq => sq.classList.remove('selected', 'highlight'));
  } else {
    // Select this square if it has a piece
    const file = squareNotation.charCodeAt(0) - 97;
    const rank = 8 - parseInt(squareNotation[1]);
    const piece = chess.board()[rank][file];

    if (piece) {
      selectedSquare = squareNotation;
      square.classList.add('selected');

      // Highlight possible moves
      const moves = chess.moves({ square: squareNotation, verbose: true });
      moves.forEach(move => {
        const targetSquare = document.querySelector(`[data-square="${move.to}"]`);
        if (targetSquare) {
          targetSquare.classList.add('highlight');
        }
      });
    }
  }
}

function initBoard() {
  debugLog(`Board init attempt ${boardInitTries + 1}: Chess=${typeof Chess}`);

  if (typeof Chess === 'undefined') {
    boardInitTries++;
    if (boardInitTries < 40) { // retry up to ~20s with more attempts
      debugLog(`Board init attempt ${boardInitTries}: Chess=${typeof Chess}`);
      setTimeout(initBoard, 500);
    } else {
      debugLog('Game board unavailable after retries; check network/CDN.');
      showError('Chess board libraries failed to load after multiple attempts. This may be due to network connectivity or CDN issues.');
    }
    return;
  }
  boardInitTries = 0;
  chess = new Chess();

  // Remove loading state
  const boardElement = document.getElementById('board');
  if (boardElement) {
    boardElement.classList.remove('loading');
    // Create the chess board HTML
    createChessBoard(boardElement);
  }

  debugLog('Chess board initialized successfully');
}

function getMovableColor() {
  if (!gameId) return null;
  const whitePlayer = document.getElementById('white').value;
  const blackPlayer = document.getElementById('black').value;
  const currentTurn = chess ? (chess.turn() === 'w' ? 'white' : 'black') : 'white';

  if (currentTurn === 'white' && whitePlayer === 'human') return 'white';
  if (currentTurn === 'black' && blackPlayer === 'human') return 'black';
  return null; // Engine's turn
}

function showError(message, options = {}) {
  const { duration = 5000, dismissible = true } = options;

  // Remove existing error
  const existing = document.querySelector('.error');
  if (existing) existing.remove();

  const errorDiv = document.createElement('div');
  errorDiv.className = 'error';
  errorDiv.innerHTML = `
    <i class="fas fa-exclamation-triangle"></i>
    <span>${message}</span>
    ${dismissible ? '<button class="error-dismiss" onclick="this.parentElement.remove()">&times;</button>' : ''}
  `;

  document.body.appendChild(errorDiv);

  // Auto-dismiss after duration
  if (duration > 0) {
    setTimeout(() => {
      if (errorDiv.parentNode) {
        errorDiv.remove();
      }
    }, duration);
  }

  // Log to console for debugging
  console.error('UI Error:', message);
}

function showWarning(message, options = {}) {
  const { duration = 4000, dismissible = true } = options;

  const existing = document.querySelector('.warning');
  if (existing) existing.remove();

  const warningDiv = document.createElement('div');
  warningDiv.className = 'warning';
  warningDiv.innerHTML = `
    <i class="fas fa-exclamation-circle"></i>
    <span>${message}</span>
    ${dismissible ? '<button class="warning-dismiss" onclick="this.parentElement.remove()">&times;</button>' : ''}
  `;

  document.body.appendChild(warningDiv);

  if (duration > 0) {
    setTimeout(() => {
      if (warningDiv.parentNode) {
        warningDiv.remove();
      }
    }, duration);
  }

  console.warn('UI Warning:', message);
}

function showSuccess(message) {
  const existing = document.querySelector('.success');
  if (existing) existing.remove();

  const successDiv = document.createElement('div');
  successDiv.className = 'success';
  successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
  document.body.appendChild(successDiv);

  setTimeout(() => {
    if (successDiv.parentNode) {
      successDiv.remove();
    }
  }, 3000);
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

function startOrchestratorPolling() {
  if (orchPollInterval) clearInterval(orchPollInterval);
  orchPollInterval = setInterval(() => {
    if (currentView === 'orchestrator') {
      loadOrchestratorStatus();
      loadOrchestratorLogs();
      loadOrchestratorWorkers();
    }
  }, 5000);
}

function stopOrchestratorPolling() {
  if (orchPollInterval) {
    clearInterval(orchPollInterval);
    orchPollInterval = null;
  }
}

function startOrchestratorSSE() {
  try {
    if (orchEventSource) orchEventSource.close();
    orchEventSource = new EventSource('/orchestrator/stream?kind=structured');
    orchEventSource.onmessage = (ev) => {
      const el = document.getElementById('orchLogs');
      if (!el) return;
      el.textContent += (ev.data + '\n');
      el.scrollTop = el.scrollHeight;

      // Try to parse structured data for metrics updates
      try {
        const data = JSON.parse(ev.data);
        if (data && typeof data === 'object') {
          // Update metrics if available
          if (data.games_played !== undefined) {
            const gamesEl = document.getElementById('metricActiveGames');
            if (gamesEl) gamesEl.textContent = data.games_played;
          }
          if (data.training_status) {
            const trainingEl = document.getElementById('metricTrainingStatus');
            if (trainingEl) trainingEl.textContent = data.training_status;
          }
        }
      } catch (e) {
        // Ignore parsing errors for log messages
      }
    };
    orchEventSource.onerror = () => {
      // SSE may fail (browser or server). We'll rely on polling fallback.
    };
  } catch (e) {
    // ignore
  }
}

function stopOrchestratorSSE() {
  if (orchEventSource) {
    try { orchEventSource.close(); } catch {}
    orchEventSource = null;
  }
}

async function loadOrchestratorStatus() {
  try {
    const res = await fetch('/orchestrator/status');
    const data = await res.json();

    const runningEl = document.getElementById('orchRunning');
    const statusDetailEl = document.getElementById('orchStatusDetail');
    const cycleInfoEl = document.getElementById('orchCycleInfo');
    const runtimeEl = document.getElementById('orchRuntime');

    if (data.running) {
      runningEl.textContent = `Running (pid ${data.pid})`;
      runningEl.className = 'status-indicator training';
      statusDetailEl.textContent = data.selfplay ? `Self-play: ${data.selfplay.completed}/${data.selfplay.total}` : 'Running';
      cycleInfoEl.textContent = data.cycle ? `Cycle ${data.cycle}` : '—';
      runtimeEl.textContent = data.started_at ? formatRuntime(data.started_at) : '—';

      // Update metrics dashboard
      const gamesEl = document.getElementById('metricActiveGames');
      if (gamesEl && data.selfplay) {
        gamesEl.textContent = data.selfplay.completed || 0;
      }
      const trainingEl = document.getElementById('metricTrainingStatus');
      if (trainingEl) {
        trainingEl.textContent = data.training?.is_training ? 'Training' : 'Idle';
      }
    } else {
      runningEl.textContent = 'Idle';
      runningEl.className = 'status-indicator not-training';
      statusDetailEl.textContent = 'Ready to start';
      cycleInfoEl.textContent = '—';
      runtimeEl.textContent = '—';

      // Update metrics dashboard
      const gamesEl = document.getElementById('metricActiveGames');
      if (gamesEl) gamesEl.textContent = '0';
      const trainingEl = document.getElementById('metricTrainingStatus');
      if (trainingEl) trainingEl.textContent = 'Idle';
    }

    const completed = data.selfplay?.completed ?? 0;
    const total = data.selfplay?.total ?? 0;
    const pct = data.selfplay?.progress_pct ?? 0;
    const wins = data.selfplay?.wins ?? 0;
    const losses = data.selfplay?.losses ?? 0;
    const draws = data.selfplay?.draws ?? 0;

    document.getElementById('orchCompleted').textContent = completed;
    document.getElementById('orchTotal').textContent = total || '—';
    document.getElementById('orchProgressPct').textContent = `${pct.toFixed(1)}%`;
    document.getElementById('orchWLD').textContent = `${wins}/${losses}/${draws}`;
    document.getElementById('orchProgressBar').style.width = `${Math.min(100, Math.max(0, pct))}%`;

    const evalStatus = data.evaluation?.status || '—';
    document.getElementById('orchEvalStatus').textContent = evalStatus;
    document.getElementById('orchEvalWR').textContent = (data.evaluation?.win_rate != null) ? (data.evaluation.win_rate.toFixed(3)) : '—';
    document.getElementById('orchPromo').textContent = data.promotion?.promoted ? 'Promoted' : '—';
  } catch (e) {
    log('Failed to load orchestrator status: ' + e.message);
  }
}

async function loadOrchestratorWorkers() {
  try {
    const res = await fetch('/orchestrator/workers');
    const data = await res.json();
    const el = document.getElementById('orchWorkersTable');
    if (!el) return;
    const rows = (data.workers || []).map(w => {
      const hb = (w.hb_age != null) ? `${Math.round(w.hb_age)}s` : '—';
      return `W${w.worker}: ${w.done}/${w.games_per_worker} | ${w.avg_ms.toFixed(1)} ms | ${w.avg_sims.toFixed(1)} sims | moves ${w.moves} | HB ${hb}`;
    });
    el.textContent = rows.join('\n');
  } catch (e) {
    // ignore
  }
}

async function loadOrchestratorLogs() {
  try {
    const res = await fetch('/orchestrator/logs?tail=200&kind=structured');
    const data = await res.json();
    const el = document.getElementById('orchLogs');
    if (data.lines && el) {
      el.textContent = data.lines.join('\n');
      el.scrollTop = el.scrollHeight;
    }
  } catch (e) { /* ignore */ }
}

async function startOrchestrator() {
  const payload = {
    config: document.getElementById('orchConfig').value || 'config.yaml',
    games: toInt(document.getElementById('orchGames').value),
    workers: toInt(document.getElementById('orchWorkers').value),
    sims: toInt(document.getElementById('orchSims').value),
    eval_games: toInt(document.getElementById('orchEvalGames').value),
    promotion_threshold: toFloat(document.getElementById('orchPromote').value),
    device: (document.getElementById('orchDevice').value || undefined),
    quick_start: document.getElementById('orchQuickStart').checked,
    no_shared_infer: document.getElementById('orchNoSharedInfer').checked,
    tui: (document.getElementById('orchTui').value || undefined),
    continuous: document.getElementById('orchContinuous').checked,
  };
  try {
    const res = await fetch('/orchestrator/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    log('Orchestrator started: pid ' + d.pid);
    showSuccess('Orchestrator started successfully!');
    loadOrchestratorStatus();
    loadOrchestratorLogs();
  } catch (e) {
    log('Failed to start orchestrator: ' + e.message);
  }
}

async function stopOrchestrator() {
  try {
    const res = await fetch('/orchestrator/stop', { method: 'POST' });
    const d = await res.json();
    log('Orchestrator stop: ' + (d.stopped ? 'ok' : d.message || 'not running'));
    if (d.stopped) {
      showSuccess('Orchestrator stopped successfully!');
    } else {
      showError(d.message || 'Orchestrator was not running');
    }
    loadOrchestratorStatus();
  } catch (e) {
    log('Failed to stop orchestrator: ' + e.message);
  }
}

function toInt(v) { const n = parseInt(v, 10); return Number.isFinite(n) ? n : undefined; }
function toFloat(v) { const n = parseFloat(v); return Number.isFinite(n) ? n : undefined; }

function formatRuntime(startTimestamp) {
  if (!startTimestamp) return '—';
  const start = new Date(startTimestamp * 1000);
  const now = new Date();
  const diffMs = now - start;

  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diffMs % (1000 * 60)) / 1000);

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  } else {
    return `${seconds}s`;
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

async function purgeStaleGames() {
  try {
    const res = await fetch('/admin/purge', { method: 'POST' });
    const data = await res.json();
    const infoEl = document.getElementById('purgeInfo');
    if (infoEl) infoEl.textContent = `Removed: ${data.removed}, Active: ${data.active}`;
  } catch (e) {
    const infoEl = document.getElementById('purgeInfo');
    if (infoEl) infoEl.textContent = `Error: ${e.message}`;
  }
}

async function loadPgnList() {
  try {
    const res = await fetch('/pgn/list');
    const data = await res.json();
    const el = document.getElementById('pgnList');
    if (!el) return;
    if (!data.files || data.files.length === 0) {
      el.textContent = 'No PGN files yet';
      return;
    }
    const links = data.files.reverse().slice(0, 50).map(name => `<a href="/pgn/${name}" target="_blank">${name}</a>`).join('<br/>');
    el.innerHTML = links;
  } catch (e) {
    const el = document.getElementById('pgnList');
    if (el) el.textContent = 'Failed to load PGN list';
  }
}

async function checkEngineAvailability() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const sfAvailable = !!data.stockfish;
    const matrix0Available = data.model_params !== null;

    const disableSFInSelect = (id) => {
      const sel = document.getElementById(id);
      if (!sel) return;
      Array.from(sel.options || []).forEach(opt => {
        if (opt.value === 'stockfish') {
          opt.disabled = !sfAvailable;
          opt.textContent = sfAvailable ? 'Stockfish' : 'Stockfish (Unavailable)';
        }
        if (opt.value === 'matrix0') {
          opt.disabled = !matrix0Available;
          opt.textContent = matrix0Available ? 'Matrix0' : 'Matrix0 (No Model)';
        }
      });
    };

    disableSFInSelect('white');
    disableSFInSelect('black');

    // Tournament engine checkbox
    const sfCb = Array.from(document.querySelectorAll('#engineSelection input[type="checkbox"]')).find(cb => cb.value === 'stockfish');
    if (sfCb) {
      sfCb.disabled = !sfAvailable;
      if (!sfAvailable) sfCb.checked = false;
    }

    // Add engine status indicators
    updateEngineStatusIndicators(sfAvailable, matrix0Available);

    if (!sfAvailable) {
      log('Stockfish not available. Set MATRIX0_STOCKFISH_PATH or install stockfish.');
    }
    if (!matrix0Available) {
      log('Matrix0 model not available. Train or load a model first.');
    }

  } catch (e) {
    debugLog('Failed to check engine availability', e);
  }
}

function updateEngineStatusIndicators(stockfishAvailable, matrix0Available) {
  // Add status indicators next to player selects
  const whiteSelect = document.getElementById('white');
  const blackSelect = document.getElementById('black');

  // Remove existing indicators
  document.querySelectorAll('.engine-status').forEach(el => el.remove());

  const addStatusIndicator = (selectEl, engine, available) => {
    if (!selectEl) return;

    const indicator = document.createElement('span');
    indicator.className = `engine-status status-indicator ${available ? 'ssl-enabled' : 'ssl-disabled'}`;
    indicator.textContent = available ? '✓' : '✗';
    indicator.title = `${engine} ${available ? 'available' : 'unavailable'}`;

    // Insert after the select
    selectEl.parentNode.insertBefore(indicator, selectEl.nextSibling);
  };

  addStatusIndicator(whiteSelect, 'Matrix0', matrix0Available);
  addStatusIndicator(blackSelect, 'Matrix0', matrix0Available);

  // For Stockfish, check if it's selected
  if (whiteSelect.value === 'stockfish') {
    addStatusIndicator(whiteSelect, 'Stockfish', stockfishAvailable);
  }
  if (blackSelect.value === 'stockfish') {
    addStatusIndicator(blackSelect, 'Stockfish', stockfishAvailable);
  }
}

function updateEngineStatusOnChange() {
  // Re-check engine availability and update indicators when selection changes
  checkEngineAvailability();
}

async function viewTrainingConfig() {
  try {
    const response = await fetch('/config/view');
    const data = await response.json();

    if (data.training) {
      let configText = 'Current Training Configuration:\n\n';
      for (const [key, value] of Object.entries(data.training)) {
        configText += `${key}: ${JSON.stringify(value, null, 2)}\n`;
      }

      const historyEl = document.getElementById('trainingHistory');
      historyEl.textContent = configText;
      showSuccess('Training config loaded');
    } else {
      showError('Training config not available');
    }
  } catch (error) {
    debugLog('Failed to load training config', error);
    showError('Failed to load training configuration');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  try { 
    initBoard(); 
    // Fallback: ensure board is created after a short delay
    setTimeout(() => {
      const boardElement = document.getElementById('board');
      if (boardElement && !boardElement.querySelector('.square')) {
        debugLog('Fallback: Creating chess board');
        boardElement.classList.remove('loading');
        createChessBoard(boardElement);
      }
    }, 1000);
  } catch (e) { debugLog('initBoard failed', e); }

  // Game controls
  const newBtn = document.getElementById('new'); if (newBtn) newBtn.addEventListener('click', newGame);
  const modelVsModelBtn = document.getElementById('modelVsModel'); if (modelVsModelBtn) modelVsModelBtn.addEventListener('click', startModelVsModel);
  const matrixVsStockfishBtn = document.getElementById('matrixVsStockfish'); if (matrixVsStockfishBtn) matrixVsStockfishBtn.addEventListener('click', startMatrixVsStockfish);
  const engBtn = document.getElementById('engineMove'); if (engBtn) engBtn.addEventListener('click', engineMove);
  const evalBtn = document.getElementById('evalBtn'); if (evalBtn) evalBtn.addEventListener('click', evalPos);
  const deepAnalysisBtn = document.getElementById('deepAnalysis'); if (deepAnalysisBtn) deepAnalysisBtn.addEventListener('click', performDeepAnalysis);
  const clearEvalBtn = document.getElementById('clearEval'); if (clearEvalBtn) clearEvalBtn.addEventListener('click', clearEvaluation);
  const purgeBtn = document.getElementById('purgeBtn');
  if (purgeBtn) purgeBtn.addEventListener('click', purgeStaleGames);
  const refreshPgnBtn = document.getElementById('refreshPgn');
  if (refreshPgnBtn) refreshPgnBtn.addEventListener('click', loadPgnList);

  // View switching controls
  const bGame = document.getElementById('showGame'); if (bGame) bGame.addEventListener('click', () => switchView('game'));
  const bTrain = document.getElementById('showTraining'); if (bTrain) bTrain.addEventListener('click', () => switchView('training'));
  const bOrch = document.getElementById('showOrchestrator'); if (bOrch) bOrch.addEventListener('click', () => switchView('orchestrator'));
  const bSSL = document.getElementById('showSSL'); if (bSSL) bSSL.addEventListener('click', () => switchView('ssl'));
  const bTourn = document.getElementById('showTournament'); if (bTourn) bTourn.addEventListener('click', () => switchView('tournament'));
  const bAnal = document.getElementById('showAnalysis'); if (bAnal) bAnal.addEventListener('click', () => switchView('analysis'));

  // Training controls
  const refreshTrainingBtn = document.getElementById('refreshTraining'); if (refreshTrainingBtn) refreshTrainingBtn.addEventListener('click', () => loadTrainingStatus());
  const viewTrainingConfigBtn = document.getElementById('viewTrainingConfig'); if (viewTrainingConfigBtn) viewTrainingConfigBtn.addEventListener('click', viewTrainingConfig);
  const refreshMetricsBtn = document.getElementById('refreshMetrics'); if (refreshMetricsBtn) refreshMetricsBtn.addEventListener('click', () => loadModelAnalysis());

  // Tournament controls
  document.getElementById('createTournament').addEventListener('click', createTournament);
  document.getElementById('quickTournament').addEventListener('click', createQuickTournament);

  // Orchestrator controls
  const sBtn = document.getElementById('orchStart');
  const xBtn = document.getElementById('orchStop');
  if (sBtn) sBtn.addEventListener('click', startOrchestrator);
  if (xBtn) xBtn.addEventListener('click', stopOrchestrator);

  // Load initial data
  loadAnalytics();
  loadPgnList();
  checkEngineAvailability();

  // Add event listeners for player select changes
  document.getElementById('white').addEventListener('change', updateEngineStatusOnChange);
  document.getElementById('black').addEventListener('change', updateEngineStatusOnChange);

  // Start periodic updates
  startPeriodicUpdates();

  // Start connection monitoring
  checkServerConnection(); // Initial check
  setInterval(checkServerConnection, 30000); // Check every 30 seconds

  // Load only the current view's data; others load on tab switch
  switchView('game');
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

async function createQuickTournament() {
  try {
    const name = `Quick Match ${new Date().toLocaleTimeString()}`;
    const selectedEngines = ['matrix0', 'stockfish']; // Always include Matrix0 and Stockfish for quick matches

    const config = {
      name: name,
      engines: selectedEngines,
      format: 'round_robin',
      num_games_per_pairing: 1,
      time_control_ms: 100,
      max_concurrency: 1
    };

    const response = await fetch('/tournament/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });

    if (response.ok) {
      const result = await response.json();
      log(`Quick tournament created: ${result.tournament_id}`);
      showSuccess('Quick tournament started!');
      document.getElementById('tournamentName').value = '';
      await loadTournamentData();
    } else {
      const error = await response.json();
      showError(`Failed to create quick tournament: ${error.detail}`);
    }
  } catch (error) {
    log(`Quick tournament creation failed: ${error.message}`);
    showError('Failed to create quick tournament');
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
        <h4><i class="fas fa-trophy"></i> ${tournament.name}</h4>
        <div class="tournament-stats">
          <div class="tournament-stat">
            <div class="tournament-stat-label">Status</div>
            <div class="tournament-stat-value">
              <span class="status-indicator ${tournament.status === 'running' ? 'training' : 'not-training'}">${tournament.status}</span>
            </div>
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
        <div class="tournament-engines" style="margin-top: 8px; font-size: 12px; color: var(--muted);">
          <strong>Participants:</strong> ${tournament.engines.join(', ')}
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
