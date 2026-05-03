// baibot live UI — frontend wiring
//
// Renders ChartPayload (see tradingagents/web/overlays/base.py) onto a
// lightweight-charts candlestick chart, with overlays for chan/intraday
// strategies. Subscribes to /events/stream for the live event feed.

// ── State ────────────────────────────────────────────────────────────
const state = {
  variants: [],
  selectedVariant: null,
  trades: [],
  selectedTradeId: null,
  chart: null,
  candleSeries: null,
  overlaySeries: [],   // line series we own; cleared on each chart redraw
  fillPriceLines: [],  // IPriceLine handles for entry/exit price lines
  zsBoxes:       [],   // {el, from_t, to_t, low, high} — DOM rectangles
  zsLayer:       null, // overlay div sibling to the chart canvas
  zsRafActive:   false,
  // Indicator series (rebuilt per trade selection)
  volumeSeries: null,
  macdLineSeries: null,
  macdSignalSeries: null,
  macdHistSeries: null,
  // Persisted indicator toggles (survive chart redraws)
  indicators: {
    structures: true,  // chan BI/SEG/ZS/BSP overlays
    volume:     true,
    macd:       false,
  },
  currentPayload: null,  // last-rendered payload, for re-render after toggle
  // Live UX
  variantStream: null,         // EventSource for trade arrivals on selected variant
  incomingTrades: [],          // arrived this session, not yet viewed (clickable queue)
  wakeLockEnabled: false,      // user-intent flag (the real lock is below)
  wakeLock: null,              // active WakeLockSentinel, if any
  lastActivityMs: null,        // unix ms of last trade we saw
  lastActivityTimer: null,     // setInterval handle for the topbar clock
  // Live-tail (auto-refresh open positions)
  tailTimer: null,             // setInterval for the open-trade polling
  tailIntervalMs: 60_000,      // refresh cadence
};

// ── Init ─────────────────────────────────────────────────────────────
async function init() {
  buildChart();
  wireToolbar();
  wireKeyboard();
  wireLiveControls();
  wireRouter();
  startLastActivityTicker();
  await loadVariants();
  if (state.variants.length === 0) {
    document.getElementById('chart-error').textContent = 'no variants found — check experiments/paper_launch_v2.yaml';
    document.getElementById('chart-error').hidden = false;
    return;
  }
  // Set selectedVariant + dropdown BEFORE routing — every route's
  // onActivate (TODAY, PERFORMANCE, etc.) reads state.selectedVariant.
  // If we route first, landing directly on /#today shows a blank page
  // because renderTodayPage early-returns on no variant.
  state.selectedVariant = state.variants[0].name;
  populateVariantSelect();
  routeFromHash();
  await loadTrades(state.selectedVariant);
  startEventStream();
}

function buildChart() {
  const el = document.getElementById('chart');
  const chart = LightweightCharts.createChart(el, {
    layout: {
      background: { color: '#0a0e14' },
      textColor: '#9ca3af',
      fontFamily: 'IBM Plex Mono, Menlo, monospace',
    },
    grid: {
      vertLines: { color: '#1a2230' },
      horzLines: { color: '#1a2230' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1f2937' },
    timeScale: {
      borderColor: '#1f2937',
      timeVisible: true,
      secondsVisible: false,
    },
    autoSize: true,
  });
  const candle = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  state.chart = chart;
  state.candleSeries = candle;

  // Double-click anywhere on the chart fits content. Matches the
  // TradingView pattern of double-clicking the time axis to reset.
  el.addEventListener('dblclick', () => fitChart());

  // ZS overlay layer — DOM rectangles positioned via the chart's
  // coordinate API. lightweight-charts v4 has no rectangle primitive,
  // so we render boxes as absolutely-positioned divs and reposition
  // them every frame against series.priceToCoordinate / timeScale.
  el.style.position = 'relative';
  const zsLayer = document.createElement('div');
  zsLayer.id = 'zs-layer';
  Object.assign(zsLayer.style, {
    position: 'absolute', inset: '0',
    pointerEvents: 'none', zIndex: '2',
  });
  el.appendChild(zsLayer);
  state.zsLayer = zsLayer;

  // Watch resize
  window.addEventListener('resize', () => chart.applyOptions({}));
}

// ── ZS box positioning ──────────────────────────────────────────────
function repositionZS() {
  if (!state.zsBoxes.length) return;
  const ts = state.chart.timeScale();
  for (const box of state.zsBoxes) {
    const x1 = ts.timeToCoordinate(box.from_t);
    const x2 = ts.timeToCoordinate(box.to_t);
    const yT = state.candleSeries.priceToCoordinate(box.high);
    const yB = state.candleSeries.priceToCoordinate(box.low);
    if (x1 == null || x2 == null || yT == null || yB == null) {
      box.el.style.display = 'none';
      continue;
    }
    box.el.style.display = 'block';
    box.el.style.left   = `${Math.min(x1, x2)}px`;
    box.el.style.top    = `${Math.min(yT, yB)}px`;
    box.el.style.width  = `${Math.abs(x2 - x1)}px`;
    box.el.style.height = `${Math.abs(yB - yT)}px`;
  }
}

function startZSLoop() {
  if (state.zsRafActive) return;
  state.zsRafActive = true;
  const tick = () => {
    if (state.zsBoxes.length === 0) {
      state.zsRafActive = false;
      return;
    }
    repositionZS();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ── Indicator math (volume, MACD) ───────────────────────────────────
function ema(values, period) {
  const k = 2 / (period + 1);
  const out = new Array(values.length);
  let prev = values[0];
  out[0] = prev;
  for (let i = 1; i < values.length; i++) {
    prev = values[i] * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

function computeMACD(bars) {
  if (bars.length < 26) return { line: [], signal: [], hist: [] };
  const close  = bars.map(b => b.close);
  const e12    = ema(close, 12);
  const e26    = ema(close, 26);
  const macd   = close.map((_, i) => e12[i] - e26[i]);
  const signal = ema(macd, 9);
  const hist   = macd.map((v, i) => v - signal[i]);
  const t      = bars.map(b => b.time);
  return {
    line:   t.map((time, i) => ({ time, value: macd[i] })),
    signal: t.map((time, i) => ({ time, value: signal[i] })),
    hist:   t.map((time, i) => ({
      time,
      value: hist[i],
      color: hist[i] >= 0 ? 'rgba(94,197,183,0.8)' : 'rgba(232,90,90,0.8)',
    })),
  };
}

function buildVolumeData(bars) {
  return bars
    .filter(b => b.volume != null)
    .map(b => ({
      time:  b.time,
      value: b.volume,
      color: b.close >= b.open
        ? 'rgba(94,197,183,0.55)'
        : 'rgba(232,90,90,0.55)',
    }));
}

// ── Pane layout — adjust scaleMargins as indicators toggle ──────────
function applyPaneLayout() {
  const v = state.indicators.volume && state.volumeSeries;
  const m = state.indicators.macd   && state.macdLineSeries;
  const main = state.candleSeries.priceScale();

  if (v && m) {
    // Three stacked: candles 0–55, volume 60–75, MACD 80–100
    main.applyOptions({ scaleMargins: { top: 0.04, bottom: 0.45 }});
    state.chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.60, bottom: 0.25 }});
    state.chart.priceScale('macd').applyOptions({ scaleMargins:   { top: 0.80, bottom: 0.00 }});
  } else if (v) {
    main.applyOptions({ scaleMargins: { top: 0.04, bottom: 0.25 }});
    state.chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.78, bottom: 0.00 }});
  } else if (m) {
    main.applyOptions({ scaleMargins: { top: 0.04, bottom: 0.22 }});
    state.chart.priceScale('macd').applyOptions({ scaleMargins:   { top: 0.80, bottom: 0.00 }});
  } else {
    main.applyOptions({ scaleMargins: { top: 0.04, bottom: 0.04 }});
  }
}

function teardownIndicators() {
  if (state.volumeSeries)     { state.chart.removeSeries(state.volumeSeries);     state.volumeSeries = null; }
  if (state.macdLineSeries)   { state.chart.removeSeries(state.macdLineSeries);   state.macdLineSeries = null; }
  if (state.macdSignalSeries) { state.chart.removeSeries(state.macdSignalSeries); state.macdSignalSeries = null; }
  if (state.macdHistSeries)   { state.chart.removeSeries(state.macdHistSeries);   state.macdHistSeries = null; }
}

function buildIndicators(bars) {
  if (state.indicators.volume) {
    state.volumeSeries = state.chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    state.volumeSeries.setData(buildVolumeData(bars));
  }
  if (state.indicators.macd) {
    const m = computeMACD(bars);
    state.macdHistSeries = state.chart.addHistogramSeries({
      priceScaleId: 'macd',
      lastValueVisible: false,
      priceLineVisible: false,
      base: 0,
    });
    state.macdHistSeries.setData(m.hist);

    state.macdLineSeries = state.chart.addLineSeries({
      priceScaleId: 'macd',
      color: '#f5a524', lineWidth: 1,
      lastValueVisible: false, priceLineVisible: false,
      crosshairMarkerVisible: false,
      title: 'MACD',
    });
    state.macdLineSeries.setData(m.line);

    state.macdSignalSeries = state.chart.addLineSeries({
      priceScaleId: 'macd',
      color: '#5ec5b7', lineWidth: 1,
      lastValueVisible: false, priceLineVisible: false,
      crosshairMarkerVisible: false,
      title: 'signal',
    });
    state.macdSignalSeries.setData(m.signal);
  }
  applyPaneLayout();
}

// ── Toolbar / keyboard ───────────────────────────────────────────────
function setBtnActive(id, on) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('active', on);
}

function fitChart() {
  state.chart.timeScale().fitContent();
  state.candleSeries.priceScale().applyOptions({ autoScale: true });
  if (state.volumeSeries)   state.chart.priceScale('volume').applyOptions({ autoScale: true });
  if (state.macdLineSeries) state.chart.priceScale('macd').applyOptions({ autoScale: true });
}

function toggleStructures() {
  state.indicators.structures = !state.indicators.structures;
  setBtnActive('btn-structures', state.indicators.structures);
  // Hide via series options + zs-layer visibility
  for (const s of state.overlaySeries) {
    try { s.applyOptions({ visible: state.indicators.structures }); } catch {}
  }
  if (state.zsLayer) {
    state.zsLayer.style.display = state.indicators.structures ? 'block' : 'none';
  }
  // Markers (BSPs + fills) — re-set the fills only when structures are off
  if (state.currentPayload) {
    const fillMarkers = _buildFillMarkers(state.currentPayload.fills);
    let bspMarkers = [];
    if (state.indicators.structures) {
      bspMarkers = (state.currentPayload.overlays || [])
        .filter(o => o.kind === 'marker')
        .map(markerToSeries);
    }
    state.candleSeries.setMarkers(
      [...bspMarkers, ...fillMarkers].sort((a, b) => a.time - b.time),
    );
  }
}

function toggleVolume() {
  state.indicators.volume = !state.indicators.volume;
  setBtnActive('btn-volume', state.indicators.volume);
  if (!state.currentPayload) return;
  teardownIndicators();
  buildIndicators(state.currentPayload.bars);
}

function toggleMACD() {
  state.indicators.macd = !state.indicators.macd;
  setBtnActive('btn-macd', state.indicators.macd);
  if (!state.currentPayload) return;
  teardownIndicators();
  buildIndicators(state.currentPayload.bars);
}

function wireToolbar() {
  document.getElementById('btn-fit')       .addEventListener('click', fitChart);
  document.getElementById('btn-structures').addEventListener('click', toggleStructures);
  document.getElementById('btn-volume')    .addEventListener('click', toggleVolume);
  document.getElementById('btn-macd')      .addEventListener('click', toggleMACD);
}

function wireKeyboard() {
  document.addEventListener('keydown', e => {
    // Ignore when typing into form fields
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key.toLowerCase()) {
      case 'f': fitChart();          break;
      case 's': toggleStructures();  break;
      case 'v': toggleVolume();      break;
      case 'm': toggleMACD();        break;
      default: return;
    }
    e.preventDefault();
  });
}

// ── ROUTER ───────────────────────────────────────────────────────────
//
// Hash-based SPA router. Each route corresponds to a `<section class="route"
// id="route-X">` in the HTML. Activating a route hides every other route
// section, sets the matching tab's `.active`, and calls the route's
// onActivate hook (if defined) to refresh data lazily.

const ROUTES = {
  trade:       { onActivate: () => {
    // Verified via headless Chrome: when the route div toggles from
    // display:none → flex, .chart-frame regains its 604px height but the
    // child #chart stays at height: 0. lightweight-charts' autoSize
    // observer locked inline width:0; height:0 on #chart while it was
    // hidden, and ResizeObserver doesn't re-fire for content boxes whose
    // observed size is held flat by an inline style (chicken-and-egg).
    //
    // Recovery: clear the locked inline style, toggle autoSize off+on
    // (forces ResizeObserver to disconnect and re-observe the now-visible
    // container), and explicitly resize from .chart-frame's dimensions
    // before re-rendering the cached payload.
    if (!state.currentPayload) return;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const frame = document.querySelector('#route-trade .chart-frame');
      const el = document.getElementById('chart');
      if (!frame || frame.clientWidth === 0 || frame.clientHeight === 0) return;
      if (el) { el.style.width = ''; el.style.height = ''; }
      try {
        state.chart.applyOptions({ autoSize: false });
        state.chart.resize(frame.clientWidth, frame.clientHeight);
        state.chart.applyOptions({ autoSize: true });
      } catch (e) { console.error('chart resize failed', e); }
      renderPayload(state.currentPayload);
    }));
  } },
  today:       { onActivate: () => renderTodayPage() },
  performance: { onActivate: () => renderPerformancePage() },
  reviews:     { onActivate: () => renderReviewsPage() },
  portfolio:   { onActivate: () => renderRiskPage() },
  proposals:   { onActivate: () => renderProposalsPage() },
  ideas:       { onActivate: () => renderIdeasPage() },
  log:         { onActivate: () => renderLogPage(), onDeactivate: () => stopLogAutoRefresh() },
};

function wireRouter() {
  document.querySelectorAll('.tab').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const r = a.dataset.route;
      window.location.hash = r;
    });
  });
  window.addEventListener('hashchange', routeFromHash);
}

let _activeRoute = null;
function routeFromHash() {
  const r = (window.location.hash.replace('#', '') || 'trade').toLowerCase();
  const route = ROUTES[r] ? r : 'trade';
  // Fire prior route's deactivate hook so it can stop intervals etc.
  if (_activeRoute && _activeRoute !== route) {
    try { ROUTES[_activeRoute].onDeactivate?.(); } catch (e) { console.error(e); }
  }
  document.querySelectorAll('.route').forEach(el => {
    el.hidden = (el.id !== `route-${route}`);
  });
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.route === route);
  });
  _activeRoute = route;
  try { ROUTES[route].onActivate(); } catch (e) { console.error(e); }
}

// ── Page: TODAY ──────────────────────────────────────────────────────
let _todayChart = null, _todayChartSeries = null;

async function renderTodayPage() {
  const variant = state.selectedVariant;
  if (!variant) return;
  const meta = document.getElementById('today-meta');
  meta.textContent = `${variant} · loading…`;

  let data;
  try {
    const res = await fetch(`/api/today/${encodeURIComponent(variant)}`);
    if (!res.ok) { meta.textContent = `error ${res.status}`; return; }
    data = await res.json();
  } catch (e) { meta.textContent = String(e); return; }

  const acct = data.account || {};
  const positions = data.positions || [];
  const startEq = data.starting_equity || 0;
  const curEq   = Number(acct.equity || (data.snapshots.at?.(-1)?.equity) || 0);
  const cash    = Number(acct.cash || 0);
  const dayPL   = Number(acct.daily_pl || 0);
  const dayPLPct= acct.daily_pl_pct || '0.00%';
  const vsStart = curEq - startEq;
  const vsStartPct = startEq ? (vsStart / startEq * 100) : 0;
  const maxPos = data.max_positions || 10;

  meta.textContent = `${variant} · ${data.strategy_type}`;

  // Service-health pill (mirrors Streamlit's _service_pill)
  fetch('/api/service_status').then(r => r.json()).then(s => {
    const pill = document.getElementById('today-svc');
    if (!pill) return;
    pill.classList.toggle('running', !!s.running);
    pill.lastChild.textContent = s.label || (s.running ? 'RUNNING' : 'STOPPED');
  }).catch(() => {});

  // KPI row
  const kpiHtml = [
    kpiCard('EQUITY',     fmtUSD(curEq)),
    kpiCard('CASH',       fmtUSD(cash)),
    kpiCard('DAY P&L',    fmtUSDSigned(dayPL), dayPLPct, dayPL >= 0 ? 'pos' : 'neg'),
    kpiCard('VS. START',  fmtUSDSigned(vsStart),
            `${vsStartPct >= 0 ? '+' : ''}${vsStartPct.toFixed(2)}%`,
            vsStart >= 0 ? 'pos' : 'neg'),
    kpiCard('POSITIONS',  String(positions.length), `max ${maxPos}`),
  ].join('');
  document.getElementById('today-kpi').innerHTML = kpiHtml;

  // Equity chart (lightweight-charts area series)
  if (!_todayChart) {
    _todayChart = LightweightCharts.createChart(document.getElementById('today-equity-chart'), {
      layout: { background: { color: '#111114' }, textColor: '#9aa5b1', fontFamily: 'JetBrains Mono, monospace' },
      grid:   { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: { borderColor: '#1f2937', timeVisible: false },
      autoSize: true,
    });
    _todayChartSeries = _todayChart.addAreaSeries({
      lineColor: '#f5a524', topColor: 'rgba(245,165,36,0.18)',
      bottomColor: 'rgba(245,165,36,0.01)', lineWidth: 2,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });
  }
  const ed = (data.snapshots || [])
    .map(s => ({ time: dateToUnix(s.date), value: Number(s.equity) }))
    .filter(p => !isNaN(p.value));
  _todayChartSeries.setData(ed);
  _todayChart.timeScale().fitContent();
  document.getElementById('today-equity-meta').textContent =
    `${vsStartPct >= 0 ? '+' : ''}${vsStartPct.toFixed(2)}% total`;

  // Positions
  document.getElementById('today-positions').innerHTML = renderPositionsTable(positions);
  document.getElementById('today-positions-meta').textContent =
    `${positions.length} / ${maxPos}`;

  // Trades
  document.getElementById('today-trades').innerHTML = renderTradesTable(data.trades);
}

function kpiCard(label, value, sub = '', accent = '') {
  const subHtml = sub ? `<div class="kpi-sub ${accent}">${escapeHtml(sub)}</div>` : '';
  const cls = (accent === 'pos' || accent === 'neg') ? `kpi ${accent}` : 'kpi';
  return `<div class="${cls}"><div class="kpi-label">${escapeHtml(label)}</div>` +
         `<div class="kpi-value">${escapeHtml(value)}</div>${subHtml}</div>`;
}

function renderPositionsTable(positions) {
  if (!positions || positions.length === 0)
    return '<div class="tb-empty">— no open positions —</div>';
  const rows = positions.map(p => {
    const pnl = Number(p.pl || 0);
    const cls = pnl >= 0 ? 'pos' : 'neg';
    const pct = p.pl_pct || '0.00%';
    return `<tr>
      <td><span class="sym">${escapeHtml(p.symbol || '')}</span></td>
      <td class="num">${Math.round(Number(p.qty || 0))}</td>
      <td class="num">${fmtUSD(Number(p.entry || 0))}</td>
      <td class="num">${fmtUSD(Number(p.current || 0))}</td>
      <td class="num ${cls}">${fmtUSDSigned(pnl)}<div style="font-size:0.66rem;">${escapeHtml(pct)}</div></td>
    </tr>`;
  }).join('');
  return `<table class="tb"><thead><tr>
    <th>SYMBOL</th><th class="num">QTY</th><th class="num">ENTRY</th>
    <th class="num">CURRENT</th><th class="num">P&amp;L</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

function renderTradesTable(trades) {
  if (!trades || trades.length === 0)
    return '<div class="tb-empty">— no trades yet —</div>';
  const rows = trades.slice(0, 50).map(t => {
    const side = (t.side || '').toLowerCase();
    const tag  = side === 'buy' ? 'tag-buy' : 'tag-sell';
    const ts   = formatTradeTime(t.timestamp);
    const fp   = t.filled_price ? fmtUSD(Number(t.filled_price)) : '—';
    const status = (t.status || '').toString().replace('OrderStatus.', '');
    return `<tr>
      <td>${escapeHtml(ts)}</td>
      <td><span class="tag ${tag}">${side.toUpperCase()}</span></td>
      <td><span class="sym">${escapeHtml(t.symbol || '')}</span></td>
      <td class="num">${Math.round(Number(t.filled_qty || t.qty || 0))}</td>
      <td class="num">${fp}</td>
      <td>${escapeHtml(status)}</td>
      <td style="color:var(--text-muted); font-size:0.7rem;">${escapeHtml((t.reasoning || '').slice(0, 80))}</td>
    </tr>`;
  }).join('');
  return `<table class="tb"><thead><tr>
    <th>TIME</th><th>SIDE</th><th>SYMBOL</th>
    <th class="num">QTY</th><th class="num">PRICE</th><th>STATUS</th><th>REASON</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

// ── Page: PERFORMANCE ────────────────────────────────────────────────
const _perf = {
  variants: null,           // {name: snapshots[]}
  trades: null,             // {name: trades[]}
  selected: new Set(),      // active variant filter
  period: 'D',              // D | W | ME
  charts: {},               // id → {chart, series[]}
  inited: false,
};

const _PERF_COLORS = {
  mechanical: '#5ec5b7', llm: '#5b8def', chan: '#f5a524',
  mechanical_v2: '#84d484', chan_v2: '#ffb74d',
  intraday_mechanical: '#a78bfa', chan_daily: '#e85a9a',
  pead: '#22d3ee', pead_llm: '#0891b2',
};

async function renderPerformancePage() {
  if (!_perf.inited) {
    wirePerformanceControls();
    _perf.inited = true;
  }
  const meta = document.getElementById('perf-meta');
  meta.textContent = 'loading…';

  try {
    const [s, t] = await Promise.all([
      fetch('/api/performance/snapshots').then(r => r.json()),
      fetch('/api/performance/trades').then(r => r.json()),
    ]);
    _perf.variants = s.variants;
    _perf.trades   = t.variants;
  } catch (e) { meta.textContent = String(e); return; }

  // Default-select all variants on first load
  if (_perf.selected.size === 0) {
    Object.keys(_perf.variants).forEach(v => _perf.selected.add(v));
  }
  meta.textContent = `${Object.keys(_perf.variants).length} variants · ${_perf.period}`;
  renderVariantFilter();
  redrawPerformanceCharts();
}

function wirePerformanceControls() {
  document.querySelectorAll('#perf-period .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#perf-period .seg-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      _perf.period = b.dataset.period;
      redrawPerformanceCharts();
    });
  });
}

function renderVariantFilter() {
  const wrap = document.getElementById('perf-variants');
  wrap.innerHTML = '';
  for (const name of Object.keys(_perf.variants)) {
    const color = _PERF_COLORS[name] || '#9ca3af';
    const btn = document.createElement('button');
    btn.className = 'vf-chip' + (_perf.selected.has(name) ? ' active' : '');
    btn.style.setProperty('--vf-color', color);
    btn.textContent = name;
    btn.addEventListener('click', () => {
      if (_perf.selected.has(name)) _perf.selected.delete(name);
      else _perf.selected.add(name);
      btn.classList.toggle('active');
      redrawPerformanceCharts();
    });
    wrap.appendChild(btn);
  }
}

function _ensurePerfChart(id, opts = {}) {
  if (_perf.charts[id]) return _perf.charts[id];
  const el = document.getElementById(id);
  const chart = LightweightCharts.createChart(el, {
    layout: { background: { color: '#111114' }, textColor: '#9aa5b1', fontFamily: 'JetBrains Mono, monospace' },
    grid:   { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
    rightPriceScale: { borderColor: '#1f2937' },
    timeScale: { borderColor: '#1f2937', timeVisible: false },
    autoSize: true,
    ...opts,
  });
  _perf.charts[id] = { chart, series: [] };
  return _perf.charts[id];
}

function _clearChartSeries(holder) {
  for (const s of holder.series) {
    try { holder.chart.removeSeries(s); } catch {}
  }
  holder.series = [];
}

function _resampleByPeriod(points, period) {
  // points: [{time: unix, value: number}], period: D|W|ME
  if (period === 'D' || !points.length) return points;
  const grouped = new Map();
  for (const p of points) {
    const d = new Date(p.time * 1000);
    let key;
    if (period === 'W') {
      const day = d.getUTCDay();
      const monday = new Date(d);
      monday.setUTCDate(d.getUTCDate() - ((day + 6) % 7));
      key = monday.toISOString().slice(0, 10);
    } else { // ME
      key = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}`;
    }
    grouped.set(key, p);  // last value wins (resample-last)
  }
  return Array.from(grouped.values()).sort((a, b) => a.time - b.time);
}

function redrawPerformanceCharts() {
  // Equity curves
  const eq = _ensurePerfChart('perf-equity-chart');
  _clearChartSeries(eq);
  for (const name of _perf.selected) {
    const snaps = _perf.variants[name] || [];
    if (!snaps.length) continue;
    const data = snaps
      .map(s => ({ time: dateToUnix(s.date), value: Number(s.equity) }))
      .filter(p => !isNaN(p.value));
    const s = eq.chart.addLineSeries({
      color: _PERF_COLORS[name] || '#9ca3af',
      lineWidth: 2,
      title: name,
      priceLineVisible: false, lastValueVisible: false,
    });
    s.setData(data);
    eq.series.push(s);
  }
  eq.chart.timeScale().fitContent();

  // Period returns
  const ret = _ensurePerfChart('perf-returns-chart');
  _clearChartSeries(ret);
  for (const name of _perf.selected) {
    const snaps = _perf.variants[name] || [];
    if (snaps.length < 2) continue;
    const eqPts = snaps
      .map(s => ({ time: dateToUnix(s.date), value: Number(s.equity) }))
      .filter(p => !isNaN(p.value));
    const resampled = _resampleByPeriod(eqPts, _perf.period);
    const data = [];
    for (let i = 1; i < resampled.length; i++) {
      const r = (resampled[i].value / resampled[i - 1].value - 1) * 100;
      data.push({
        time:  resampled[i].time,
        value: r,
        color: r >= 0 ? 'rgba(125, 216, 127, 0.65)' : 'rgba(232, 90, 90, 0.65)',
      });
    }
    const s = ret.chart.addHistogramSeries({
      color: _PERF_COLORS[name] || '#9ca3af',
      priceFormat: { type: 'percent' },
      priceLineVisible: false, lastValueVisible: false,
    });
    s.setData(data);
    ret.series.push(s);
  }
  ret.chart.timeScale().fitContent();

  // Drawdown
  const dd = _ensurePerfChart('perf-drawdown-chart');
  _clearChartSeries(dd);
  for (const name of _perf.selected) {
    const snaps = _perf.variants[name] || [];
    if (!snaps.length) continue;
    let peak = 0;
    const data = snaps
      .map(s => ({ time: dateToUnix(s.date), v: Number(s.equity) }))
      .filter(p => !isNaN(p.v))
      .map(p => {
        if (p.v > peak) peak = p.v;
        const ddv = peak > 0 ? -((peak - p.v) / peak) * 100 : 0;
        return { time: p.time, value: ddv };
      });
    const s = dd.chart.addAreaSeries({
      lineColor: _PERF_COLORS[name] || '#9ca3af',
      topColor: 'rgba(232, 90, 90, 0.18)',
      bottomColor: 'rgba(232, 90, 90, 0.02)',
      lineWidth: 1,
      title: name,
      priceLineVisible: false, lastValueVisible: false,
    });
    s.setData(data);
    dd.series.push(s);
  }
  dd.chart.timeScale().fitContent();

  // Trade activity (count per period)
  const ac = _ensurePerfChart('perf-activity-chart');
  _clearChartSeries(ac);
  for (const name of _perf.selected) {
    const trades = _perf.trades[name] || [];
    if (!trades.length) continue;
    const buckets = new Map();
    for (const t of trades) {
      if (!t.timestamp) continue;
      const ts = new Date(String(t.timestamp).replace(' ', 'T'));
      if (isNaN(ts.getTime())) continue;
      let key;
      if (_perf.period === 'W') {
        const day = ts.getUTCDay();
        const monday = new Date(ts);
        monday.setUTCDate(ts.getUTCDate() - ((day + 6) % 7));
        key = monday.toISOString().slice(0, 10);
      } else if (_perf.period === 'ME') {
        key = `${ts.getUTCFullYear()}-${String(ts.getUTCMonth() + 1).padStart(2, '0')}-01`;
      } else {
        key = ts.toISOString().slice(0, 10);
      }
      buckets.set(key, (buckets.get(key) || 0) + 1);
    }
    const data = Array.from(buckets.entries())
      .sort()
      .map(([k, n]) => ({ time: Math.floor(new Date(k).getTime() / 1000), value: n }));
    const s = ac.chart.addHistogramSeries({
      color: _PERF_COLORS[name] || '#9ca3af',
      priceFormat: { type: 'volume' },
      priceLineVisible: false, lastValueVisible: false,
    });
    s.setData(data);
    ac.series.push(s);
  }
  ac.chart.timeScale().fitContent();
}

// ── Page: REVIEWS ────────────────────────────────────────────────────
const _reviews = {
  inited: false,
  mode: 'daily',
  date: null,         // YYYY-MM-DD
  includeDryRun: false,
  payload: null,      // current daily/weekly listing payload
  selectedVariant: null,
  selectedFile: null, // string stem
};

async function renderReviewsPage() {
  if (!_reviews.inited) {
    wireReviewsControls();
    _reviews.date = new Date().toISOString().slice(0, 10);
    document.getElementById('reviews-date').value = _reviews.date;
    _reviews.inited = true;
  }
  // Top overview (cross-variant cards + totals + 30d normalized strip)
  loadReviewsOverview();
  await loadReviewsListing();
}

let _reviewsNormChart = null, _reviewsNormSeries = [];
async function loadReviewsOverview() {
  let data;
  try {
    const res = await fetch('/api/reviews/overview');
    data = await res.json();
  } catch { return; }

  // Cards
  const cardsHtml = (data.cards || []).map(c => {
    const equity = c.equity != null ? `$${Math.round(c.equity).toLocaleString()}` : '—';
    const pl = c.daily_pl;
    const plPct = c.daily_pl_pct;
    const plCls = (pl ?? 0) >= 0 ? 'pos' : 'neg';
    const plLine = pl != null
      ? `${pl >= 0 ? '+' : ''}${Math.round(pl).toLocaleString()}` +
        (plPct != null ? ` (${(plPct * 100).toFixed(2)}%)` : '')
      : '—';
    const closed = c.trades_closed || 0;
    const wl = closed
      ? `<b>${closed}</b> closed today · <span class="pos">${c.wins}W</span> / <span class="neg">${c.losses}L</span>`
      : `<span class="rc-meta">No trades closed today</span>`;
    const avgRet = c.avg_return != null ? `${(c.avg_return * 100).toFixed(2)}%` : '—';
    const mfeCap = c.mfe_capture != null ? c.mfe_capture.toFixed(2) : '—';
    const metrics = closed
      ? `avg ${avgRet} · MFE capture ${mfeCap}`
      : '<span class="rc-meta">(reviews fire on exits)</span>';
    return `<div class="reviews-card">
      <div class="rc-name">${escapeHtml(c.variant)}</div>
      <div class="rc-equity">${equity}</div>
      <div class="rc-pl ${plCls}">${escapeHtml(plLine)}</div>
      <div class="rc-line">${wl}</div>
      <div class="rc-meta">${metrics}</div>
    </div>`;
  }).join('');
  document.getElementById('reviews-overview').innerHTML = cardsHtml;

  // Totals row
  const t = data.totals || {};
  document.getElementById('reviews-totals').innerHTML = `
    <div class="rt-cell"><div class="rt-label">TOTAL EQUITY</div><div class="rt-value">$${Math.round(t.equity || 0).toLocaleString()}</div></div>
    <div class="rt-cell"><div class="rt-label">TODAY P&amp;L</div><div class="rt-value ${(t.daily_pl || 0) >= 0 ? 'pos' : 'neg'}">${(t.daily_pl || 0) >= 0 ? '+' : ''}${Math.round(t.daily_pl || 0).toLocaleString()}</div></div>
    <div class="rt-cell"><div class="rt-label">TRADES CLOSED</div><div class="rt-value">${t.trades_closed || 0}</div></div>
    <div class="rt-cell"><div class="rt-label">W / L</div><div class="rt-value">${t.wins || 0} / ${t.losses || 0}</div></div>`;

  // 30d normalized equity strip
  if (!_reviewsNormChart) {
    _reviewsNormChart = LightweightCharts.createChart(document.getElementById('reviews-norm-chart'), {
      layout: { background: { color: '#111114' }, textColor: '#9aa5b1', fontFamily: 'JetBrains Mono, monospace' },
      grid:   { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: { borderColor: '#1f2937', timeVisible: false },
      autoSize: true,
    });
  }
  for (const s of _reviewsNormSeries) {
    try { _reviewsNormChart.removeSeries(s); } catch {}
  }
  _reviewsNormSeries = [];
  for (const [name, points] of Object.entries(data.normalized || {})) {
    const series = _reviewsNormChart.addLineSeries({
      color: _PERF_COLORS[name] || '#9ca3af',
      lineWidth: 2, title: name,
      priceLineVisible: false, lastValueVisible: false,
    });
    series.setData(
      points.map(p => ({ time: dateToUnix(p.date), value: Number(p.equity_norm) }))
            .filter(p => !isNaN(p.value))
    );
    _reviewsNormSeries.push(series);
  }
  _reviewsNormChart.timeScale().fitContent();
}

function wireReviewsControls() {
  document.querySelectorAll('#reviews-mode .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#reviews-mode .seg-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      _reviews.mode = b.dataset.mode;
      loadReviewsListing();
    });
  });
  document.getElementById('reviews-date').addEventListener('change', e => {
    _reviews.date = e.target.value;
    loadReviewsListing();
  });
  document.getElementById('reviews-dry').addEventListener('change', e => {
    _reviews.includeDryRun = e.target.checked;
    loadReviewsListing();
  });
}

async function loadReviewsListing() {
  const meta = document.getElementById('reviews-meta');
  meta.textContent = 'loading…';
  const path = _reviews.mode === 'daily'
    ? `/api/reviews/daily/${_reviews.date}?include_dry_run=${_reviews.includeDryRun}`
    : `/api/reviews/weekly/${_reviews.date}?include_dry_run=${_reviews.includeDryRun}`;
  let data;
  try {
    const res = await fetch(path);
    data = await res.json();
  } catch (e) { meta.textContent = String(e); return; }
  _reviews.payload = data;
  if (_reviews.mode === 'daily') {
    const variants = Object.keys(data.by_variant || {});
    if (!_reviews.selectedVariant || !variants.includes(_reviews.selectedVariant)) {
      _reviews.selectedVariant = variants[0] || null;
    }
    meta.textContent = data.exists
      ? `daily · ${variants.length} variants · ${data.directory}`
      : `no reviews for ${_reviews.date}`;
  } else {
    if (!_reviews.selectedVariant ||
        !(data.variants || []).includes(_reviews.selectedVariant)) {
      _reviews.selectedVariant = (data.variants || [])[0] || null;
    }
    meta.textContent = data.exists
      ? `weekly · ${data.iso_week} · ${(data.variants || []).length} variants`
      : `no reviews for ${data.iso_week}`;
  }
  renderReviewsSidebar();
  // Auto-select first file
  if (_reviews.selectedVariant) {
    const first = _firstReviewFile();
    if (first) selectReviewFile(first);
    else clearReviewContent();
  } else {
    clearReviewContent();
  }
}

function _firstReviewFile() {
  if (_reviews.mode === 'daily') {
    const slot = (_reviews.payload?.by_variant || {})[_reviews.selectedVariant];
    if (!slot) return null;
    return slot.summary || slot.closed[0] || slot.held[0] || null;
  }
  return _reviews.selectedVariant;
}

function renderReviewsSidebar() {
  const vWrap = document.getElementById('reviews-variants');
  const fWrap = document.getElementById('reviews-files');
  vWrap.innerHTML = ''; fWrap.innerHTML = '';

  let variants = [];
  if (_reviews.mode === 'daily') {
    variants = Object.keys(_reviews.payload?.by_variant || {});
  } else {
    variants = (_reviews.payload?.variants || []);
  }
  for (const v of variants) {
    const btn = document.createElement('button');
    btn.className = 'reviews-variant-btn' + (v === _reviews.selectedVariant ? ' active' : '');
    btn.textContent = v;
    btn.addEventListener('click', () => {
      _reviews.selectedVariant = v;
      renderReviewsSidebar();
      const first = _firstReviewFile();
      if (first) selectReviewFile(first);
    });
    vWrap.appendChild(btn);
  }

  // Files list
  const ul = document.createElement('ul');
  if (_reviews.mode === 'daily') {
    const slot = (_reviews.payload?.by_variant || {})[_reviews.selectedVariant];
    if (slot) {
      if (slot.summary) {
        ul.appendChild(_fileLi(slot.summary, 'SUMMARY', 'summary'));
      }
      if (slot.closed.length) {
        const sec = document.createElement('li');
        sec.className = 'rfl-section'; sec.textContent = `Closed · ${slot.closed.length}`;
        ul.appendChild(sec);
        slot.closed.forEach(name => ul.appendChild(_fileLi(name, '', 'closed')));
      }
      if (slot.held.length) {
        const sec = document.createElement('li');
        sec.className = 'rfl-section'; sec.textContent = `Held · ${slot.held.length}`;
        ul.appendChild(sec);
        slot.held.forEach(name => ul.appendChild(_fileLi(name, 'HELD', 'held')));
      }
    }
  } else {
    // Weekly: each variant is its own md file
    (_reviews.payload?.variants || []).forEach(v => ul.appendChild(_fileLi(v, '', 'weekly')));
  }
  fWrap.appendChild(ul);
}

function _fileLi(name, tag, kind) {
  const li = document.createElement('li');
  if (name === _reviews.selectedFile) li.classList.add('selected');
  li.dataset.kind = kind;
  const display = _reviews.mode === 'daily'
    ? name.replace(`${_reviews.selectedVariant}_`, '').replace(/_HELD$/, '')
    : name;
  li.innerHTML = `<span>${escapeHtml(display)}</span>` +
    (tag ? ` <span class="rfl-tag ${kind}">${tag}</span>` : '');
  li.addEventListener('click', () => selectReviewFile(name));
  return li;
}

async function selectReviewFile(name) {
  _reviews.selectedFile = name;
  document.querySelectorAll('#reviews-files li').forEach(el => {
    el.classList.toggle('selected', el.querySelector('span')?.textContent ===
      (name.replace(`${_reviews.selectedVariant}_`, '').replace(/_HELD$/, '')));
  });
  const path = _reviews.mode === 'daily'
    ? `/api/reviews/daily/${_reviews.date}/file/${encodeURIComponent(name)}?include_dry_run=${_reviews.includeDryRun}`
    : `/api/reviews/weekly/${_reviews.date}/file/${encodeURIComponent(name)}?include_dry_run=${_reviews.includeDryRun}`;
  try {
    const res = await fetch(path);
    if (!res.ok) { clearReviewContent(); return; }
    const data = await res.json();
    renderReviewContent(data);
  } catch (e) {
    clearReviewContent();
    document.getElementById('reviews-md').innerHTML =
      `<div class="tb-empty">${escapeHtml(String(e))}</div>`;
  }
}

function renderReviewContent(data) {
  const md = data.content || '';
  document.getElementById('reviews-md').innerHTML = renderMarkdown(md);
  document.getElementById('reviews-summary').innerHTML = '';

  // Embedded chart — prefer the lightweight-charts payload (same
  // pipeline as the TRADE tab); fall back to the legacy Plotly iframe
  // only if the backend couldn't resolve the trade.
  if (data.chart_payload) {
    renderReviewLightweightChart(data.chart_payload);
  } else if (data.chart_html) {
    _teardownReviewLightweightChart();
    const chartHost = document.getElementById('reviews-chart');
    chartHost.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.style.width = '100%';
    iframe.style.height = '600px';
    iframe.style.border = '0';
    iframe.style.background = '#0a0a0d';
    iframe.srcdoc = data.chart_html;
    chartHost.appendChild(iframe);
  } else {
    _teardownReviewLightweightChart();
    document.getElementById('reviews-chart').innerHTML = '';
  }
}

// Reviews-tab chart instance — kept separate from the TRADE tab's
// state.chart so navigating between routes doesn't clobber either.
const _reviewsChart = {
  chart: null,
  candleSeries: null,
  overlaySeries: [],
  fillPriceLines: [],
};

function _teardownReviewLightweightChart() {
  if (!_reviewsChart.chart) return;
  for (const pl of _reviewsChart.fillPriceLines) {
    try { _reviewsChart.candleSeries.removePriceLine(pl); } catch {}
  }
  _reviewsChart.fillPriceLines = [];
  for (const s of _reviewsChart.overlaySeries) {
    try { _reviewsChart.chart.removeSeries(s); } catch {}
  }
  _reviewsChart.overlaySeries = [];
  try { _reviewsChart.chart.remove(); } catch {}
  _reviewsChart.chart = null;
  _reviewsChart.candleSeries = null;
}

function renderReviewLightweightChart(payload) {
  const host = document.getElementById('reviews-chart');
  // Lazy-create on first render. We tear down + re-create on each file
  // selection so the price-scale auto-fit picks up the new symbol's
  // range cleanly (same reasoning as the trade-page autoScale toggle).
  _teardownReviewLightweightChart();
  host.innerHTML =
    '<div id="reviews-chart-canvas" class="reviews-chart-canvas"></div>';
  const el = document.getElementById('reviews-chart-canvas');
  _reviewsChart.chart = LightweightCharts.createChart(el, {
    layout: {
      background: { color: '#0a0e14' }, textColor: '#9ca3af',
      fontFamily: 'IBM Plex Mono, Menlo, monospace',
    },
    grid: { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1f2937' },
    timeScale: { borderColor: '#1f2937', timeVisible: true, secondsVisible: false },
    autoSize: true,
  });
  _reviewsChart.candleSeries = _reviewsChart.chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  // Bars
  _reviewsChart.candleSeries.setData((payload.bars || []).map(b => ({
    time: b.time, open: b.open, high: b.high, low: b.low, close: b.close,
  })));
  // Overlays — reuse the same vocabulary as the TRADE chart, but map
  // 'zone' onto a horizontal price-band-style level pair (no DOM-rect
  // fancy stuff — keeps this chart self-contained).
  for (const o of payload.overlays || []) {
    if (o.kind === 'line') {
      const s = _reviewsChart.chart.addLineSeries({
        color: o.color || '#9ca3af',
        lineWidth: o.width || 1,
        lineStyle: o.style === 'dashed' ? 2 : (o.style === 'dotted' ? 3 : 0),
        priceLineVisible: false, lastValueVisible: false,
      });
      s.setData([{ time: o.from_t, value: o.from_p },
                 { time: o.to_t,   value: o.to_p }]);
      _reviewsChart.overlaySeries.push(s);
    } else if (o.kind === 'level') {
      const pl = _reviewsChart.candleSeries.createPriceLine({
        price: o.price, color: o.color || '#9aa5b1',
        lineWidth: 1, lineStyle: o.style === 'dashed' ? 2 : (o.style === 'dotted' ? 3 : 0),
        axisLabelVisible: true, title: o.label || '',
      });
      _reviewsChart.fillPriceLines.push(pl);
    } else if (o.kind === 'ma') {
      const s = _reviewsChart.chart.addLineSeries({
        color: o.color || '#9ca3af', lineWidth: 1,
        priceLineVisible: false, lastValueVisible: false, title: o.label,
      });
      s.setData((o.values || []).map(([t, v]) => ({ time: t, value: v })));
      _reviewsChart.overlaySeries.push(s);
    }
    // 'zone' / 'band' / 'marker' — markers are surfaced via setMarkers below.
  }
  // Markers (BSPs from chan + entry/exit fills)
  const bspMarkers = (payload.overlays || [])
    .filter(o => o.kind === 'marker').map(markerToSeries);
  const fillMarkers = _buildFillMarkers(payload.fills);
  _reviewsChart.candleSeries.setMarkers(
    [...bspMarkers, ...fillMarkers].sort((a, b) => a.time - b.time),
  );
  // Fill price lines (entry + exits) — same look as the TRADE tab
  for (const f of payload.fills || []) {
    const pl = _reviewsChart.candleSeries.createPriceLine({
      price: Number(f.price),
      color: f.side === 'buy' ? '#10b981' : '#ef4444',
      lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true,
      title: f.side === 'buy' ? 'BUY' : 'SELL',
    });
    _reviewsChart.fillPriceLines.push(pl);
  }
  _reviewsChart.chart.timeScale().fitContent();
}

function clearReviewContent() {
  document.getElementById('reviews-md').innerHTML =
    '<div class="tb-empty">— pick a date and a file to read —</div>';
  document.getElementById('reviews-summary').innerHTML = '';
  _teardownReviewLightweightChart();
  document.getElementById('reviews-chart').innerHTML = '';
}

function renderMarkdown(md) {
  if (!md) return '';
  if (typeof marked !== 'undefined') {
    try {
      let html = marked.parse(md, { breaks: true, gfm: true });
      // Promote a "RED-TEAM VERDICT" line to a styled badge for weekly reviews
      html = html.replace(
        /<p><strong>RED-TEAM VERDICT:\s*([^<]+)<\/strong><\/p>/i,
        (_, v) => {
          const lc = v.toLowerCase().trim();
          const cls = lc.startsWith('support') ? 'support'
                    : lc.startsWith('partial') ? 'partial' : 'reject';
          return `<div class="reviews-verdict ${cls}">RED-TEAM VERDICT: ${v}</div>`;
        }
      );
      // Health badge for HELD reviews
      html = html.replace(
        /<h3[^>]*>Health:\s*([A-Z]+)<\/h3>/g,
        (_, h) => `<span class="health-badge ${h.toLowerCase()}">${h}</span>`
      );
      return html;
    } catch (e) { /* fall through to plain */ }
  }
  return `<pre>${escapeHtml(md)}</pre>`;
}

// ── Page: PORTFOLIO RISK ────────────────────────────────────────────
let _riskCostChart = null, _riskCostSeries = null;

async function renderRiskPage() {
  document.getElementById('risk-meta').textContent = 'loading…';
  let corrFull, corrRecent, outcomes, cost, sizing, regime, patterns;
  try {
    [corrFull, corrRecent, outcomes, cost, sizing, regime, patterns] = await Promise.all([
      fetch('/api/risk/correlation?days=180').then(r => r.json()),
      fetch('/api/risk/correlation?days=60').then(r => r.json()),
      fetch('/api/risk/outcomes').then(r => r.json()),
      fetch('/api/risk/llm_cost').then(r => r.json()),
      fetch('/api/risk/sizing?days=60').then(r => r.json()),
      fetch('/api/risk/regime').then(r => r.json()),
      fetch('/api/risk/patterns').then(r => r.json()),
    ]);
  } catch (e) {
    document.getElementById('risk-meta').textContent = String(e);
    return;
  }
  document.getElementById('risk-meta').textContent =
    `${corrFull.variants.length} variants · ${corrFull.n_days} aligned days`;

  // Section 1 — full correlation
  document.getElementById('risk-corr-table').innerHTML = renderCorrelationTable(corrFull);
  document.getElementById('risk-corr-meta').textContent =
    corrFull.variants.length >= 2 ? `${corrFull.n_days}d full history · pairwise` : 'need ≥2 variants';
  renderCorrCallout(corrFull);

  // Section 2 — recent vs full
  document.getElementById('risk-corr-recent-table').innerHTML = renderCorrelationTable(corrRecent);
  renderCorrShift(corrFull, corrRecent);

  // Section 3 — risk-parity sizing
  renderSizingTable(sizing);

  // Section 4 — rolling pair correlation
  setupPairPicker(corrFull.variants);

  // Section 5 — regime Sharpe
  renderRegimeTable(regime);

  // Section 6 — per-pattern attribution
  renderPatternsTable(patterns);

  document.getElementById('risk-outcomes-meta').textContent =
    `${outcomes.outcomes.length} closes · ${outcomes.start} → ${outcomes.end}`;
  document.getElementById('risk-outcomes').innerHTML = renderOutcomesTable(outcomes.outcomes);

  renderRiskCost(cost);

  // AI synthesis button
  const aiBtn = document.getElementById('risk-ai-btn');
  aiBtn.onclick = () => triggerAiSynthesis();
}

function renderCorrCallout(corr) {
  const el = document.getElementById('risk-corr-callout');
  if (!corr.variants || corr.variants.length < 2) { el.hidden = true; return; }
  const pairs = [];
  for (let i = 0; i < corr.variants.length; i++) {
    for (let j = i + 1; j < corr.variants.length; j++) {
      const c = corr.matrix[i][j];
      if (c != null) pairs.push([corr.variants[i], corr.variants[j], c]);
    }
  }
  pairs.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
  const redundant = pairs.filter(p => p[2] > 0.6).slice(0, 5);
  const independent = pairs.filter(p => Math.abs(p[2]) < 0.2).slice(0, 5);
  const hedges = pairs.filter(p => p[2] < -0.3).slice(0, 3);
  const fmt = arr => arr.map(p => `<code>${escapeHtml(p[0])} ↔ ${escapeHtml(p[1])}</code> (${p[2] >= 0 ? '+' : ''}${p[2].toFixed(2)})`).join(', ');
  const lines = [];
  if (redundant.length)   lines.push(`<strong>Redundant pairs</strong> (>0.6 — same trade): ${fmt(redundant)}.`);
  if (independent.length) lines.push(`<strong>Genuinely independent</strong> (|r|<0.2): ${fmt(independent)}.`);
  if (hedges.length)      lines.push(`<strong>Natural hedges</strong> (negative): ${fmt(hedges)}.`);
  if (!lines.length)      lines.push("Most pairs in 0.2–0.6 range — moderate diversification.");
  el.innerHTML = lines.join('<br><br>');
  el.hidden = false;
}

function renderCorrShift(full, recent) {
  const el = document.getElementById('risk-corr-shift');
  if (!recent.variants || recent.variants.length < 2 || !full.variants) { el.hidden = true; return; }
  const fullMap = new Map();
  for (let i = 0; i < full.variants.length; i++) {
    for (let j = i + 1; j < full.variants.length; j++) {
      fullMap.set(`${full.variants[i]}|${full.variants[j]}`, full.matrix[i][j]);
    }
  }
  const shifts = [];
  for (let i = 0; i < recent.variants.length; i++) {
    for (let j = i + 1; j < recent.variants.length; j++) {
      const a = recent.variants[i], b = recent.variants[j];
      const r = recent.matrix[i][j];
      const f = fullMap.get(`${a}|${b}`) ?? fullMap.get(`${b}|${a}`);
      if (r == null || f == null) continue;
      const d = r - f;
      if (Math.abs(d) > 0.2) shifts.push([a, b, f, r, d]);
    }
  }
  if (!shifts.length) {
    el.innerHTML = `<strong>All pairs stable within ±0.2.</strong> No major regime shift; full-history correlations remain a fair guide for current sizing.`;
  } else {
    shifts.sort((a, b) => Math.abs(b[4]) - Math.abs(a[4]));
    el.innerHTML = `<strong>Pairs that shifted >0.2 between full and recent:</strong><br>` +
      shifts.slice(0, 8).map(s =>
        `<code>${escapeHtml(s[0])} ↔ ${escapeHtml(s[1])}</code>: ${s[2].toFixed(2)} → ${s[3].toFixed(2)} (${s[4] >= 0 ? 'stronger' : 'weaker'})`
      ).join('<br>');
  }
  el.hidden = false;
}

function renderSizingTable(sizing) {
  const tbl = document.getElementById('risk-sizing-table');
  const meta = document.getElementById('risk-sizing-meta');
  const callout = document.getElementById('risk-sizing-callout');
  if (!sizing.rows || !sizing.rows.length) {
    tbl.innerHTML = `<div class="tb-empty">— need ≥5 fully-paired days; have ${sizing.n_days || 0} —</div>`;
    callout.hidden = true;
    meta.textContent = '';
    return;
  }
  meta.textContent = `paired ${sizing.n_days}d · annualized σ ≈ ${sizing.portfolio_sigma_annual.toFixed(1)}%`;
  const rows = sizing.rows.map(r => `<tr>
    <td><b>${escapeHtml(r.variant)}</b></td>
    <td class="num">$${Math.round(r.current_dollars).toLocaleString()}</td>
    <td class="num">${(r.current_weight * 100).toFixed(1)}%</td>
    <td class="num">${r.daily_vol_pct.toFixed(2)}%</td>
    <td class="num">${(r.pct_of_portfolio_swings * 100).toFixed(0)}%</td>
    <td class="num">${(r.risk_parity_weight * 100).toFixed(1)}%</td>
    <td class="num ${r.shift_pp >= 0 ? 'pos' : 'neg'}">${r.shift_pp >= 0 ? '+' : ''}${r.shift_pp.toFixed(1)}pp</td>
  </tr>`).join('');
  tbl.innerHTML = `<table class="tb"><thead><tr>
    <th>STRATEGY</th><th class="num">CURRENT $</th><th class="num">WEIGHT</th>
    <th class="num">DAILY VOL</th><th class="num">% SWINGS</th>
    <th class="num">RISK-PARITY</th><th class="num">SHIFT</th>
    </tr></thead><tbody>${rows}</tbody></table>`;

  // Bottom-line callout — top 2 contributors
  const sorted = sizing.rows.slice().sort((a, b) => b.pct_of_portfolio_swings - a.pct_of_portfolio_swings);
  const top2 = sorted.slice(0, 2);
  if (top2.length === 2) {
    const share = (top2[0].pct_of_portfolio_swings + top2[1].pct_of_portfolio_swings) * 100;
    const biggestShifts = sizing.rows.slice().sort((a, b) => Math.abs(b.shift_pp) - Math.abs(a.shift_pp)).slice(0, 3);
    callout.innerHTML = `<strong>Bottom line</strong>: <code>${escapeHtml(top2[0].variant)}</code> (${(top2[0].pct_of_portfolio_swings * 100).toFixed(0)}%) and <code>${escapeHtml(top2[1].variant)}</code> (${(top2[1].pct_of_portfolio_swings * 100).toFixed(0)}%) cause <strong>${share.toFixed(0)}%</strong> of portfolio swings.<br><br>` +
      `Annualized σ ≈ ${sizing.portfolio_sigma_annual.toFixed(1)}% (SPY runs 15-20%).<br><br>` +
      `Biggest risk-parity shifts: ${biggestShifts.map(s => `<code>${escapeHtml(s.variant)}</code> ${s.shift_pp >= 0 ? '+' : ''}${s.shift_pp.toFixed(1)}pp`).join(', ')}.`;
    callout.hidden = false;
  } else {
    callout.hidden = true;
  }
}

let _riskPairChart = null, _riskPairSeries = null;
function setupPairPicker(variants) {
  if (!variants || variants.length < 2) return;
  const aSel = document.getElementById('risk-pair-a');
  const bSel = document.getElementById('risk-pair-b');
  const opts = variants.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  aSel.innerHTML = opts; bSel.innerHTML = opts;
  aSel.value = variants[0]; bSel.value = variants[1];
  const onChange = () => loadRollingCorr(aSel.value, bSel.value);
  aSel.onchange = onChange; bSel.onchange = onChange;
  loadRollingCorr(aSel.value, bSel.value);
}

async function loadRollingCorr(a, b) {
  const callout = document.getElementById('risk-pair-callout');
  if (a === b) {
    callout.innerHTML = '<em>Pick two different variants.</em>'; callout.hidden = false;
    return;
  }
  let data;
  try {
    data = await fetch(`/api/risk/rolling_corr?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&window=20`).then(r => r.json());
  } catch (e) { callout.innerHTML = String(e); callout.hidden = false; return; }

  if (!_riskPairChart) {
    _riskPairChart = LightweightCharts.createChart(document.getElementById('risk-pair-chart'), {
      layout: { background: { color: '#111114' }, textColor: '#9aa5b1', fontFamily: 'JetBrains Mono, monospace' },
      grid:   { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: { borderColor: '#1f2937', timeVisible: false },
      autoSize: true,
    });
  }
  if (_riskPairSeries) { try { _riskPairChart.removeSeries(_riskPairSeries); } catch {} }
  _riskPairSeries = _riskPairChart.addLineSeries({
    color: '#0097a7', lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
  });
  const points = (data.points || [])
    .map(p => ({ time: dateToUnix(p.date), value: p.corr }))
    .filter(p => !isNaN(p.value) && p.value != null);
  _riskPairSeries.setData(points);
  _riskPairChart.timeScale().fitContent();

  if (!points.length) {
    callout.innerHTML = data.note ? escapeHtml(data.note) : 'no overlap';
    callout.hidden = false;
    return;
  }
  const cur = points[points.length - 1].value;
  const vals = points.map(p => p.value);
  const min = Math.min(...vals), max = Math.max(...vals);
  let line = `<strong>Now</strong>: <code>${escapeHtml(a)}</code> ↔ <code>${escapeHtml(b)}</code> = ${cur >= 0 ? '+' : ''}${cur.toFixed(2)} `;
  line += Math.abs(cur) > 0.6 ? '(<strong>redundant</strong>)' : Math.abs(cur) > 0.3 ? '(moderately related)' : '(mostly independent)';
  line += `.<br><br>`;
  if (max - min > 0.4) {
    line += `<strong>Historically unstable</strong>: range ${min.toFixed(2)} to ${max.toFixed(2)}. Size assuming the high-correlation case.`;
  } else {
    line += `<strong>Historically stable</strong>: range ${min.toFixed(2)} to ${max.toFixed(2)}. Current number is reliable.`;
  }
  callout.innerHTML = line;
  callout.hidden = false;
}

function renderRegimeTable(regime) {
  const tbl = document.getElementById('risk-regime-table');
  if (!regime.available || !regime.rows || !regime.rows.length) {
    tbl.innerHTML = '<div class="tb-empty">— SPY history unavailable —</div>';
    return;
  }
  // Pivot rows: variant × regime → (sharpe, n_days)
  const variants = Array.from(new Set(regime.rows.map(r => r.variant)));
  const regimes = ["confirmed_uptrend", "uptrend_under_pressure", "market_correction"];
  const map = new Map();
  for (const r of regime.rows) map.set(`${r.variant}|${r.regime}`, r);
  const head = '<th></th>' + regimes.map(rg => `<th>${escapeHtml(rg.replace(/_/g, ' '))}</th>`).join('');
  const trs = variants.map(v => {
    const cells = regimes.map(rg => {
      const r = map.get(`${v}|${rg}`);
      if (!r || r.sharpe == null) return '<td class="cell-na">—</td>';
      const cls = r.sharpe >= 0.5 ? 'cell-pos-strong' : r.sharpe > 0 ? 'cell-pos' : r.sharpe < -0.5 ? 'cell-neg-strong' : 'cell-neg';
      return `<td class="${cls}">${r.sharpe >= 0 ? '+' : ''}${r.sharpe.toFixed(1)}<br><span style="font-size:0.62rem;color:#64748b;">over ${r.n_days}d</span></td>`;
    }).join('');
    return `<tr><th>${escapeHtml(v)}</th>${cells}</tr>`;
  }).join('');
  tbl.innerHTML = `<table class="corr-table"><thead><tr>${head}</tr></thead><tbody>${trs}</tbody></table>`;
}

function renderPatternsTable(patterns) {
  const tbl = document.getElementById('risk-patterns-table');
  if (!patterns.rows || !patterns.rows.length) {
    tbl.innerHTML = '<div class="tb-empty">— no closed-trade outcomes yet —</div>';
    return;
  }
  // Sort by variant then by total_return desc within variant
  const rows = patterns.rows.slice().sort((a, b) => {
    if (a.variant !== b.variant) return a.variant.localeCompare(b.variant);
    return (b.total_return_pct || 0) - (a.total_return_pct || 0);
  });
  const trs = rows.map(r => `<tr>
    <td>${escapeHtml(r.variant)}</td>
    <td>${escapeHtml(r.base_pattern)}</td>
    <td class="num">${r.n_trades}</td>
    <td class="num">${(r.win_rate * 100).toFixed(1)}%</td>
    <td class="num ${r.avg_return_pct >= 0 ? 'pos' : 'neg'}">${(r.avg_return_pct * 100).toFixed(2)}%</td>
    <td class="num">${(r.median_return_pct * 100).toFixed(2)}%</td>
    <td class="num ${r.total_return_pct >= 0 ? 'pos' : 'neg'}">${(r.total_return_pct * 100).toFixed(2)}%</td>
    <td class="num">${r.avg_hold_days != null ? r.avg_hold_days.toFixed(1) : '—'}</td>
  </tr>`).join('');
  tbl.innerHTML = `<table class="tb"><thead><tr>
    <th>STRATEGY</th><th>PATTERN</th><th class="num">TRADES</th><th class="num">WIN%</th>
    <th class="num">AVG %</th><th class="num">MEDIAN %</th><th class="num">TOTAL %</th><th class="num">HELD</th>
    </tr></thead><tbody>${trs}</tbody></table>`;
}

async function triggerAiSynthesis() {
  const btn = document.getElementById('risk-ai-btn');
  const out = document.getElementById('risk-ai-output');
  btn.disabled = true;
  out.hidden = false;
  out.innerHTML = '<em>gpt-5.4-pro reading the data…</em>';
  try {
    const res = await fetch('/api/risk/synthesis', { method: 'POST' });
    if (!res.ok) {
      const err = await res.text();
      out.innerHTML = `<span class="neg">synthesis failed: ${escapeHtml(err)}</span>`;
      return;
    }
    const data = await res.json();
    const meta = `<small style="color:#64748b;">${escapeHtml(data.model)} · ${data.duration_s}s · ~$${data.cost_estimate_usd.toFixed(2)} · ${data.cached ? 'cached' : 'fresh'}</small><br><br>`;
    // Render markdown if marked is loaded; else escape + linebreaks
    let body;
    if (typeof marked !== 'undefined') {
      try { body = marked.parse(data.text || '', { breaks: true, gfm: true }); }
      catch { body = escapeHtml(data.text || '').replace(/\n/g, '<br>'); }
    } else {
      body = escapeHtml(data.text || '').replace(/\n/g, '<br>');
    }
    out.innerHTML = meta + body;
  } catch (e) {
    out.innerHTML = `<span class="neg">${escapeHtml(String(e))}</span>`;
  } finally {
    btn.disabled = false;
  }
}

function renderCorrelationTable(corr) {
  if (!corr.variants || corr.variants.length < 2) {
    return '<div class="tb-empty">— need at least 2 variants with ≥5 days of overlap —</div>';
  }
  const head = '<th></th>' + corr.variants.map(v => `<th>${escapeHtml(v)}</th>`).join('');
  const rows = corr.variants.map((v, i) => {
    const cells = corr.variants.map((w, j) => {
      if (i === j) return '<td class="diag">1.00</td>';
      const c = corr.matrix[i][j];
      if (c == null) return '<td class="cell-na">—</td>';
      const cls = c >= 0.7 ? 'cell-pos-strong'
                : c >= 0.3 ? 'cell-pos'
                : c <= -0.3 ? 'cell-neg-strong'
                : c < 0      ? 'cell-neg' : '';
      return `<td class="${cls}">${c.toFixed(2)}</td>`;
    }).join('');
    return `<tr><th>${escapeHtml(v)}</th>${cells}</tr>`;
  }).join('');
  return `<table class="corr-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderOutcomesTable(rows) {
  if (!rows || !rows.length) return '<div class="tb-empty">— no closed trades in window —</div>';
  // Most recent first
  rows = rows.slice().sort((a, b) => (b.exit_date || '').localeCompare(a.exit_date || ''));
  const trs = rows.slice(0, 200).map(o => {
    const r = Number(o.return_pct || 0);
    const cls = r >= 0 ? 'pos' : 'neg';
    const mfe = Number(o.max_favorable_excursion || 0);
    const cap = (mfe > 0 && r != null) ? (r / mfe) : null;
    return `<tr>
      <td>${escapeHtml(o.exit_date || '')}</td>
      <td><span class="sym">${escapeHtml(o.symbol || '')}</span></td>
      <td style="font-size:0.7rem; color:var(--text-muted);">${escapeHtml(o._variant)}</td>
      <td class="num ${cls}">${(r * 100).toFixed(2)}%</td>
      <td class="num">${mfe ? (mfe * 100).toFixed(2) + '%' : '—'}</td>
      <td class="num">${cap != null ? cap.toFixed(2) : '—'}</td>
      <td>${escapeHtml(o.exit_reason || o.reason || '')}</td>
    </tr>`;
  }).join('');
  return `<table class="tb"><thead><tr>
    <th>EXIT DATE</th><th>SYMBOL</th><th>VARIANT</th>
    <th class="num">RETURN</th><th class="num">MFE</th>
    <th class="num">CAPTURE</th><th>EXIT REASON</th>
    </tr></thead><tbody>${trs}</tbody></table>`;
}

function renderRiskCost(cost) {
  const meta = document.getElementById('risk-cost-meta');
  if (!cost.available || !cost.rows.length) {
    meta.textContent = cost.available ? 'no analyses in window' : 'no LLM cost data';
    document.getElementById('risk-cost-chart').innerHTML =
      '<div class="tb-empty" style="height:100%; display:flex; align-items:center; justify-content:center;">— LLM cost source unavailable —</div>';
    document.getElementById('risk-cost-table').innerHTML = '';
    return;
  }
  const rows = cost.rows;
  const total = rows.reduce((s, r) => s + (r.daily_cost || 0), 0);
  const errs  = rows.reduce((s, r) => s + (r.n_errors || 0), 0);
  meta.textContent = `${rows.length} days · total $${total.toFixed(2)} · ${errs} errors`;

  if (!_riskCostChart) {
    _riskCostChart = LightweightCharts.createChart(
      document.getElementById('risk-cost-chart'), {
        layout: { background: { color: '#111114' }, textColor: '#9aa5b1', fontFamily: 'JetBrains Mono, monospace' },
        grid:   { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
        rightPriceScale: { borderColor: '#1f2937' },
        timeScale: { borderColor: '#1f2937', timeVisible: false },
        autoSize: true,
      });
    _riskCostSeries = _riskCostChart.addHistogramSeries({
      color: '#f5a524',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      priceLineVisible: false, lastValueVisible: false,
    });
  }
  _riskCostSeries.setData(rows.map(r => ({
    time: dateToUnix(r.day),
    value: Number(r.daily_cost || 0),
    color: r.n_errors > 0 ? 'rgba(232,90,90,0.6)' : 'rgba(245,165,36,0.6)',
  })));
  _riskCostChart.timeScale().fitContent();

  // Cost table
  const trs = rows.slice().reverse().map(r => `<tr>
    <td>${escapeHtml(r.day)}</td>
    <td class="num">$${Number(r.daily_cost || 0).toFixed(2)}</td>
    <td class="num">${r.n_analyses || 0}</td>
    <td class="num ${(r.n_errors || 0) > 0 ? 'neg' : ''}">${r.n_errors || 0}</td>
  </tr>`).join('');
  document.getElementById('risk-cost-table').innerHTML = `<table class="tb"><thead><tr>
    <th>DAY</th><th class="num">COST</th><th class="num">ANALYSES</th><th class="num">ERRORS</th>
    </tr></thead><tbody>${trs}</tbody></table>`;
}

// ── Page: PROPOSALS ─────────────────────────────────────────────────
const _props = {
  inited: false,
  rows: [],
  active: new Set(['open', 'accepted']),  // status filter
};

async function renderProposalsPage() {
  if (!_props.inited) {
    wireProposalsControls();
    _props.inited = true;
  }
  await loadProposals();
}

function wireProposalsControls() {
  document.querySelectorAll('#prop-status .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      const s = b.dataset.status;
      if (_props.active.has(s)) _props.active.delete(s);
      else _props.active.add(s);
      b.classList.toggle('active');
      renderProposalsList();
    });
  });
}

async function loadProposals() {
  const meta = document.getElementById('prop-meta');
  meta.textContent = 'loading…';
  try {
    const data = await fetch('/api/proposals').then(r => r.json());
    _props.rows = data.proposals || [];
  } catch (e) { meta.textContent = String(e); return; }
  meta.textContent = `${_props.rows.length} total`;
  renderProposalKpi();
  renderProposalsList();
}

function renderProposalKpi() {
  const counts = { open: 0, accepted: 0, rejected: 0, tested: 0 };
  for (const r of _props.rows) counts[r.status] = (counts[r.status] || 0) + 1;
  document.getElementById('prop-kpi').innerHTML =
    kpiCard('OPEN',     String(counts.open || 0)) +
    kpiCard('ACCEPTED', String(counts.accepted || 0), '', 'pos') +
    kpiCard('TESTED',   String(counts.tested || 0)) +
    kpiCard('REJECTED', String(counts.rejected || 0), '', 'neg') +
    kpiCard('TOTAL',    String(_props.rows.length));
}

function renderProposalsList() {
  const list = document.getElementById('prop-list');
  list.innerHTML = '';
  const filtered = _props.rows.filter(r => _props.active.has(r.status));
  if (!filtered.length) {
    list.innerHTML = '<div class="tb-empty">— no proposals match the current filter —</div>';
    return;
  }
  // Sort newest first
  filtered.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
  for (const p of filtered) list.appendChild(renderProposalCard(p));
}

function renderProposalCard(p) {
  const card = document.createElement('div');
  card.className = 'proposal-card';
  card.dataset.status = p.status;

  const head = document.createElement('div');
  head.className = 'proposal-head';
  head.innerHTML = `
    <span class="proposal-title">${escapeHtml(p.title || 'Untitled')}</span>
    <span class="proposal-tag ${p.status}">${(p.status || 'open').toUpperCase()}</span>
    <span class="proposal-meta">[${escapeHtml(p._db || p.variant || '?')}]
      · ${escapeHtml(p.iso_week || '')}
      · ${escapeHtml((p.created_at || '').slice(0, 10))}</span>
  `;
  head.addEventListener('click', () => card.classList.toggle('open-card'));
  card.appendChild(head);

  const body = document.createElement('div');
  body.className = 'proposal-body';
  for (const [label, key] of [['What','what'],['Why','why'],
                              ['How to validate','how_to_validate'],['Risk','risk'],
                              ['Outcome','outcome_summary']]) {
    if (p[key]) {
      const sec = document.createElement('div');
      sec.className = 'proposal-section';
      sec.innerHTML = `<h4>${escapeHtml(label)}</h4><p>${escapeHtml(p[key]).replace(/\n/g, '<br>')}</p>`;
      body.appendChild(sec);
    }
  }

  const actions = document.createElement('div');
  actions.className = 'proposal-actions';
  actions.innerHTML = `
    <select class="status-pick">
      ${['open','accepted','rejected','tested'].map(s =>
        `<option value="${s}" ${s === p.status ? 'selected' : ''}>${s.toUpperCase()}</option>`).join('')}
    </select>
    <textarea class="outcome-text" placeholder="outcome (saved only on TESTED)">${escapeHtml(p.outcome_summary || '')}</textarea>
    <button class="save-btn">SAVE</button>
    <span class="proposal-toast"></span>
  `;
  const sel = actions.querySelector('.status-pick');
  const txt = actions.querySelector('.outcome-text');
  const btn = actions.querySelector('.save-btn');
  const toast = actions.querySelector('.proposal-toast');
  btn.addEventListener('click', async () => {
    btn.disabled = true; toast.textContent = 'saving…';
    try {
      const res = await fetch(`/api/proposals/${encodeURIComponent(p._db)}/${p.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: sel.value,
          outcome_summary: sel.value === 'tested' ? txt.value : null,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      toast.textContent = `→ ${sel.value}`;
      // Update local row + re-render
      p.status = sel.value;
      if (sel.value === 'tested') p.outcome_summary = txt.value;
      renderProposalKpi(); renderProposalsList();
    } catch (e) {
      toast.textContent = `error: ${e}`;
      toast.style.color = 'var(--coral)';
    } finally {
      btn.disabled = false;
    }
  });
  body.appendChild(actions);
  card.appendChild(body);
  return card;
}

// ── Page: IDEAS ─────────────────────────────────────────────────────
const _ideas = {
  inited: false,
  mode: 'digest',
  digests: [],
  selectedDigest: null,
};

async function renderIdeasPage() {
  if (!_ideas.inited) {
    wireIdeasControls();
    _ideas.inited = true;
  }
  if (_ideas.mode === 'digest') {
    await loadDigests();
  } else {
    await loadCorpusMeta();
  }
}

function wireIdeasControls() {
  document.querySelectorAll('#ideas-mode .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#ideas-mode .seg-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      _ideas.mode = b.dataset.mode;
      document.getElementById('ideas-digest').hidden = (_ideas.mode !== 'digest');
      document.getElementById('ideas-corpus').hidden = (_ideas.mode !== 'corpus');
      renderIdeasPage();
    });
  });
  document.getElementById('corpus-search-btn').addEventListener('click', searchCorpus);
  document.getElementById('corpus-q').addEventListener('keydown', e => {
    if (e.key === 'Enter') searchCorpus();
  });
}

async function loadDigests() {
  let data;
  try { data = await fetch('/api/ideas/digests').then(r => r.json()); }
  catch (e) {
    document.getElementById('ideas-digest-content').innerHTML =
      `<div class="tb-empty">${escapeHtml(String(e))}</div>`;
    return;
  }
  _ideas.digests = data.digests || [];
  const ul = document.getElementById('ideas-digest-list');
  ul.innerHTML = '';
  if (!_ideas.digests.length) {
    document.getElementById('ideas-digest-content').innerHTML =
      `<div class="tb-empty">— no digests yet · runs Saturday 10:00 CDT —</div>`;
    return;
  }
  for (const d of _ideas.digests) {
    const li = document.createElement('li');
    li.textContent = d.name;
    if (d.name === _ideas.selectedDigest) li.classList.add('selected');
    li.addEventListener('click', () => loadDigest(d.name));
    ul.appendChild(li);
  }
  if (!_ideas.selectedDigest) loadDigest(_ideas.digests[0].name);
}

async function loadDigest(name) {
  _ideas.selectedDigest = name;
  document.querySelectorAll('#ideas-digest-list li').forEach(el => {
    el.classList.toggle('selected', el.textContent === name);
  });
  try {
    const data = await fetch(`/api/ideas/digest/${encodeURIComponent(name)}`).then(r => r.json());
    document.getElementById('ideas-digest-content').innerHTML = renderMarkdown(data.content || '');
  } catch (e) {
    document.getElementById('ideas-digest-content').innerHTML =
      `<div class="tb-empty">${escapeHtml(String(e))}</div>`;
  }
}

async function loadCorpusMeta() {
  let m;
  try { m = await fetch('/api/ideas/corpus/meta').then(r => r.json()); }
  catch (e) {
    document.getElementById('corpus-meta').textContent = String(e); return;
  }
  if (!m.available) {
    document.getElementById('corpus-meta').textContent =
      'CORPUS UNAVAILABLE · run telegram_listener (see /ideas → corpus setup)';
    document.getElementById('corpus-results').innerHTML = '';
    return;
  }
  const top = (m.chats || []).slice(0, 5)
    .map(c => `${c.chat_title || c.chat_id}=${c.n.toLocaleString()}`)
    .join(' · ');
  document.getElementById('corpus-meta').textContent =
    `CORPUS · ${m.n_messages.toLocaleString()} msgs · ${(m.min_ts || '').slice(0,10)} → ${(m.max_ts || '').slice(0,10)} · ${top}`;
  searchCorpus();  // initial search with defaults
}

async function searchCorpus() {
  const q       = document.getElementById('corpus-q').value;
  const author  = document.getElementById('corpus-author').value;
  const days    = document.getElementById('corpus-days').value || 30;
  const limit   = document.getElementById('corpus-limit').value || 500;
  const params = new URLSearchParams({ q, author, days, limit });
  try {
    const data = await fetch(`/api/ideas/corpus/search?${params}`).then(r => r.json());
    const msgs = data.messages || [];
    if (!msgs.length) {
      document.getElementById('corpus-results').innerHTML =
        '<div class="tb-empty">— no matches —</div>';
      return;
    }
    const trs = msgs.map(m => {
      const ts = (m.timestamp || '').slice(0, 16).replace('T', ' ');
      const author = (m.author_username || '') + (m.author_display ? ` (${m.author_display})` : '');
      return `<tr>
        <td>${escapeHtml(ts)}</td>
        <td style="color:var(--text-muted);">${escapeHtml(m.chat_title || '')}</td>
        <td style="color:var(--text-muted);">${escapeHtml(author)}</td>
        <td>${escapeHtml(m.text || '')}</td>
      </tr>`;
    }).join('');
    document.getElementById('corpus-results').innerHTML =
      `<div style="font-size:0.62rem; color:var(--text-faint); letter-spacing:0.18em; margin-bottom:0.4rem;">${msgs.length} MATCHES</div>
       <table class="tb"><thead><tr>
         <th>WHEN</th><th>CHAT</th><th>AUTHOR</th><th>TEXT</th>
       </tr></thead><tbody>${trs}</tbody></table>`;
  } catch (e) {
    document.getElementById('corpus-results').innerHTML =
      `<div class="tb-empty">${escapeHtml(String(e))}</div>`;
  }
}

// ── Helpers shared across pages ─────────────────────────────────────
function dateToUnix(s) {
  // Accept "YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS", ISO with tz.
  const d = new Date(String(s).replace(' ', 'T'));
  return Math.floor(d.getTime() / 1000);
}
function fmtUSD(n) {
  return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtUSDSigned(n) {
  const v = Number(n);
  return (v >= 0 ? '+$' : '-$') + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Live UX: FOLLOW, AWAKE, LAST-ACTIVITY ────────────────────────────
function wireLiveControls() {
  document.getElementById('btn-awake').addEventListener('click', toggleWakeLock);
  document.getElementById('btn-clear-incoming').addEventListener('click', clearIncoming);
  // Initial empty-state render so the "— quiet —" placeholder shows
  // immediately on page load.
  renderIncomingList();

  // Re-acquire wake lock when the page becomes visible again. iPad
  // releases the lock automatically when the tab is backgrounded.
  document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === 'visible' && state.wakeLockEnabled && !state.wakeLock) {
      await acquireWakeLock();
    }
  });
}

async function acquireWakeLock() {
  if (!('wakeLock' in navigator)) return false;
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    state.wakeLock.addEventListener('release', () => { state.wakeLock = null; });
    return true;
  } catch (e) {
    console.warn('wakeLock denied:', e);
    return false;
  }
}

async function toggleWakeLock() {
  const btn = document.getElementById('btn-awake');
  if (state.wakeLock) {
    try { await state.wakeLock.release(); } catch {}
    state.wakeLock = null;
    state.wakeLockEnabled = false;
    btn.classList.remove('active');
    return;
  }
  if (!('wakeLock' in navigator)) {
    btn.disabled = true;
    btn.title = 'Wake Lock API unavailable in this browser';
    return;
  }
  const ok = await acquireWakeLock();
  if (ok) {
    state.wakeLockEnabled = true;
    btn.classList.add('active');
  }
}

function startLastActivityTicker() {
  // Updates the topbar "LAST · N MIN AGO" readout once a second.
  const valueEl = document.getElementById('last-activity-value');
  const wrap    = document.getElementById('last-activity');
  const tick = () => {
    if (state.lastActivityMs == null) {
      valueEl.textContent = '—';
      wrap.dataset.fresh = 'cold';
      return;
    }
    const ageS = Math.max(0, Math.floor((Date.now() - state.lastActivityMs) / 1000));
    valueEl.textContent = formatAge(ageS);
    wrap.dataset.fresh = ageS < 120 ? 'hot' : ageS < 1800 ? 'warm' : 'cold';
  };
  tick();
  state.lastActivityTimer = setInterval(tick, 1000);
}

function formatAge(s) {
  if (s < 60)    return `${s}s ago`;
  if (s < 3600)  return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// ── Variants ─────────────────────────────────────────────────────────
async function loadVariants() {
  const res = await fetch('/api/variants');
  state.variants = await res.json();
}

function populateVariantSelect() {
  const sel = document.getElementById('variant-select');
  sel.innerHTML = '';
  state.variants.forEach(v => {
    const o = document.createElement('option');
    o.value = v.name;
    o.textContent = `${v.name} · ${v.strategy_type}`;
    sel.appendChild(o);
  });
  sel.value = state.selectedVariant;
  sel.addEventListener('change', async e => {
    state.selectedVariant = e.target.value;
    await loadTrades(state.selectedVariant);
    // Switch the variant's per-account TODAY/PERFORMANCE/REVIEWS data too.
    // loadTrades only refreshes the TRADE page's left rail + chart; non-trade
    // routes need their onActivate re-fired to pull the new variant's data.
    if (_activeRoute && _activeRoute !== 'trade') {
      try { ROUTES[_activeRoute].onActivate?.(); } catch (e) { console.error(e); }
    }
  });
}

// ── Trades list ──────────────────────────────────────────────────────
async function loadTrades(variant) {
  const res = await fetch(`/api/variants/${encodeURIComponent(variant)}/trades?limit=80`);
  state.trades = await res.json();
  // Switching variants is a clean slate — incoming is per-session per-variant.
  state.incomingTrades = [];
  renderIncomingList();
  renderTradesList();
  if (state.trades.length > 0) {
    // Initialize last-activity from the most recent trade.
    const ts = state.trades[0].timestamp;
    state.lastActivityMs = ts ? new Date(ts.replace(' ', 'T')).getTime() : null;
    selectTrade(state.trades[0].trade_id);
  } else {
    clearChart();
    document.getElementById('chart-title').textContent = '— no trades for this variant —';
    state.lastActivityMs = null;
  }
  // Re-subscribe the trade stream for the new variant.
  openVariantStream(variant);
}

// ── Live trade stream (per variant) ──────────────────────────────────
function openVariantStream(variant) {
  if (state.variantStream) {
    try { state.variantStream.close(); } catch {}
    state.variantStream = null;
  }
  const url = `/api/variants/${encodeURIComponent(variant)}/stream`;
  const ev = new EventSource(url);
  state.variantStream = ev;
  ev.onmessage = e => {
    let rec;
    try { rec = JSON.parse(e.data); } catch { return; }
    if (rec.type === 'trade')        handleNewTrade(rec.trade);
    else if (rec.type === 'snapshot') { /* established baseline; ignore */ }
    else if (rec.type === 'error')    console.warn('variant stream error:', rec.message);
  };
  ev.onerror = () => {
    // EventSource auto-reconnects with exponential backoff. We don't
    // explicitly handle errors here — the connection will resume when
    // the server is reachable again.
  };
}

function handleNewTrade(trade) {
  // Skip duplicates if we already have this trade_id (shouldn't happen
  // post-snapshot, but be safe).
  if (state.trades.some(t => t.trade_id === trade.trade_id)) return;

  state.trades.unshift(trade);
  // Cap to 80 in memory like the initial load
  if (state.trades.length > 80) state.trades.pop();

  // Push into the explicit INCOMING queue. The user clicks to navigate;
  // we never yank the chart away from what they're reviewing.
  if (!state.incomingTrades.some(t => t.trade_id === trade.trade_id)) {
    state.incomingTrades.unshift(trade);
    if (state.incomingTrades.length > 25) state.incomingTrades.pop();
  }

  state.lastActivityMs = Date.now();
  renderTradesList({ markNew: trade.trade_id });
  renderIncomingList({ markNew: trade.trade_id });
}

// ── INCOMING list ────────────────────────────────────────────────────
function renderIncomingList(opts = {}) {
  const ul     = document.getElementById('incoming-list');
  const empty  = document.getElementById('incoming-empty');
  const count  = document.getElementById('incoming-count');
  const clear  = document.getElementById('btn-clear-incoming');

  count.textContent = String(state.incomingTrades.length);
  clear.hidden = state.incomingTrades.length === 0;
  empty.style.display = state.incomingTrades.length === 0 ? 'block' : 'none';

  ul.innerHTML = '';
  state.incomingTrades.forEach(t => {
    const li = document.createElement('li');
    li.dataset.tradeId = t.trade_id;
    if (opts.markNew === t.trade_id) {
      li.classList.add('is-new');
      setTimeout(() => li.classList.remove('is-new'), 8500);
    }
    const side = (t.side || '').toLowerCase();
    li.innerHTML = `
      ${opts.markNew === t.trade_id ? '<span class="new-ribbon">NEW</span>' : ''}
      <div class="trade-row1">
        <span class="trade-sym">${t.symbol || '?'}</span>
        <span class="trade-side ${side}">${side.toUpperCase()}</span>
        <span class="trade-time">${formatTradeTime(t.timestamp)}</span>
      </div>
      <div class="trade-meta">
        ${t.filled_price ? `@ ${Number(t.filled_price).toFixed(2)}` : (t.status || '')}
        ${t.signal_metadata && t.signal_metadata.t_types
          ? '· ' + escapeHtml(t.signal_metadata.t_types)
          : ''}
      </div>`;
    li.addEventListener('click', () => {
      // Slide-out animation, then dismiss + navigate.
      li.classList.add('dismissing');
      setTimeout(() => {
        state.incomingTrades = state.incomingTrades.filter(x => x.trade_id !== t.trade_id);
        renderIncomingList();
        selectTrade(t.trade_id);
      }, 280);
    });
    ul.appendChild(li);
  });
}

function clearIncoming() {
  state.incomingTrades = [];
  renderIncomingList();
}

function renderTradesList(opts = {}) {
  const ul = document.getElementById('trades-list');
  ul.innerHTML = '';
  state.trades.forEach(t => {
    const li = document.createElement('li');
    li.dataset.tradeId = t.trade_id;
    if (t.trade_id === state.selectedTradeId) li.classList.add('selected');
    const isNew = opts.markNew === t.trade_id;
    if (isNew) {
      li.classList.add('is-new');
      // Drop the class after the animation so re-renders don't replay it.
      setTimeout(() => li.classList.remove('is-new'), 8500);
    }
    const side = (t.side || '').toLowerCase();
    li.innerHTML = `
      ${isNew ? '<span class="new-ribbon">NEW</span>' : ''}
      <div class="trade-row1">
        <span class="trade-sym">${t.symbol || '?'}</span>
        <span class="trade-side ${side}">${side.toUpperCase()}</span>
        <span class="trade-time">${formatTradeTime(t.timestamp)}</span>
      </div>
      <div class="trade-meta">
        ${t.filled_price ? `@ ${Number(t.filled_price).toFixed(2)}` : t.status}
        · ${t.status || ''}
      </div>`;
    li.addEventListener('click', () => selectTrade(t.trade_id));
    ul.appendChild(li);
  });
}

function formatTradeTime(ts) {
  if (!ts) return '';
  // Try a couple of formats — the trades table mixes "YYYY-MM-DD HH:MM:SS"
  // and ISO8601 with timezone.
  const d = new Date(ts.replace(' ', 'T'));
  if (isNaN(d.getTime())) return ts;
  return d.toISOString().slice(5, 16).replace('T', ' ');
}

// ── Chart rendering ──────────────────────────────────────────────────
async function selectTrade(tradeId) {
  state.selectedTradeId = tradeId;
  document.querySelectorAll('.trades-list li').forEach(el => {
    el.classList.toggle('selected', Number(el.dataset.tradeId) === tradeId);
  });

  // Stop any in-flight live-tail loop from a previous selection.
  stopLiveTail();

  document.getElementById('chart-error').hidden = true;
  document.getElementById('chart-title').textContent = 'loading…';

  let payload;
  try {
    const res = await fetch(`/api/trade/${encodeURIComponent(state.selectedVariant)}/${tradeId}/chart`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || `HTTP ${res.status}`);
      return;
    }
    payload = await res.json();
  } catch (e) {
    showError(String(e));
    return;
  }

  applyChartHeader(payload);
  renderPayload(payload);
  renderReasoning(payload.reasoning);

  // If the position is still open, start the auto-refresh loop so new
  // bars + chan structures appear without user action.
  if (payload.is_open) startLiveTail(tradeId);
}

function applyChartHeader(payload) {
  document.getElementById('chart-title').textContent = `${payload.symbol} · ${payload.variant}`;
  const meta = document.getElementById('chart-meta');
  meta.textContent =
    `${payload.strategy_type} · ${payload.timeframe} · ${payload.bars.length} bars`;
  if (payload.is_open) {
    meta.classList.add('is-open');
    meta.innerHTML = `<span class="open-pip"></span>OPEN · ` + meta.textContent;
  } else {
    meta.classList.remove('is-open');
  }
  if (payload.error) {
    document.getElementById('chart-error').textContent = payload.error;
    document.getElementById('chart-error').hidden = false;
  }
}

// ── LIVE TAIL — auto-refresh while viewing an open position ─────────
function startLiveTail(tradeId) {
  stopLiveTail();
  const meta = document.getElementById('chart-meta');
  meta.classList.add('is-tailing');
  state.tailTimer = setInterval(() => refreshOpenTrade(tradeId), state.tailIntervalMs);
}

function stopLiveTail() {
  if (state.tailTimer) { clearInterval(state.tailTimer); state.tailTimer = null; }
  document.getElementById('chart-meta').classList.remove('is-tailing');
}

async function refreshOpenTrade(tradeId) {
  // Bail if the user has navigated away
  if (state.selectedTradeId !== tradeId) { stopLiveTail(); return; }
  if (document.visibilityState === 'hidden') return;  // pause when tab is bg
  let payload;
  try {
    const res = await fetch(
      `/api/trade/${encodeURIComponent(state.selectedVariant)}/${tradeId}/chart`,
    );
    if (!res.ok) return;
    payload = await res.json();
  } catch { return; }

  // Re-render in-place without resetting the user's pan/zoom. setData
  // with new bars preserves the time scale; we skip the fitContent()
  // call that the initial render does.
  renderPayloadInPlace(payload);
  renderReasoning(payload.reasoning);
  applyChartHeader(payload);

  // Position closed since last check → stop polling.
  if (!payload.is_open) stopLiveTail();
}

function clearChart() {
  state.candleSeries.setData([]);
  state.overlaySeries.forEach(s => state.chart.removeSeries(s));
  state.overlaySeries = [];
  // Remove entry/exit price lines from previous trade
  for (const pl of state.fillPriceLines) {
    try { state.candleSeries.removePriceLine(pl); } catch {}
  }
  state.fillPriceLines = [];
  // Clear ZS boxes (DOM nodes + tracking)
  for (const box of state.zsBoxes) box.el.remove();
  state.zsBoxes = [];
  // Tear down indicator series — buildIndicators re-creates them.
  teardownIndicators();
}

function renderPayload(p) {
  clearChart();
  state.currentPayload = p;

  // Bars
  const candleData = p.bars.map(b => ({
    time: b.time,
    open: b.open, high: b.high, low: b.low, close: b.close,
  }));
  state.candleSeries.setData(candleData);

  const fillMarkers = _buildFillMarkers(p.fills);

  const bspMarkers = [];
  if (state.indicators.structures) {
    for (const o of p.overlays || []) {
      switch (o.kind) {
        case 'line':   renderLine(o); break;
        case 'zone':   renderZone(o); break;
        case 'marker': bspMarkers.push(markerToSeries(o)); break;
        case 'level':  renderLevel(o); break;
        case 'ma':     renderMA(o); break;
        case 'band':   renderBand(o); break;
        default: console.warn('unknown overlay kind:', o.kind);
      }
    }
  }

  // Markers (BSP labels + entry/exit fills) + horizontal price lines for fills
  state.candleSeries.setMarkers(
    [...bspMarkers, ...fillMarkers].sort((a, b) => a.time - b.time),
  );
  _drawFillPriceLines(p.fills);

  // Indicators (volume / MACD) — applies pane layout via scaleMargins
  buildIndicators(p.bars);

  // Full reset on trade/variant switch — fitContent alone only resets the
  // time axis; without re-enabling autoScale on each price pane, the
  // previous trade's price range stays locked (e.g. switching from a $50
  // stock to a $500 stock leaves the y-axis on the old range, hiding the
  // new candles). Mirror fitChart()'s full reset.
  state.chart.timeScale().fitContent();
  state.candleSeries.priceScale().applyOptions({ autoScale: true });
  if (state.volumeSeries)   state.chart.priceScale('volume').applyOptions({ autoScale: true });
  if (state.macdLineSeries) state.chart.priceScale('macd').applyOptions({ autoScale: true });
}

// Build prominent BUY/SELL markers. The text is intentionally short
// ("▲ BUY 102.45" / "▼ SELL 116.30") so it's readable at the chart edge
// without overlapping; the qty isn't displayed here (it's in the
// reasoning panel). The unicode triangle gives a stronger visual cue
// than lightweight-charts' default arrow shape alone.
function _buildFillMarkers(fills) {
  return (fills || []).map(f => {
    const isBuy = f.side === 'buy';
    return {
      time: f.time,
      position: isBuy ? 'belowBar' : 'aboveBar',
      color: isBuy ? '#10b981' : '#ef4444',
      shape: isBuy ? 'arrowUp' : 'arrowDown',
      text: `${isBuy ? '▲ BUY' : '▼ SELL'} ${Number(f.price).toFixed(2)}`,
      size: 2,  // larger than default 1
    };
  });
}

// Horizontal price lines at each fill price — green for entry, red for
// exit — span the full chart width. Much more visible than the marker
// arrows alone, especially for closed trades where you want to read off
// the entry-to-exit travel at a glance.
function _drawFillPriceLines(fills) {
  for (const f of fills || []) {
    const isBuy = f.side === 'buy';
    const pl = state.candleSeries.createPriceLine({
      price: Number(f.price),
      color: isBuy ? '#10b981' : '#ef4444',
      lineWidth: 1,
      lineStyle: 2,  // dashed
      axisLabelVisible: true,
      title: isBuy ? 'BUY' : 'SELL',
    });
    state.fillPriceLines.push(pl);
  }
}

// In-place re-render used by the live-tail refresh — same pipeline as
// renderPayload, but without fitContent() so the user's pan/zoom is
// preserved. Called every tailIntervalMs while a position is open.
function renderPayloadInPlace(p) {
  // Save visible range so it survives any scaleMargin shuffling.
  const visible = state.chart.timeScale().getVisibleLogicalRange();
  clearChart();
  state.currentPayload = p;

  const candleData = p.bars.map(b => ({
    time: b.time, open: b.open, high: b.high, low: b.low, close: b.close,
  }));
  state.candleSeries.setData(candleData);

  const fillMarkers = _buildFillMarkers(p.fills);

  const bspMarkers = [];
  if (state.indicators.structures) {
    for (const o of p.overlays || []) {
      switch (o.kind) {
        case 'line':   renderLine(o); break;
        case 'zone':   renderZone(o); break;
        case 'marker': bspMarkers.push(markerToSeries(o)); break;
        case 'level':  renderLevel(o); break;
        case 'ma':     renderMA(o); break;
        case 'band':   renderBand(o); break;
      }
    }
  }
  state.candleSeries.setMarkers(
    [...bspMarkers, ...fillMarkers].sort((a, b) => a.time - b.time),
  );
  _drawFillPriceLines(p.fills);
  buildIndicators(p.bars);

  if (visible) state.chart.timeScale().setVisibleLogicalRange(visible);
}

function renderLine(o) {
  // Two-point line drawn as a line series (lightweight-charts doesn't have
  // a true 2-point segment primitive on v4; a 2-point line series is
  // visually equivalent and supports color/width/dash).
  const s = state.chart.addLineSeries({
    color: o.color || '#5b8def',
    lineWidth: o.width || 1,
    lineStyle: o.style === 'dashed' ? LightweightCharts.LineStyle.Dashed
              : o.style === 'dotted' ? LightweightCharts.LineStyle.Dotted
              : LightweightCharts.LineStyle.Solid,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
  });
  s.setData([
    { time: o.from_t, value: o.from_p },
    { time: o.to_t,   value: o.to_p },
  ]);
  state.overlaySeries.push(s);
}

function renderZone(o) {
  // ZS box: a real filled rectangle, rendered as an absolutely-positioned
  // <div> over the chart. lightweight-charts v4 has no rectangle primitive,
  // so we sync the DOM box's pixel rect to the series coordinate API every
  // animation frame (see startZSLoop / repositionZS). Single label "ZS"
  // sits inside the top-left corner of the box — no axis-legend pollution.
  const el = document.createElement('div');
  el.className = 'zs-box';
  const label = document.createElement('span');
  label.className = 'zs-label';
  label.textContent = o.label || 'ZS';
  el.appendChild(label);
  state.zsLayer.appendChild(el);

  state.zsBoxes.push({
    el,
    from_t: o.from_t,
    to_t:   o.to_t,
    low:    o.low,
    high:   o.high,
  });
  startZSLoop();
}

function renderLevel(o) {
  // Horizontal price line. If from_t/to_t are provided, draw a line series
  // between them; otherwise use a global price line on the candle series.
  if (o.from_t && o.to_t && o.from_t !== o.to_t) {
    const s = state.chart.addLineSeries({
      color: o.color || '#a78bfa', lineWidth: 1,
      lineStyle: o.style === 'dashed' ? LightweightCharts.LineStyle.Dashed
                : o.style === 'dotted' ? LightweightCharts.LineStyle.Dotted
                : LightweightCharts.LineStyle.Solid,
      lastValueVisible: true,
      title: o.label || '',
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    s.setData([
      { time: o.from_t, value: o.price },
      { time: o.to_t,   value: o.price },
    ]);
    state.overlaySeries.push(s);
  } else {
    // Global price line
    state.candleSeries.createPriceLine({
      price: o.price,
      color: o.color || '#a78bfa',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      title: o.label || '',
    });
  }
}

function renderMA(o) {
  const s = state.chart.addLineSeries({
    color: o.color || '#888', lineWidth: 1,
    lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false,
    title: o.label || `MA${o.period}`,
  });
  s.setData(o.values.map(([t, v]) => ({ time: t, value: v })));
  state.overlaySeries.push(s);
}

function renderBand(o) {
  // Band: filled rectangle, used for VCP base / opening-range box. Uses
  // the same mechanism as renderZone.
  renderZone({
    kind: 'zone',
    from_t: o.from_t, to_t: o.to_t,
    low: o.low, high: o.high,
    label: o.label, color: o.color || 'rgba(91, 141, 239, 0.15)',
  });
}

function markerToSeries(o) {
  return {
    time: o.time,
    position: o.side === 'buy' ? 'belowBar'
            : o.side === 'sell' ? 'aboveBar'
            : 'inBar',
    color: o.color || '#a78bfa',
    shape: o.side === 'buy' ? 'circle'
          : o.side === 'sell' ? 'circle'
          : 'square',
    text: o.label || '',
  };
}

// ── Reasoning panel ──────────────────────────────────────────────────
function renderReasoning(r) {
  document.getElementById('reasoning-headline').textContent = r.headline || '—';
  const ul = document.getElementById('reasoning-criteria');
  ul.innerHTML = '';
  (r.criteria || []).forEach(c => {
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="${c.passed ? 'ok' : 'fail'}">${c.passed ? '✓' : '✗'}</span>
      <span class="name">${escapeHtml(c.name)}</span>
      <span class="value">${c.value == null ? '' : escapeHtml(String(c.value))}</span>`;
    ul.appendChild(li);
  });
  const dl = document.getElementById('reasoning-metrics');
  dl.innerHTML = '';
  (r.metrics || []).forEach(m => {
    const dt = document.createElement('dt'); dt.textContent = m.label;
    const dd = document.createElement('dd'); dd.textContent = m.value;
    dl.appendChild(dt); dl.appendChild(dd);
  });
  const pre = document.getElementById('reasoning-narrative');
  pre.textContent = r.narrative || '—';
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

function showError(msg) {
  const el = document.getElementById('chart-error');
  el.textContent = msg;
  el.hidden = false;
}

// ── Page: LIVE LOG ──────────────────────────────────────────────────
const _log = {
  inited: false,
  n: 200,
  autoRefresh: true,
  timer: null,
};

async function renderLogPage() {
  if (!_log.inited) {
    document.querySelectorAll('#log-tail-n .seg-btn').forEach(b => {
      b.addEventListener('click', () => {
        document.querySelectorAll('#log-tail-n .seg-btn').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        _log.n = parseInt(b.dataset.n, 10) || 200;
        loadLogTail();
      });
    });
    document.getElementById('log-autorefresh').addEventListener('change', e => {
      _log.autoRefresh = e.target.checked;
      if (_log.autoRefresh) startLogAutoRefresh();
      else stopLogAutoRefresh();
    });
    _log.inited = true;
  }
  await loadLogTail();
  if (_log.autoRefresh) startLogAutoRefresh();
}

async function loadLogTail() {
  const meta = document.getElementById('log-meta');
  const body = document.getElementById('log-body');
  try {
    const res = await fetch(`/api/log/tail?n=${_log.n}`);
    const data = await res.json();
    if (!data.exists) {
      meta.textContent = `${data.path} · missing`;
      body.textContent = '(log file not found)';
      return;
    }
    const mt = data.mtime ? new Date(data.mtime * 1000).toLocaleTimeString() : '';
    meta.textContent = `${data.path} · last write ${mt} · ${data.lines.length} lines`;
    // Color-code error lines via spans; preserve original ordering.
    body.innerHTML = data.lines.map(line => {
      const isErr = /ERROR|Traceback|\berror\b/i.test(line);
      const safe = escapeHtml(line);
      return isErr ? `<span class="log-err">${safe}</span>` : safe;
    }).join('\n');
    body.scrollTop = body.scrollHeight;
  } catch (e) {
    meta.textContent = `error: ${e}`;
  }
}

function startLogAutoRefresh() {
  stopLogAutoRefresh();
  _log.timer = setInterval(() => {
    if (document.visibilityState === 'visible') loadLogTail();
  }, 5000);
}

function stopLogAutoRefresh() {
  if (_log.timer) { clearInterval(_log.timer); _log.timer = null; }
}

// ── SSE event feed ───────────────────────────────────────────────────
function startEventStream() {
  const ev = new EventSource('/events/stream');
  const status = document.getElementById('event-status');
  ev.onopen = () => { status.textContent = 'events: live'; };
  ev.onerror = () => { status.textContent = 'events: reconnecting…'; };
  ev.onmessage = e => {
    try {
      const rec = JSON.parse(e.data);
      appendEvent(rec);
    } catch { /* ignore non-JSON heartbeats */ }
  };
}

function appendEvent(rec) {
  const ul = document.getElementById('event-feed');
  const li = document.createElement('li');
  li.classList.add(rec.level || 'info');
  const ts = (rec.ts || '').slice(11, 19);
  li.innerHTML = `
    <span class="ts">${ts}</span>
    <span class="cat">${escapeHtml(rec.category || '?')}</span>
    <span class="msg">${escapeHtml(rec.message || '')}${rec.symbol ? ' · ' + escapeHtml(rec.symbol) : ''}</span>`;
  ul.insertBefore(li, ul.firstChild);
  // Cap at 200 rows
  while (ul.children.length > 200) ul.removeChild(ul.lastChild);
}

// ── Boot ────────────────────────────────────────────────────────────
init();
