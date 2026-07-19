// ─────────────────────────────────────────────────────────────
// Stockex · frontend
// ─────────────────────────────────────────────────────────────
const LWC = LightweightCharts;

const state = {
  period: "1y",
  interval: "1d",
  homeChart: null, homeSeries: null,
  tkrChart: null, tkrSeries: null, tkrExtraSeries: [], tkrPriceLines: [],
  subChart: null, subSeries: [],
  currentTicker: null,
  analysis: null,
  activeOverlay: "none",
};

// ── helpers ────────────────────────────────────────────────
function toUnixTime(str) {
  if (!str) return null;
  const iso = str.includes("T") ? str : str.replace(" ", "T");
  const t = Date.parse(iso.length <= 10 ? iso : iso + "Z");
  return Math.floor(t / 1000);
}

function toBarTime(dates, barIdx) {
  if (barIdx == null || !dates || barIdx < 0 || barIdx >= dates.length) return null;
  return toUnixTime(dates[barIdx]);
}

async function fetchJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `Request failed: ${url}`);
  return data;
}

function fmtPrice(v) {
  if (v == null || isNaN(v)) return "—";
  return v >= 1000 ? v.toLocaleString(undefined, {maximumFractionDigits: 2})
                    : v.toFixed(2);
}

function pctColorClass(pct) {
  return pct >= 0 ? "up" : "down";
}

function seriesToOHLC(series) {
  const out = [];
  for (let i = 0; i < series.dates.length; i++) {
    if (series.open[i] == null || series.high[i] == null || series.low[i] == null || series.close[i] == null) continue;
    out.push({
      time: toUnixTime(series.dates[i]),
      open: series.open[i], high: series.high[i], low: series.low[i], close: series.close[i],
    });
  }
  out.sort((a, b) => a.time - b.time);
  return out;
}

// ── chart factories ───────────────────────────────────────
function makeCandleChart(containerId) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  el.style.position = "relative";
  const chart = LWC.createChart(el, {
    width: el.clientWidth, height: el.clientHeight,
    layout: { background: { color: "#ffffff" }, textColor: "#52545e", fontFamily: "IBM Plex Mono, ui-monospace, monospace" },
    grid: { vertLines: { color: "#f4f5f7" }, horzLines: { color: "#f4f5f7" } },
    rightPriceScale: { borderColor: "#e8e9ed" },
    timeScale: { borderColor: "#e8e9ed", timeVisible: true, secondsVisible: false },
    crosshair: { mode: LWC.CrosshairMode.Normal },
  });
  const series = chart.addCandlestickSeries({
    upColor: "#0e9d6e", downColor: "#e0233f",
    borderUpColor: "#0e9d6e", borderDownColor: "#e0233f",
    wickUpColor: "#0e9d6e", wickDownColor: "#e0233f",
  });

  // ── zone overlay (order blocks / FVGs) ──────────────────────
  // Lightweight Charts has no built-in "shaded box" primitive, so zones are
  // painted onto a transparent <canvas> stacked on top of the chart and
  // repositioned on every pan/zoom/resize using the chart's own coordinate
  // conversion (series.priceToCoordinate / timeScale().timeToCoordinate).
  const zoneCanvas = document.createElement("canvas");
  zoneCanvas.className = "zone-canvas";
  el.appendChild(zoneCanvas);
  const zctx = zoneCanvas.getContext("2d");
  let zones = [];

  function drawZones() {
    const w = el.clientWidth, h = el.clientHeight;
    zctx.clearRect(0, 0, w, h);
    zones.forEach(z => {
      const y1 = series.priceToCoordinate(z.top);
      const y2 = series.priceToCoordinate(z.bottom);
      if (y1 == null || y2 == null) return;
      // zones are "active" (unfilled/unmitigated) so they always extend
      // to the right edge of the chart; if the start bar has scrolled out
      // of view to the left, flush the box to the left edge instead.
      let x1 = chart.timeScale().timeToCoordinate(z.fromTime);
      if (x1 == null) x1 = 0;
      const x2 = w;
      if (x2 <= x1) return;
      const top = Math.min(y1, y2), height = Math.max(1, Math.abs(y2 - y1));
      zctx.fillStyle = z.fill;
      zctx.fillRect(x1, top, x2 - x1, height);
      zctx.strokeStyle = z.stroke;
      zctx.lineWidth = 1;
      zctx.strokeRect(x1 + 0.5, top + 0.5, Math.max(0, x2 - x1 - 1), Math.max(0, height - 1));
    });
  }

  function resizeZoneCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = el.clientWidth, h = el.clientHeight;
    zoneCanvas.width = w * dpr;
    zoneCanvas.height = h * dpr;
    zoneCanvas.style.width = w + "px";
    zoneCanvas.style.height = h + "px";
    zctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawZones();
  }

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => drawZones());
  new ResizeObserver(() => {
    chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    resizeZoneCanvas();
  }).observe(el);
  requestAnimationFrame(resizeZoneCanvas);

  const zoneApi = {
    setZones(list) { zones = list || []; drawZones(); },
    clear() { zones = []; drawZones(); },
  };

  return { chart, series, zoneApi };
}

function makeSubChart(containerId) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  const chart = LWC.createChart(el, {
    width: el.clientWidth, height: el.clientHeight,
    layout: { background: { color: "#ffffff" }, textColor: "#52545e", fontFamily: "IBM Plex Mono, ui-monospace, monospace" },
    grid: { vertLines: { color: "#f4f5f7" }, horzLines: { color: "#f4f5f7" } },
    rightPriceScale: { borderColor: "#e8e9ed" },
    timeScale: { borderColor: "#e8e9ed", timeVisible: true, secondsVisible: false },
  });
  new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight })).observe(el);
  return chart;
}

function syncTimeScales(chartA, chartB) {
  chartA.timeScale().subscribeVisibleLogicalRangeChange(r => {
    if (r) chartB.timeScale().setVisibleLogicalRange(r);
  });
  chartB.timeScale().subscribeVisibleLogicalRangeChange(r => {
    if (r) chartA.timeScale().setVisibleLogicalRange(r);
  });
}

// ── home view ──────────────────────────────────────────────
async function initHome() {
  const { chart, series } = makeCandleChart("homeChart");
  state.homeChart = chart; state.homeSeries = series;
  try {
    const data = await fetchJSON(`/api/candles?ticker=${encodeURIComponent("^IXIC")}&period=${state.period}&interval=${state.interval}`);
    series.setData(seriesToOHLC(data));
    chart.timeScale().fitContent();
    document.getElementById("homePrice").textContent = fmtPrice(data.current_price);
    const chEl = document.getElementById("homeChange");
    chEl.textContent = `${data.change >= 0 ? "+" : ""}${data.change} (${data.change_pct}%)`;
    chEl.className = "price-change " + pctColorClass(data.change_pct);

    // header index badge — short-form snapshot, visible on every view
    const badgeVal = document.getElementById("indexBadgeValue");
    const badgeChg = document.getElementById("indexBadgeChange");
    if (badgeVal) badgeVal.textContent = fmtShort(data.current_price);
    if (badgeChg) {
      badgeChg.textContent = `${data.change_pct >= 0 ? "+" : ""}${data.change_pct}%`;
      badgeChg.className = "index-badge-change mono " + (data.change_pct >= 0 ? "up" : "down");
    }
  } catch (e) {
    document.getElementById("homeChart").innerHTML = `<div class="muted">Couldn't load NASDAQ data: ${e.message}</div>`;
  }
  loadMovers(["gainersList", "losersList"]);
}

// compact "short form" number — 18,540.20 stays as-is, larger values get a K/M suffix
function fmtShort(n) {
  if (n == null || isNaN(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (abs >= 100_000) return (n / 1000).toFixed(1) + "K";
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function loadMovers(targetIds) {
  try {
    const data = await fetchJSON("/api/movers");
    renderMoverList(targetIds[0], data.gainers);
    renderMoverList(targetIds[1], data.losers);
  } catch (e) {
    targetIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = `<div class="muted small">Unavailable</div>`;
    });
  }
}

function renderMoverList(id, list) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!list || !list.length) { el.innerHTML = `<div class="muted small">No data</div>`; return; }
  el.innerHTML = list.map(m => `
    <div class="mover-row" data-ticker="${m.ticker}">
      <span class="sym">${m.ticker}</span>
      <span>
        <span>${fmtPrice(m.price)}</span>
        <span class="pct ${pctColorClass(m.change_pct)}">${m.change_pct >= 0 ? "+" : ""}${m.change_pct}%</span>
      </span>
    </div>`).join("");
  el.querySelectorAll(".mover-row").forEach(row => {
    row.addEventListener("click", () => selectTicker(row.dataset.ticker));
  });
}

// ── search ─────────────────────────────────────────────────
let searchDebounce = null;
document.getElementById("searchInput").addEventListener("input", (e) => {
  clearTimeout(searchDebounce);
  const q = e.target.value.trim();
  const box = document.getElementById("searchResults");
  if (!q) { box.classList.add("hidden"); return; }
  searchDebounce = setTimeout(async () => {
    try {
      const data = await fetchJSON(`/api/search?q=${encodeURIComponent(q)}`);
      renderSearchResults(data.results || []);
    } catch (e) { /* silent */ }
  }, 280);
});
document.getElementById("searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const q = e.target.value.trim();
    if (q) selectTicker(q.toUpperCase());
  }
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) document.getElementById("searchResults").classList.add("hidden");
});

function renderSearchResults(results) {
  const box = document.getElementById("searchResults");
  if (!results.length) { box.classList.add("hidden"); return; }
  box.innerHTML = results.map(r => `
    <div class="search-result-item" data-symbol="${r.symbol}">
      <span class="sym">${r.symbol}</span>
      <span class="name">${r.name || ""} ${r.exchange ? "· " + r.exchange : ""}</span>
    </div>`).join("");
  box.classList.remove("hidden");
  box.querySelectorAll(".search-result-item").forEach(item => {
    item.addEventListener("click", () => selectTicker(item.dataset.symbol));
  });
}

document.getElementById("brandHome").addEventListener("click", showHome);
document.getElementById("backHomeBtn").addEventListener("click", showHome);
document.getElementById("periodSelect").addEventListener("change", (e) => { state.period = e.target.value; refreshCurrent(); });
document.getElementById("intervalSelect").addEventListener("change", (e) => { state.interval = e.target.value; refreshCurrent(); });

function refreshCurrent() {
  if (state.currentTicker) selectTicker(state.currentTicker);
  else initHome();
}

function showHome() {
  document.getElementById("tickerView").classList.add("hidden");
  document.getElementById("homeView").classList.remove("hidden");
  state.currentTicker = null;
  document.getElementById("searchInput").value = "";
}

// ── ticker analysis view ──────────────────────────────────
async function selectTicker(symbol) {
  document.getElementById("searchResults").classList.add("hidden");
  document.getElementById("searchInput").value = symbol;
  document.getElementById("homeView").classList.add("hidden");
  document.getElementById("tickerView").classList.remove("hidden");
  state.currentTicker = symbol;

  document.getElementById("tkrTitle").innerHTML = `${symbol} <span class="ticker-pill">loading…</span>`;
  const { chart, series, zoneApi } = makeCandleChart("tkrChart");
  state.tkrChart = chart; state.tkrSeries = series; state.tkrZoneApi = zoneApi;
  state.tkrExtraSeries = []; state.tkrPriceLines = [];
  document.getElementById("tkrSubChart").classList.add("hidden");
  document.getElementById("overlayLegend").innerHTML = "";
  setActiveToggle("none");

  loadMovers(["gainersList2", "losersList2"]);

  try {
    const data = await fetchJSON(`/api/analyze?ticker=${encodeURIComponent(symbol)}&period=${state.period}&interval=${state.interval}`);
    state.analysis = data;
    renderTickerView(data);
  } catch (e) {
    document.getElementById("tkrTitle").innerHTML = `${symbol} <span class="ticker-pill">error</span>`;
    document.getElementById("tkrChart").innerHTML = `<div class="muted">Couldn't analyze ${symbol}: ${e.message}</div>`;
  }
}

function baseOHLCFromAnalysis(data) {
  // Any indicator's series has the same underlying OHLC — RSI is always run.
  for (const key of ["RSI", "OrderBlock", "MACD", "Bollinger", "Ichimoku", "Other"]) {
    const ind = data.indicators[key];
    if (ind && ind.series && ind.series.dates) return ind.series;
  }
  return null;
}

function renderTickerView(data) {
  const base = baseOHLCFromAnalysis(data);
  document.getElementById("tkrTitle").innerHTML = `${data.ticker} <span class="ticker-pill">${state.interval.toUpperCase()} · ${state.period}</span>`;
  document.getElementById("tkrPrice").textContent = fmtPrice(data.current_price);

  const rsi = data.indicators.RSI;
  let changeEl = document.getElementById("tkrChange");
  if (base && base.close && base.close.length > 1) {
    const last = base.close[base.close.length - 1], prev = base.close[base.close.length - 2];
    const pct = prev ? ((last - prev) / prev * 100) : 0;
    changeEl.textContent = `${(last - prev) >= 0 ? "+" : ""}${(last - prev).toFixed(2)} (${pct.toFixed(2)}%)`;
    changeEl.className = "price-change " + pctColorClass(pct);
  } else {
    changeEl.textContent = ""; changeEl.className = "price-change";
  }

  if (base) {
    state.tkrSeries.setData(seriesToOHLC(base));
    state.tkrChart.timeScale().fitContent();
  }

  renderIndicatorGrid(data.indicators);
  renderMomentumGauge(data.momentum_gauge);
  renderVerdict(data.composite);
  renderTradeLevels(data);
  applyOverlay("none");
}

// ── indicator snapshot grid ────────────────────────────────
function signalClass(sig) {
  if (!sig) return "neutral";
  const s = sig.toUpperCase();
  if (s.includes("BUY") && !s.includes("WATCH")) return "buy";
  if (s.includes("SELL") && !s.includes("WATCH")) return "sell";
  return "neutral";
}

function renderIndicatorGrid(indicators) {
  const el = document.getElementById("indicatorGrid");
  const order = ["RSI", "MACD", "Bollinger", "Ichimoku", "OrderBlock", "SR"];
  el.innerHTML = order.map(name => {
    const ind = indicators[name];
    if (!ind || ind.error) {
      return `<div class="ind-card err"><div class="ind-name">${name}</div><div class="ind-signal">N/A</div><div class="ind-conf">${ind ? ind.error : "no data"}</div></div>`;
    }
    const cls = signalClass(ind.signal);
    return `<div class="ind-card ${cls}">
      <div class="ind-name">${name}</div>
      <div class="ind-signal">${ind.signal}</div>
      <div class="ind-conf">Confidence ${ind.confidence}%</div>
    </div>`;
  }).join("");
}

// ── buy / sell levels (relative to current price) ─────────
function fmtSignedPct(pct) {
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function renderTradeLevels(data) {
  const el = document.getElementById("tradeLevels");
  if (!el) return;
  const current = data.current_price;
  const levels = data.levels;
  const ob = data.indicators && data.indicators.OrderBlock;
  const plan = ob && !ob.error ? ob.entry_plan : null;

  let html = `<div class="tl-current"><span>Current Price</span><span class="tl-price">${fmtPrice(current)}</span></div>`;

  const support = ((levels && levels.support) || [])
    .filter(z => z.mid < current)
    .sort((a, b) => (current - a.mid) - (current - b.mid))
    .slice(0, 3);
  const resistance = ((levels && levels.resistance) || [])
    .filter(z => z.mid > current)
    .sort((a, b) => (a.mid - current) - (b.mid - current))
    .slice(0, 3);

  html += `<div class="tl-cols">
    <div>
      <div class="tl-col-title buy">Buy near (support)</div>
      ${support.length ? support.map(z => `
        <div class="tl-zone buy">
          <div class="tl-range">${fmtPrice(z.low)} – ${fmtPrice(z.high)}</div>
          <div class="tl-dist">${fmtSignedPct((z.mid - current) / current * 100)} from current · strength ${z.strength}</div>
        </div>`).join("") : `<div class="tl-empty">No nearby support zone</div>`}
    </div>
    <div>
      <div class="tl-col-title sell">Sell near (resistance)</div>
      ${resistance.length ? resistance.map(z => `
        <div class="tl-zone sell">
          <div class="tl-range">${fmtPrice(z.low)} – ${fmtPrice(z.high)}</div>
          <div class="tl-dist">${fmtSignedPct((z.mid - current) / current * 100)} from current · strength ${z.strength}</div>
        </div>`).join("") : `<div class="tl-empty">No nearby resistance zone</div>`}
    </div>
  </div>`;

  if (levels && (levels.stop_loss != null || levels.take_profit != null)) {
    html += `<div class="tl-setup">
      <div class="tl-setup-head"><span>Suggested Stop / Target</span></div>
      <div>Stop-loss <strong>${levels.stop_loss != null ? fmtPrice(levels.stop_loss) : "—"}</strong>
        &nbsp;·&nbsp; Take-profit <strong>${levels.take_profit != null ? fmtPrice(levels.take_profit) : "—"}</strong></div>
    </div>`;
  }

  if (plan) {
    const dirClass = plan.direction === "BUY" ? "tl-dir-buy" : "tl-dir-sell";
    html += `<div class="tl-setup">
      <div class="tl-setup-head"><span>Order Block Setup — ${plan.status}</span>
        <span class="${dirClass}">${plan.direction}</span></div>
      <div>Entry <strong>${fmtPrice(plan.entry_price)}</strong> · Stop <strong>${fmtPrice(plan.stop_loss)}</strong>
      ${plan.take_profit != null ? ` · Target <strong>${fmtPrice(plan.take_profit)}</strong>` : ""}</div>
    </div>`;
  }

  el.innerHTML = html;
}

// ── momentum gauge (SVG dial, gradient arc + ticks + zone labels) ──
function polar(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.sin(rad), y: cy - r * Math.cos(rad) };
}

function renderMomentumGauge(gauge) {
  const wrap = document.getElementById("momentumGauge");
  const score = gauge ? gauge.score : 50;
  const angle = -90 + (score / 100) * 180; // -90 (bearish) .. 0 (neutral) .. +90 (bullish)
  const cx = 130, cy = 132, r = 104;
  const needle = polar(cx, cy, r - 18, angle);
  const scoreColor = score >= 60 ? "#0e9d6e" : score <= 40 ? "#e0233f" : "#b5790f";

  // 11 tick marks at every 10 points; every 5th tick (0/50/100) drawn longer/darker
  let ticks = "";
  for (let i = 0; i <= 10; i++) {
    const a = -90 + i * 18;
    const major = i === 0 || i === 5 || i === 10;
    const p1 = polar(cx, cy, r + 10, a);
    const p2 = polar(cx, cy, r + (major ? 20 : 16), a);
    ticks += `<line x1="${p1.x.toFixed(1)}" y1="${p1.y.toFixed(1)}" x2="${p2.x.toFixed(1)}" y2="${p2.y.toFixed(1)}"
      stroke="${major ? '#c3c5cc' : '#dfe0e5'}" stroke-width="${major ? 2 : 1.4}" stroke-linecap="round" />`;
  }

  const start = polar(cx, cy, r, -90), end = polar(cx, cy, r, 90);

  wrap.innerHTML = `
  <svg width="260" height="192" viewBox="0 0 260 192">
    <defs>
      <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#e0233f" />
        <stop offset="50%" stop-color="#b5790f" />
        <stop offset="100%" stop-color="#0e9d6e" />
      </linearGradient>
      <filter id="needleGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="${scoreColor}" flood-opacity="0.45" />
      </filter>
    </defs>

    <path d="M ${start.x.toFixed(1)} ${start.y.toFixed(1)} A ${r} ${r} 0 0 1 ${end.x.toFixed(1)} ${end.y.toFixed(1)}"
      fill="none" stroke="#eef0f3" stroke-width="20" stroke-linecap="round" />
    <path d="M ${start.x.toFixed(1)} ${start.y.toFixed(1)} A ${r} ${r} 0 0 1 ${end.x.toFixed(1)} ${end.y.toFixed(1)}"
      fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round" opacity="0.92" />
    ${ticks}

    <text x="${cx - r - 4}" y="${cy + 26}" text-anchor="start" font-size="10" font-weight="700" letter-spacing=".06em" fill="#e0233f">BEARISH</text>
    <text x="${cx}" y="${cy - r - 14}" text-anchor="middle" font-size="10" font-weight="700" letter-spacing=".06em" fill="#b5790f">NEUTRAL</text>
    <text x="${cx + r + 4}" y="${cy + 26}" text-anchor="end" font-size="10" font-weight="700" letter-spacing=".06em" fill="#0e9d6e">BULLISH</text>

    <g filter="url(#needleGlow)">
      <line x1="${cx}" y1="${cy}" x2="${needle.x.toFixed(1)}" y2="${needle.y.toFixed(1)}" stroke="#13151a" stroke-width="3.5" stroke-linecap="round" />
      <circle cx="${needle.x.toFixed(1)}" cy="${needle.y.toFixed(1)}" r="4" fill="${scoreColor}" />
    </g>
    <circle cx="${cx}" cy="${cy}" r="7" fill="#fff" stroke="#13151a" stroke-width="3" />

    <text x="${cx}" y="${cy + 44}" text-anchor="middle" font-size="30" font-weight="700" fill="${scoreColor}" font-family="IBM Plex Mono, monospace">${score}</text>
    <text x="${cx}" y="${cy + 60}" text-anchor="middle" font-size="10" font-weight="600" fill="#94969f" letter-spacing=".04em">OUT OF 100</text>
  </svg>`;

  document.getElementById("momentumLabel").textContent = gauge ? gauge.label : "—";
  const compEl = document.getElementById("momentumComponents");
  compEl.innerHTML = (gauge && gauge.components || []).map(c => {
    const num = parseFloat(c.value);
    const hasBar = !isNaN(num) && isFinite(num);
    const pct = hasBar ? Math.max(0, Math.min(100, Math.abs(num) <= 1 ? Math.abs(num) * 100 : num)) : 0;
    const barColor = pct >= 60 ? "var(--buy)" : pct <= 40 ? "var(--sell)" : "var(--neutral)";
    return `
    <div class="gauge-comp-row">
      <div class="gauge-comp-top"><span>${c.name}</span><span>${c.value}</span></div>
      ${hasBar ? `<div class="gauge-comp-bar"><div class="gauge-comp-fill" style="width:${pct}%;background:${barColor}"></div></div>` : ""}
    </div>`;
  }).join("");
}

// ── final verdict ──────────────────────────────────────────
function verdictClass(label) {
  return (label || "NEUTRAL").toLowerCase().replace(/\s+/g, "-");
}

function renderVerdict(composite) {
  if (!composite) return;
  const badge = document.getElementById("verdictBadge");
  badge.textContent = `${composite.label} (${composite.rating > 0 ? "+" : ""}${composite.rating})`;
  badge.className = "verdict-badge " + verdictClass(composite.label);

  // rating -100..100 -> 0..100% position on the scale
  const pos = ((composite.rating + 100) / 200) * 100;
  document.getElementById("verdictMarker").style.left = `${pos}%`;
  document.getElementById("verdictReason").textContent = composite.reason || "";

  const bEl = document.getElementById("verdictBreakdown");
  bEl.innerHTML = (composite.breakdown || [])
    .filter(b => b.indicator !== "Other")
    .map(b => {
    if (!b.included) {
      return `<div class="vb-row"><div class="vb-row-top"><span class="ind">${b.indicator}</span><span class="vb-excluded">excluded</span></div></div>`;
    }
    const pct = Math.min(50, Math.abs(b.score) * 50);
    const cls = b.score >= 0 ? "pos" : "neg";
    return `<div class="vb-row">
      <div class="vb-row-top"><span class="ind">${b.indicator}</span><span>${b.signal} · w=${b.weight}</span></div>
      <div class="vb-bar-track"><div class="vb-bar-fill ${cls}" style="width:${pct}%"></div></div>
    </div>`;
  }).join("");
}

// ── overlay toggle logic ───────────────────────────────────
document.getElementById("overlayToggleBar").addEventListener("click", (e) => {
  const btn = e.target.closest(".toggle-btn");
  if (!btn) return;
  applyOverlay(btn.dataset.view);
});

function setActiveToggle(view) {
  document.querySelectorAll(".toggle-btn").forEach(b => b.classList.toggle("active", b.dataset.view === view));
}

function clearOverlays() {
  state.tkrExtraSeries.forEach(s => { try { state.tkrChart.removeSeries(s); } catch (e) {} });
  state.tkrExtraSeries = [];
  state.tkrPriceLines.forEach(pl => { try { state.tkrSeries.removePriceLine(pl); } catch (e) {} });
  state.tkrPriceLines = [];
  state.tkrSeries.setMarkers([]);
  if (state.tkrZoneApi) state.tkrZoneApi.clear();
  document.getElementById("overlayLegend").innerHTML = "";
  const subEl = document.getElementById("tkrSubChart");
  subEl.classList.add("hidden");
  subEl.innerHTML = "";
  state.subChart = null; state.subSeries = [];
}

function legend(items) {
  document.getElementById("overlayLegend").innerHTML = items.map(
    it => `<span><span class="dot" style="background:${it.color}"></span>${it.label}</span>`
  ).join("");
}

function addPriceLine(price, color, title, dashed = true) {
  const pl = state.tkrSeries.createPriceLine({
    price, color, lineWidth: 1, lineStyle: dashed ? LWC.LineStyle.Dashed : LWC.LineStyle.Solid,
    axisLabelVisible: true, title,
  });
  state.tkrPriceLines.push(pl);
}

function openSubChart() {
  document.getElementById("tkrSubChart").classList.remove("hidden");
  const chart = makeSubChart("tkrSubChart");
  state.subChart = chart;
  syncTimeScales(state.tkrChart, chart);
  return chart;
}

function applyOverlay(view) {
  clearOverlays();
  state.activeOverlay = view;
  setActiveToggle(view);
  const data = state.analysis;
  if (!data || view === "none") return;

  const indicators = data.indicators;

  if (view === "orderblock") {
    const ob = indicators.OrderBlock;
    if (!ob || ob.error) { legend([{ color: "#999", label: ob ? ob.error : "no data" }]); return; }
    const obDates = (ob.series && ob.series.dates) || [];
    const zones = [];
    (ob.active_bullish_order_blocks || []).forEach(z => {
      const t = toBarTime(obDates, z.formed_bar);
      if (t != null) zones.push({ fromTime: t, top: z.zone_high, bottom: z.zone_low,
        fill: "rgba(14,157,110,0.15)", stroke: "rgba(14,157,110,0.55)" });
    });
    (ob.active_bearish_order_blocks || []).forEach(z => {
      const t = toBarTime(obDates, z.formed_bar);
      if (t != null) zones.push({ fromTime: t, top: z.zone_high, bottom: z.zone_low,
        fill: "rgba(224,35,63,0.15)", stroke: "rgba(224,35,63,0.55)" });
    });
    state.tkrZoneApi.setZones(zones);
    if (ob.entry_plan) {
      addPriceLine(ob.entry_plan.entry_price, "#b5790f", "Entry", false);
      addPriceLine(ob.entry_plan.stop_loss, "#e0233f", "Stop Loss", false);
      if (ob.entry_plan.take_profit) addPriceLine(ob.entry_plan.take_profit, "#0e9d6e", "Take Profit", false);
    }
    legend([
      { color: "#0e9d6e", label: "Bullish Order Block" },
      { color: "#e0233f", label: "Bearish Order Block" },
      { color: "#b5790f", label: "Entry / SL / TP" },
    ]);

  } else if (view === "fvg") {
    const ob = indicators.OrderBlock;
    if (!ob || ob.error) { legend([{ color: "#999", label: ob ? ob.error : "no data" }]); return; }
    const obDates = (ob.series && ob.series.dates) || [];
    const zones = [];
    (ob.active_fvgs || []).forEach(f => {
      const t = toBarTime(obDates, f.formed_bar);
      if (t == null) return;
      const bullish = f.direction === "bullish";
      zones.push({ fromTime: t, top: f.top, bottom: f.bottom,
        fill: bullish ? "rgba(14,157,110,0.15)" : "rgba(224,35,63,0.15)",
        stroke: bullish ? "rgba(14,157,110,0.55)" : "rgba(224,35,63,0.55)" });
    });
    state.tkrZoneApi.setZones(zones);
    legend([{ color: "#0e9d6e", label: "Bullish FVG" }, { color: "#e0233f", label: "Bearish FVG" }]);

  } else if (view === "bollinger") {
    const b = indicators.Bollinger;
    if (!b || b.error || !b.series) { legend([{ color: "#999", label: b ? b.error : "no data" }]); return; }
    const s = b.series;
    ["upper", "mid", "lower"].forEach((k, i) => {
      const color = ["#2563eb", "#6b7280", "#2563eb"][i];
      const line = state.tkrChart.addLineSeries({ color, lineWidth: 1 });
      line.setData(s.dates.map((d, idx) => ({ time: toUnixTime(d), value: s[k][idx] })).filter(p => p.value != null));
      state.tkrExtraSeries.push(line);
    });
    legend([{ color: "#2563eb", label: "Upper / Lower Band" }, { color: "#6b7280", label: "Middle (SMA20)" }]);

  } else if (view === "ichimoku") {
    const ic = indicators.Ichimoku;
    if (!ic || ic.error || !ic.series) { legend([{ color: "#999", label: ic ? ic.error : "no data" }]); return; }
    const s = ic.series;
    const specs = [["tenkan", "#2563eb"], ["kijun", "#e0233f"], ["span_a", "#0e9d6e"], ["span_b", "#f59e0b"]];
    specs.forEach(([k, color]) => {
      const line = state.tkrChart.addLineSeries({ color, lineWidth: 1 });
      line.setData(s.dates.map((d, idx) => ({ time: toUnixTime(d), value: s[k][idx] })).filter(p => p.value != null));
      state.tkrExtraSeries.push(line);
    });
    legend([
      { color: "#2563eb", label: "Tenkan-sen" }, { color: "#e0233f", label: "Kijun-sen" },
      { color: "#0e9d6e", label: "Senkou A" }, { color: "#f59e0b", label: "Senkou B" },
    ]);

  } else if (view === "sr") {
    const sr = indicators.SR;
    if (!sr || sr.error) { legend([{ color: "#999", label: sr ? sr.error : "no data" }]); return; }
    (sr.support_zones || []).slice(0, 5).forEach(z => addPriceLine(z.mid, "#0e9d6e", `S ${z.mid.toFixed(2)}`));
    (sr.resistance_zones || []).slice(0, 5).forEach(z => addPriceLine(z.mid, "#e0233f", `R ${z.mid.toFixed(2)}`));
    legend([{ color: "#0e9d6e", label: "Support" }, { color: "#e0233f", label: "Resistance" }]);

  } else if (view === "rsi") {
    const r = indicators.RSI;
    if (!r || r.error || !r.series) { legend([{ color: "#999", label: r ? r.error : "no data" }]); return; }
    const s = r.series;
    const markers = (s.divergences || []).map(d => ({
      time: toBarTime(s.dates, d.bar),
      position: d.direction === "BUY" ? "belowBar" : "aboveBar",
      color: d.direction === "BUY" ? "#0e9d6e" : "#e0233f",
      shape: d.direction === "BUY" ? "arrowUp" : "arrowDown",
      text: d.label + (d.strength === "STRONG" ? " ★" : ""),
    })).filter(m => m.time != null).sort((a, b) => a.time - b.time);
    state.tkrSeries.setMarkers(markers);

    const sub = openSubChart();
    const line = sub.addLineSeries({ color: "#7c3aed", lineWidth: 1.5 });
    line.setData(s.dates.map((d, idx) => ({ time: toUnixTime(d), value: s.rsi[idx] })).filter(p => p.value != null));
    line.createPriceLine({ price: 70, color: "#e0233f", lineWidth: 1, lineStyle: LWC.LineStyle.Dashed, axisLabelVisible: false, title: "70" });
    line.createPriceLine({ price: 30, color: "#0e9d6e", lineWidth: 1, lineStyle: LWC.LineStyle.Dashed, axisLabelVisible: false, title: "30" });
    state.subSeries.push(line);
    legend([{ color: "#7c3aed", label: "RSI (14)" }, { color: "#0e9d6e", label: "BUY divergence" }, { color: "#e0233f", label: "SELL divergence" }]);

  } else if (view === "macd") {
    const m = indicators.MACD;
    if (!m || m.error || !m.series) { legend([{ color: "#999", label: m ? m.error : "no data" }]); return; }
    const s = m.series;
    const markers = (s.divergences || []).map(d => ({
      time: toBarTime(s.dates, d.bar),
      position: d.direction === "BUY" ? "belowBar" : "aboveBar",
      color: d.direction === "BUY" ? "#0e9d6e" : "#e0233f",
      shape: d.direction === "BUY" ? "arrowUp" : "arrowDown",
      text: d.label + (d.strength === "STRONG" ? " ★" : ""),
    })).filter(mk => mk.time != null).sort((a, b) => a.time - b.time);
    state.tkrSeries.setMarkers(markers);

    const sub = openSubChart();
    const hist = sub.addHistogramSeries({ color: "#9ca3af" });
    hist.setData(s.dates.map((d, idx) => ({
      time: toUnixTime(d), value: s.hist[idx],
      color: (s.hist[idx] || 0) >= 0 ? "#86efac" : "#fca5a5",
    })).filter(p => p.value != null));
    const macdLine = sub.addLineSeries({ color: "#2563eb", lineWidth: 1.5 });
    macdLine.setData(s.dates.map((d, idx) => ({ time: toUnixTime(d), value: s.macd[idx] })).filter(p => p.value != null));
    const sigLine = sub.addLineSeries({ color: "#f59e0b", lineWidth: 1.5 });
    sigLine.setData(s.dates.map((d, idx) => ({ time: toUnixTime(d), value: s.signal[idx] })).filter(p => p.value != null));
    state.subSeries.push(hist, macdLine, sigLine);
    legend([
      { color: "#2563eb", label: "MACD" }, { color: "#f59e0b", label: "Signal" },
      { color: "#0e9d6e", label: "BUY divergence" }, { color: "#e0233f", label: "SELL divergence" },
    ]);
  }
}

// ── header: live ET clock + market session pill ────────────
// Heuristic only (regular NYSE/NASDAQ hours, no holiday calendar) —
// good enough for an at-a-glance "is the market open right now" cue.
function getETParts() {
  const now = new Date();
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  return et;
}

function updateHeaderClock() {
  const et = getETParts();
  const day = et.getDay(); // 0 Sun .. 6 Sat
  const hours = et.getHours() + et.getMinutes() / 60;
  const isWeekday = day >= 1 && day <= 5;
  const isOpen = isWeekday && hours >= 9.5 && hours < 16;

  const pill = document.getElementById("marketStatus");
  const text = document.getElementById("marketStatusText");
  if (pill && text) {
    pill.classList.toggle("open", isOpen);
    text.textContent = isOpen ? "Market Open" : "Market Closed";
  }
  const clock = document.getElementById("liveClock");
  if (clock) {
    clock.textContent = et.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }) + " ET";
  }
}
updateHeaderClock();
setInterval(updateHeaderClock, 1000);

// ── boot ───────────────────────────────────────────────────
initHome();