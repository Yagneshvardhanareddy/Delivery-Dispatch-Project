/*
  app.js - front end of the dispatch dashboard (plain JavaScript, no build step).

  Flow
    1. On load: GET /api/status  -> fill the city list and the data badge.
    2. Press "Run optimisation": POST /api/dispatch -> draw the map, table and charts.
    3. Dataset tab:      GET /api/eda         (loaded the first time the tab opens)
       How it works tab: GET /api/algorithms  (loaded the first time the tab opens)
*/
(() => {
  'use strict';

  const COLORS = {
    harbour: '#1F5E7A', harbourLight: '#7FA6B8', saffron: '#E0952B', ink: '#14222B',
    stone: '#5F6E74', rule: '#CBD3CE', chilli: '#B5402D', moss: '#3E7D55', baseline: '#9AA7AC',
  };
  const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 1) => Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtAuto = (n) => (Number.isInteger(n) ? String(n) : fmt(n, 2));

  let map = null;
  let mapLayer = null;
  let drawTimers = [];
  let appStatus = null;
  const loaded = { dataset: false, how: false };

  // ------------------------------------------------------------------ helpers
  async function getJSON(url, options) {
    const res = await fetch(url, options);
    let body = null;
    try { body = await res.json(); } catch (e) { /* not JSON */ }
    if (!res.ok) {
      let msg = `Request failed (${res.status}).`;
      if (body && typeof body.detail === 'string') msg = body.detail;
      else if (body && Array.isArray(body.detail)) {
        msg = body.detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
      }
      throw new Error(msg);
    }
    return body;
  }

  const baseLayout = (extra = {}) => Object.assign({
    margin: { l: 55, r: 15, t: 30, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Public Sans, system-ui, sans-serif', color: COLORS.ink, size: 12 },
    xaxis: { gridcolor: COLORS.rule, zerolinecolor: COLORS.rule },
    yaxis: { gridcolor: COLORS.rule, zerolinecolor: COLORS.rule },
  }, extra);
  const plotConfig = { displayModeBar: false, responsive: true };

  function showError(message) {
    const box = $('error-box');
    box.textContent = message;
    box.classList.toggle('hidden', !message);
  }

  // --------------------------------------------------------------------- tabs
  function switchTab(name) {
    ['dispatch', 'dataset', 'how'].forEach((t) => {
      $('tab-' + t).classList.toggle('hidden', t !== name);
      $('nav-' + t).setAttribute('aria-selected', String(t === name));
    });
    if (name === 'dispatch' && map) setTimeout(() => map.invalidateSize(), 50);
    if (name === 'dataset' && !loaded.dataset) loadDataset();
    if (name === 'how' && !loaded.how) loadAlgorithms();
  }

  // ---------------------------------------------------------------- start-up
  function linkRange(numberId, rangeId) {
    const num = $(numberId);
    const range = $(rangeId);
    range.addEventListener('input', () => { num.value = range.value; });
    num.addEventListener('input', () => { range.value = num.value; });
  }

  let routesLayer;
  function initMap() {
    map = L.map('map', { zoomControl: true }).setView([20.5, 78.9], 5);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
    }).addTo(map);
    routesLayer = L.layerGroup();
    mapLayer = L.layerGroup().addTo(map);
    const rt = document.getElementById('f-routes');
    const syncRoutes = () => { if (rt.checked) map.addLayer(routesLayer); else map.removeLayer(routesLayer); };
    rt.addEventListener('change', syncRoutes);
    syncRoutes();
  }

  async function init() {
    document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
    linkRange('f-orders', 'f-orders-range');
    linkRange('f-riders', 'f-riders-range');
    linkRange('f-zones', 'f-zones-range');
    $('run-btn').addEventListener('click', run);
    initMap();

    try {
      appStatus = await getJSON('/api/status');
    } catch (err) {
      $('data-badge').textContent = 'Server not reachable';
      $('data-badge').classList.add('warn');
      showError('Could not reach the server: ' + err.message + ' Start it with: uvicorn app:app');
      return;
    }

    const badge = $('data-badge');
    if (appStatus.source === 'synthetic') {
      badge.textContent = 'Test data in use. Put train.csv in the data folder and restart.';
      badge.classList.add('warn');
    } else {
      badge.textContent = `Kaggle train.csv, ${fmt(appStatus.rows_clean, 0)} clean orders`;
    }

    $('f-city').innerHTML = appStatus.cities
      .map((c) => `<option value="${esc(c.code)}">${esc(c.code)} (${fmt(c.orders, 0)} orders)</option>`).join('');

    const nn = $('f-nn');
    if (appStatus.nn_available) {
      nn.disabled = false;
      $('nn-check').classList.remove('disabled');
      $('nn-hint').textContent = 'A trained model was found.';
    } else {
      $('nn-hint').textContent = 'Not available. Install torch and run python train_nn.py first.';
    }

    run();   // show a first result straight away
  }

  // -------------------------------------------------------------------- run
  function readConfig() {
    const num = (id) => Number($(id).value);
    const cfg = {
      city: $('f-city').value || null,
      n_orders: num('f-orders'), n_riders: num('f-riders'), n_zones: num('f-zones'), seed: num('f-seed'),
      rebalance: $('f-rebalance').checked, use_nn: $('f-nn').checked, lp_method: $('f-lp').value,
      revenue_per_order: num('f-revenue'), rider_cost_per_min: num('f-cost'),
      delay_target_min: num('f-target'), delay_penalty_per_min: num('f-penalty'),
      max_delivery_min: num('f-maxmin'), min_service_share: num('f-share'),
    };
    const bad = Object.entries(cfg).find(([, v]) => typeof v === 'number' && Number.isNaN(v));
    if (bad) throw new Error('Enter a number in every field. Check: ' + bad[0]);
    return cfg;
  }

  async function run() {
    const btn = $('run-btn');
    showError('');
    let cfg;
    try { cfg = readConfig(); } catch (err) { showError(err.message); return; }

    btn.disabled = true;
    $('run-status').textContent = 'Running the optimisation...';
    try {
      const result = await getJSON('/api/dispatch', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
      });
      render(result);
      $('run-status').textContent = `Optimisation finished in ${fmt(result.runtime_ms, 0)} ms.`;
    } catch (err) {
      $('run-status').textContent = '';
      showError(err.message);
    } finally {
      btn.disabled = false;
    }
  }

  // ----------------------------------------------------------------- render
  function render(result) {
    renderHeadline(result);
    drawMap(result);
    renderStrategies(result);
    renderZones(result);
    renderLP(result);
    renderNN(result);
  }

  function renderHeadline(result) {
    const final = result.strategies.find((s) => s.name === result.final_strategy);
    const left = result.n_orders - final.served;
    let sentence = `${final.name} serves ${final.served} of ${result.n_orders} orders in ${fmt(final.avg_minutes, 1)} minutes on average`;
    if (left === 0) {
      // Same orders in every strategy, so the comparison is fair.
      const pct = final.improvement_vs_nearest_pct;
      if (Math.abs(pct) < 0.5) sentence += ', about the same as nearest rider first';
      else if (pct > 0) sentence += `, ${fmt(pct, 1)}% faster than nearest rider first`;
      else sentence += `, ${fmt(-pct, 1)}% slower than nearest rider first`;
    }
    $('headline').textContent = sentence + '.';

    const c = result.config;
    $('subline').textContent =
      `City ${c.city}, ${result.n_orders} orders, ${result.n_riders} riders, ${c.n_zones} zones, seed ${c.seed}.` +
      (left > 0
        ? ` ${left} orders have no rider. The optimiser chooses which orders to serve, the baselines do not, so their averages are not directly comparable.`
        : '') +
      ' Times are model estimates.';
  }

  // --- map -------------------------------------------------------------------
  function drawMap(result) {
    drawTimers.forEach(clearTimeout);
    drawTimers = [];
    mapLayer.clearLayers();
    routesLayer.clearLayers();
    const points = [];

    result.zones.centroids.forEach((c, i) => {
      points.push(c);
      L.circleMarker(c, { radius: 18, color: COLORS.stone, weight: 1.5, dashArray: '4 4', fillOpacity: 0.04 })
        .bindTooltip(`Zone ${i + 1}`, { permanent: true, direction: 'center', className: 'zone-label' })
        .addTo(mapLayer);
    });

    const pairs = result.assignments;
    pairs.forEach((a) => {
      points.push(a.rider, a.restaurant, a.customer);
      L.circleMarker(a.rider, { radius: 6, color: '#fff', weight: 2, fillColor: COLORS.harbour, fillOpacity: 1 })
        .bindPopup(`<b>Rider</b> ${esc(a.rider_id)}<br>Estimated ${fmt(a.est_minutes, 1)} min`).addTo(mapLayer);
      L.circleMarker(a.restaurant, { radius: 6, color: '#fff', weight: 2, fillColor: COLORS.saffron, fillOpacity: 1 })
        .bindPopup(`<b>Order</b> ${esc(a.order_id)}<br>${esc(a.traffic)} traffic, ${esc(a.weather)}`).addTo(mapLayer);
      L.circleMarker(a.customer, { radius: 4, color: '#fff', weight: 1.5, fillColor: COLORS.ink, fillOpacity: 1 })
        .addTo(mapLayer);
    });

    // The routes draw one after another, so you can follow each rider's path.
    const step = REDUCED_MOTION ? 0 : Math.min(45, 1400 / Math.max(pairs.length, 1));
    pairs.forEach((a, i) => {
      const draw = () => {
        const l1 = L.polyline([a.rider, a.restaurant], { color: COLORS.harbour, weight: 1.5, dashArray: '4 6', opacity: 0.45 }).addTo(routesLayer);
        const l2 = L.polyline([a.restaurant, a.customer], { color: COLORS.saffron, weight: 1.5, opacity: 0.4 }).addTo(routesLayer);
        [l1, l2].forEach((l) => {
          l.on('mouseover', () => { l1.setStyle({ weight: 4, opacity: 1 }); l2.setStyle({ weight: 4, opacity: 1 }); l1.bringToFront(); l2.bringToFront(); });
          l.on('mouseout', () => { l1.setStyle({ weight: 1.5, opacity: 0.45 }); l2.setStyle({ weight: 1.5, opacity: 0.4 }); });
        });
      };
      if (step === 0) draw();
      else drawTimers.push(setTimeout(draw, i * step));
    });

    map.invalidateSize();
    if (points.length) map.fitBounds(L.latLngBounds(points).pad(0.15));
  }

  // --- strategy table and chart ---------------------------------------------
  function renderStrategies(result) {
    const rows = result.strategies;
    const maxAvg = Math.max(...rows.map((s) => s.avg_minutes)) || 1;
    $('strategy-body').innerHTML = rows.map((s) => {
      const isFinal = s.name === result.final_strategy;
      const vs = s.improvement_vs_nearest_pct;
      let vsText = 'baseline';
      let vsClass = '';
      if (s.name !== 'Nearest rider first') {
        vsText = `${vs > 0 ? '+' : ''}${fmt(vs, 1)}%`;
        vsClass = vs > 0.05 ? 'good' : (vs < -0.05 ? 'bad' : '');
      }
      return `<tr class="${isFinal ? 'final' : ''}">
        <td>${esc(s.name)}</td>
        <td class="bar-cell"><div class="val">${fmt(s.avg_minutes, 1)}</div><div class="bar" style="width:${(100 * s.avg_minutes / maxAvg).toFixed(1)}%"></div></td>
        <td class="num">${fmt(s.total_minutes, 0)}</td>
        <td class="num">${fmtAuto(Math.round(s.delayed_orders * 10) / 10)}</td>
        <td class="num">${fmt(s.profit, 0)}</td>
        <td class="num ${vsClass}">${vsText}</td></tr>`;
    }).join('');

    const names = rows.map((s) => s.name);
    const colours = rows.map((s) => {
      if (s.name === result.final_strategy) return COLORS.harbour;
      if (s.name === 'Random assignment' || s.name === 'Nearest rider first') return COLORS.baseline;
      return COLORS.harbourLight;
    });
    Plotly.react('chart-strategies', [{
      type: 'bar', orientation: 'h', x: rows.map((s) => s.avg_minutes), y: names,
      marker: { color: colours }, text: rows.map((s) => fmt(s.avg_minutes, 1)), textposition: 'outside',
      hovertemplate: '%{y}: %{x:.1f} min<extra></extra>',
    }], baseLayout({
      margin: { l: 230, r: 40, t: 10, b: 45 },
      xaxis: { title: 'Average delivery time per order (minutes, estimated)', gridcolor: COLORS.rule, rangemode: 'tozero' },
      yaxis: { autorange: 'reversed' },
    }), plotConfig);
  }

  // --- zones / transportation ------------------------------------------------
  function renderZones(result) {
    const z = result.zones;
    const k = z.centroids.length;
    const labels = z.centroids.map((_, i) => `Zone ${i + 1}`);

    $('zones-note').textContent =
      `KMeans cuts the city into ${k} zone${k === 1 ? '' : 's'}. Supply is the number of riders waiting in a zone, ` +
      'demand is the number of orders picked up there, and the cost is the distance between zone centres.';

    const dummyText = { column: 'extra destination (more riders than orders)', row: 'extra source (more orders than riders)' }[z.dummy] || 'none, supply equals demand';
    $('zones-kv').innerHTML = `
      <dt>Riders per zone</dt><dd>${z.riders_per_zone.map((v, i) => `${i + 1}: ${v}`).join(', ')}</dd>
      <dt>Orders per zone</dt><dd>${z.orders_per_zone.map((v, i) => `${i + 1}: ${v}`).join(', ')}</dd>
      <dt>Vogel start plan</dt><dd>${fmt(z.initial_vam_cost_km, 1)} km of riding</dd>
      <dt>Optimal plan (MODI)</dt><dd>${fmt(z.transport_cost_km, 1)} km after ${z.modi_iterations} improvement step${z.modi_iterations === 1 ? '' : 's'}</dd>
      <dt>Riders moved</dt><dd>${z.riders_moved}${result.config.rebalance ? '' : ' (repositioning is switched off)'}</dd>
      <dt>Dummy node</dt><dd>${esc(dummyText)}</dd>`;

    if (k < 2) {
      Plotly.purge('chart-flow');
      $('chart-flow').innerHTML = '<p class="section-note">With one zone there is nothing to move.</p>';
      return;
    }
    Plotly.react('chart-flow', [heatmap(z.flow, labels.map((l) => 'to ' + l), labels.map((l) => 'from ' + l))],
      baseLayout({
        margin: { l: 80, r: 10, t: 30, b: 50 }, title: { text: 'Riders sent between zones', font: { size: 13 } },
        yaxis: { autorange: 'reversed' }
      }), plotConfig);
  }

  function heatmap(matrix, xLabels, yLabels) {
    return {
      type: 'heatmap', z: matrix, x: xLabels, y: yLabels, showscale: false,
      colorscale: [[0, '#F2F4F1'], [1, COLORS.harbour]], xgap: 2, ygap: 2,
      texttemplate: '%{z:.1f}', hovertemplate: '%{y} %{x}: %{z:.2f}<extra></extra>',
    };
  }

  // --- profit LP -------------------------------------------------------------
  function renderLP(result) {
    const lp = result.profit_plan;
    const k = result.zones.centroids.length;
    const method = lp.method === 'big_m' ? 'Big-M' : 'Two-Phase';
    $('lp-note').textContent =
      `Maximise revenue minus rider cost minus late penalty over zone-to-zone deliveries, solved by our own ${method} Simplex. ` +
      'Rider capacity, the longest allowed trip and a minimum share of served orders in every zone are the constraints.';

    const statusText = { optimal: 'Optimal', infeasible: 'No feasible plan', no_variables: 'No allowed deliveries' }[lp.status] || lp.status;
    $('lp-kv').innerHTML = `
      <dt>Status</dt><dd>${esc(statusText)}</dd>
      <dt>Expected profit</dt><dd>Rs ${fmt(lp.objective, 0)} for the planned deliveries</dd>
      <dt>Minimum share served</dt><dd>${fmt(lp.share_used * 100, 0)}% of each zone's orders${lp.share_used < result.config.min_service_share ? ' (relaxed, not enough riders for the full target)' : ''}</dd>
      <dt>Simplex iterations</dt><dd>${lp.iterations}</dd>`;

    if (lp.status !== 'optimal') {
      Plotly.purge('chart-plan');
      $('chart-plan').innerHTML = '<p class="section-note">There is no plan to draw. Raise the longest allowed trip or add riders.</p>';
      return;
    }
    const labels = Array.from({ length: k }, (_, i) => `Zone ${i + 1}`);
    Plotly.react('chart-plan', [heatmap(lp.plan, labels.map((l) => 'riders from ' + l), labels.map((l) => 'orders in ' + l))],
      baseLayout({
        margin: { l: 90, r: 10, t: 30, b: 70 }, title: { text: 'Deliveries planned by the LP', font: { size: 13 } },
        yaxis: { autorange: 'reversed' }
      }), plotConfig);
  }

  // --- neural network ----------------------------------------------------------
  function renderNN(result) {
    const nn = result.neural_network;
    $('nn-box').classList.toggle('hidden', !nn);
    if (!nn) return;
    const m = nn.metrics;
    $('nn-kv').innerHTML = `
      <dt>Error of the neural network</dt><dd>${fmt(m.nn_mae_min, 2)} min on held-out real orders</dd>
      <dt>Error of the formula</dt><dd>${fmt(m.formula_mae_min, 2)} min on the same orders</dd>
      <dt>Assignment cost, network view</dt><dd>${fmt(nn.total_minutes_under_nn_costs, 0)} min (Hungarian on network costs)</dd>
      <dt>Formula plan, network view</dt><dd>${fmt(nn.total_minutes_hungarian_under_nn_costs, 0)} min (Hungarian on formula costs)</dd>`;
    $('nn-verdict').textContent = m.nn_mae_min < 0.97 * m.formula_mae_min
      ? 'The network predicts held-out delivery times better than the formula, so its cost matrix is a reasonable alternative.'
      : 'The network is not clearly better than the formula on held-out orders. Report this honestly: the simpler model is enough.';
  }

  // ----------------------------------------------------------------- dataset
  async function loadDataset() {
    loaded.dataset = true;
    let eda;
    try { eda = await getJSON('/api/eda'); } catch (err) { $('ds-sub').textContent = err.message; return; }

    $('ds-sub').textContent = appStatus && appStatus.source === 'synthetic'
      ? 'These numbers come from random test data, not from the Kaggle file.'
      : 'Cleaned from train.csv. Delivery time includes preparation and handover, not only riding.';
    $('ds-stats').innerHTML = [
      [fmt(eda.rows, 0), `clean orders (from ${fmt(eda.raw_rows, 0)} raw rows)`],
      [fmt(eda.mean_minutes, 1), 'average delivery time, minutes'],
      [fmt(eda.mean_distance_km, 1), 'average restaurant to customer distance, km'],
      [String(Object.keys(eda.cities).length), 'cities'],
    ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join('');

    const bar = (id, group, xTitle, colour) => Plotly.newPlot(id, [{
      type: 'bar', x: group.labels, y: group.mean_minutes, marker: { color: colour || COLORS.harbour },
      customdata: group.count, hovertemplate: '%{x}: %{y:.1f} min (%{customdata} orders)<extra></extra>',
    }], baseLayout({ xaxis: { title: xTitle, gridcolor: COLORS.rule }, yaxis: { title: 'Average minutes', gridcolor: COLORS.rule, rangemode: 'tozero' } }), plotConfig);

    bar('chart-dist', eda.distance_vs_time, 'Restaurant to customer distance (km)');
    bar('chart-traffic', eda.traffic_vs_time, 'Road traffic density', COLORS.saffron);
    bar('chart-weather', eda.weather_vs_time, 'Weather');
    Plotly.newPlot('chart-hour', [{
      type: 'scatter', mode: 'lines+markers', x: eda.hour_vs_time.labels, y: eda.hour_vs_time.mean_minutes,
      line: { color: COLORS.harbour, width: 2 }, marker: { color: COLORS.harbour },
      hovertemplate: 'Hour %{x}: %{y:.1f} min<extra></extra>',
    }], baseLayout({ xaxis: { title: 'Hour the order was placed', gridcolor: COLORS.rule }, yaxis: { title: 'Average minutes', gridcolor: COLORS.rule, rangemode: 'tozero' } }), plotConfig);

    const cities = Object.entries(eda.cities).sort((a, b) => b[1] - a[1]).slice(0, 20);
    Plotly.newPlot('chart-cities', [{
      type: 'bar', x: cities.map((c) => c[0]), y: cities.map((c) => c[1]), marker: { color: COLORS.harbourLight },
      hovertemplate: '%{x}: %{y} orders<extra></extra>',
    }], baseLayout({ xaxis: { title: 'City code (from the rider ID)', gridcolor: COLORS.rule }, yaxis: { title: 'Orders', gridcolor: COLORS.rule } }), plotConfig);

    const tm = appStatus ? appStatus.time_model : null;
    if (tm) {
      const list = (obj) => `<ul class="mult-list">${Object.entries(obj).sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `<li>${esc(k)}: <b class="${v > 1.02 ? 'bad' : (v < 0.98 ? 'good' : '')}">x${fmt(v, 2)}</b></li>`).join('')}</ul>`;
      $('model-box').innerHTML = `
        <dl class="kv"><dt>Fixed time</dt><dd>${fmt(tm.fixed_min, 1)} min per order</dd>
        <dt>Delivery leg</dt><dd>${fmt(tm.min_per_km, 2)} min per km of restaurant to customer distance (fitted)</dd>
        <dt>Pickup leg</dt><dd>${fmt(tm.pickup_min_per_km || tm.min_per_km, 1)} min per km for the rider's trip to the restaurant (assumed, the data has no rider positions)</dd>
        <dt>Rows used</dt><dd>${fmt(tm.n_rows_used, 0)}</dd></dl>
        <div class="two-col"><div><b>Traffic</b>${list(tm.traffic_mult)}</div><div><b>Weather</b>${list(tm.weather_mult)}</div></div>
        <div style="margin-top:1rem"><b>Vehicle</b>${list(tm.vehicle_mult)}</div>`;
    }
  }

  // ----------------------------------------------------------- how it works
  function matrixTable(matrix, highlight = [], rowPrefix = 'Row ', colPrefix = 'Col ') {
    const hot = new Set(highlight.map(([r, c]) => `${r}-${c}`));
    const head = `<tr><th></th>${matrix[0].map((_, c) => `<th class="num">${colPrefix}${c + 1}</th>`).join('')}</tr>`;
    const body = matrix.map((row, r) => `<tr><td>${rowPrefix}${r + 1}</td>${row.map((v, c) =>
      `<td class="num" ${hot.has(`${r}-${c}`) ? 'style="font-weight:700;background:#FBEBD0"' : ''}>${fmtAuto(v)}</td>`).join('')}</tr>`).join('');
    return `<table class="data" style="width:auto">${head}${body}</table>`;
  }

  async function loadAlgorithms() {
    loaded.how = true;
    let a;
    try { a = await getJSON('/api/algorithms'); } catch (err) {
      ['alg-hungarian', 'alg-transport', 'alg-simplex'].forEach((id) => { $(id).textContent = err.message; });
      return;
    }
    $('alg-hungarian').innerHTML = `<b>Example:</b> cost of giving each order to each rider (minutes).
      ${matrixTable(a.hungarian.cost, a.hungarian.pairs, 'Order ', 'Rider ')}
      <p style="margin:.5rem 0 0">Optimal assignment (highlighted): total ${fmtAuto(a.hungarian.total)} minutes.</p>`;

    const t = a.transportation;
    $('alg-transport').innerHTML = `<b>Example:</b> supply ${t.supply.join(', ')}, demand ${t.demand.join(', ')}. Unit costs:
      ${matrixTable(t.cost, [], 'Source ', 'Destination ')}
      <p style="margin:.5rem 0">Optimal shipments:</p>
      ${matrixTable(t.flow, [], 'Source ', 'Destination ')}
      <p style="margin:.5rem 0 0">Vogel start cost ${fmtAuto(t.initial_cost)}, optimal cost ${fmtAuto(t.total_cost)} after ${t.iterations} MODI step${t.iterations === 1 ? '' : 's'}.</p>`;

    const s = a.simplex;
    $('alg-simplex').innerHTML = `<b>Maximise:</b> ${esc(s.maximise.text)}<br>
      Solution x = ${s.maximise.x.map(fmtAuto).join(', ')}, objective ${fmtAuto(s.maximise.objective)} (${s.maximise.iterations} iterations).<br><br>
      <b>Minimise with "at least" rows:</b> ${esc(s.minimise.text)}<br>
      Two-Phase: x = ${s.minimise.two_phase.x.map(fmtAuto).join(', ')}, objective ${fmtAuto(s.minimise.two_phase.objective)} (${s.minimise.two_phase.iterations} iterations).<br>
      Big-M: x = ${s.minimise.big_m.x.map(fmtAuto).join(', ')}, objective ${fmtAuto(s.minimise.big_m.objective)} (${s.minimise.big_m.iterations} iterations).`;
  }

  document.addEventListener('DOMContentLoaded', init);
})();