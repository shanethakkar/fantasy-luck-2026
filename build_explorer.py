#!/usr/bin/env python3
"""Build the self-contained 2026 fantasy luck explorer HTML."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
data = json.loads((OUT / "explorer_data.json").read_text())
payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")

html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026 Fantasy Luck Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,620&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #f3efe6;
    --bg-2: #eae4d6;
    --ink: #16171c;
    --muted: #5b5e66;
    --card: #fffcf6;
    --line: #ddd6c7;
    --accent: #0f5c4c;
    --accent-soft: #e3f0eb;
    --good: #0b7a4b;
    --good-bg: #e4f5ec;
    --bad: #c13b2a;
    --bad-bg: #fbecea;
    --qb: #5b4fcf;
    --rb: #c05621;
    --wr: #1a6f9e;
    --te: #8a6a16;
    --shadow: 0 10px 30px rgba(22, 23, 28, 0.06);
    --radius: 16px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink); }
  body {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.45;
  }
  h1, h2, .serif { font-family: Fraunces, Georgia, serif; font-optical-sizing: auto; }
  a { color: var(--accent); }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 28px 22px 64px; }
  header.hero { margin-bottom: 22px; }
  header.hero h1 {
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 620;
    letter-spacing: -0.03em;
    margin: 0 0 8px;
    line-height: 1.05;
  }
  .lede { color: var(--muted); max-width: 46rem; margin: 0; }
  .lede strong { color: var(--ink); font-weight: 600; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 14px 18px;
    align-items: end;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px 18px;
    margin: 22px 0 18px;
  }
  .ctrl { display: flex; flex-direction: column; gap: 7px; min-width: 0; }
  .ctrl label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .pills { display: flex; flex-wrap: wrap; gap: 6px; }
  .pill, .tog button {
    appearance: none; border: 1px solid var(--line); background: #fff;
    color: var(--ink); border-radius: 999px; padding: 6px 12px;
    font: 600 13px/1 "IBM Plex Sans", sans-serif; cursor: pointer;
  }
  .pill:hover, .tog button:hover { border-color: #c9c0ad; }
  .pill.on { background: var(--ink); color: #fff; border-color: var(--ink); }
  .pill.on.QB { background: var(--qb); border-color: var(--qb); }
  .pill.on.RB { background: var(--rb); border-color: var(--rb); }
  .pill.on.WR { background: var(--wr); border-color: var(--wr); }
  .pill.on.TE { background: var(--te); border-color: var(--te); }
  .tog { display: flex; background: var(--bg-2); padding: 3px; border-radius: 999px; }
  .tog button { border: 0; background: transparent; padding: 7px 12px; }
  .tog button.on { background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .slider-row { display: flex; align-items: center; gap: 10px; min-width: 240px; }
  input[type=range] { width: 180px; accent-color: var(--accent); }
  #oppVal { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 2ch; }
  .hint { font-size: 12px; color: var(--muted); }
  #search {
    border: 1px solid var(--line); border-radius: 999px; padding: 8px 14px;
    font: 15px "IBM Plex Sans", sans-serif; min-width: 200px; background: #fff;
  }
  #search:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
    gap: 16px;
  }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px 14px 10px;
    min-width: 0;
  }
  .card h2 { margin: 0 0 4px; font-size: 1.25rem; font-weight: 620; letter-spacing: -0.02em; }
  .card .sub { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
  #scatter, #weekly { width: 100%; }
  #scatter { height: 520px; }
  #weekly { height: 220px; }
  .weekly-card { margin-top: 16px; display: none; }
  .weekly-card.show { display: block; }
  .player-chip { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
  .player-chip .name { font-family: Fraunces, serif; font-size: 1.15rem; }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin: 0 0 6px; }
  .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
  .legend .QB i { background: var(--qb); }
  .legend .RB i { background: var(--rb); }
  .legend .WR i { background: var(--wr); }
  .legend .TE i { background: var(--te); }
  .table-wrap { overflow: auto; max-height: 760px; border-radius: 12px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  th, td { padding: 7px 8px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), td.pos, th.pos { text-align: left; }
  th {
    position: sticky; top: 0; background: var(--card);
    font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); cursor: pointer; font-weight: 600; z-index: 1;
  }
  th:hover { color: var(--ink); }
  th .arrow { font-size: 9px; margin-left: 2px; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: var(--bg); }
  tbody tr.sel { background: var(--accent-soft); }
  .pos-tag {
    display: inline-block; font-size: 10px; font-weight: 700;
    letter-spacing: 0.06em; padding: 1px 6px; border-radius: 999px; color: #fff;
  }
  .pos-tag.QB { background: var(--qb); }
  .pos-tag.RB { background: var(--rb); }
  .pos-tag.WR { background: var(--wr); }
  .pos-tag.TE { background: var(--te); }
  .pos { color: var(--good); font-weight: 600; }
  .neg { color: var(--bad); font-weight: 600; }
  .count { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
  .method {
    margin-top: 22px; background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 8px 18px 4px; box-shadow: var(--shadow);
  }
  .method summary {
    cursor: pointer; font-weight: 600; padding: 12px 0; list-style: none;
  }
  .method summary::-webkit-details-marker { display: none; }
  .method summary:before { content: "▸ "; color: var(--accent); }
  .method[open] summary:before { content: "▾ "; }
  .method .body { color: var(--muted); padding: 0 0 16px; max-width: 58rem; }
  .method .body p { margin: 0 0 10px; }
  .cals { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 12px 0; }
  .cal { background: var(--bg); border-radius: 10px; padding: 8px 10px; }
  .cal b { display: block; font-size: 13px; color: var(--ink); }
  .cal span { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
  .warn { display: none; background: var(--bad-bg); color: var(--bad); padding: 10px 14px; border-radius: 10px; }
  footer { margin-top: 18px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>2026 Fantasy Luck Explorer</h1>
      <p class="lede">Expected vs actual PPR through <strong>Sunday, November 15, 2026</strong> (Weeks 1–10). Week 10 Monday Night Football, Chargers at Ravens, is not included. A league-average play in the same situation is the baseline — this is opportunity, not a talent rating.</p>
    </header>

    <div id="cdnWarn" class="warn">The chart library failed to load. Check your network connection and refresh.</div>

    <div class="controls">
      <div class="ctrl">
        <label>Position</label>
        <div class="pills" id="posPills"></div>
      </div>
      <div class="ctrl">
        <label>Luck view</label>
        <div class="tog" id="modeTog">
          <button type="button" data-mode="fp" class="on">Full xPPR</button>
          <button type="button" data-mode="td">TD only</button>
        </div>
      </div>
      <div class="ctrl">
        <label>Minimum opportunities</label>
        <div class="slider-row">
          <input type="range" id="opp" min="0" max="300" step="5" value="25">
          <span id="oppVal">25</span>
        </div>
        <div class="hint" id="oppHint">Rushes + targets (pass attempts for QBs)</div>
      </div>
      <div class="ctrl">
        <label>Find a player</label>
        <input id="search" type="search" placeholder="Search name or team" autocomplete="off">
      </div>
    </div>

    <div class="grid">
      <div>
        <section class="card">
          <h2 id="chartTitle">Actual vs expected PPR</h2>
          <p class="sub" id="chartSub">Points above the diagonal have outproduced their opportunity. Click a point to see the weekly line.</p>
          <div class="legend" id="legend"></div>
          <div id="scatter"></div>
        </section>
        <section class="card weekly-card" id="weeklyCard">
          <div class="player-chip">
            <div class="name" id="wkName"></div>
            <div class="hint" id="wkMeta"></div>
          </div>
          <div id="weekly"></div>
        </section>
      </div>
      <section class="card">
        <h2>Leaderboard</h2>
        <p class="count" id="count"></p>
        <div class="table-wrap">
          <table>
            <thead id="thead"></thead>
            <tbody id="tbody"></tbody>
          </table>
        </div>
      </section>
    </div>

    <details class="method">
      <summary>Methodology and validation</summary>
      <div class="body" id="methodBody"></div>
    </details>
    <footer>Data: nflverse / nflfastR play-by-play (CC BY 4.0). Expected points from a transparent gradient-boosting model trained on 2023–2025 regular-season plays. Updated from a file last written November 16, 2026, 1:32 PM America/Chicago.</footer>
  </div>

<script>
const DATA = ''' + payload + r''';
(function () {
  "use strict";
  if (typeof Plotly === "undefined") {
    document.getElementById("cdnWarn").style.display = "block";
  }

  const POS_COLORS = { QB: "#5b4fcf", RB: "#c05621", WR: "#1a6f9e", TE: "#8a6a16" };
  const POS_DEFAULT_MIN = { ALL: 25, QB: 150, RB: 40, WR: 25, TE: 25 };
  const state = { pos: "ALL", mode: "fp", minOpps: 25, sortKey: "dfp", sortDir: -1, q: "", selected: null };

  const players = DATA.players;
  const weeksById = {};
  DATA.weeks.forEach(function (w) {
    if (!weeksById[w.id]) weeksById[w.id] = [];
    weeksById[w.id].push(w);
  });
  Object.keys(weeksById).forEach(function (id) {
    weeksById[id].sort(function (a, b) { return a.w - b.w; });
  });

  const posPills = document.getElementById("posPills");
  ["ALL", "QB", "RB", "WR", "TE"].forEach(function (p) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "pill" + (p === "ALL" ? " on" : "");
    if (p !== "ALL") b.classList.add(p);
    b.textContent = p === "ALL" ? "All" : p;
    b.addEventListener("click", function () {
      state.pos = p;
      state.minOpps = POS_DEFAULT_MIN[p];
      document.getElementById("opp").value = String(state.minOpps);
      document.querySelectorAll("#posPills .pill").forEach(function (el) { el.classList.remove("on"); });
      b.classList.add("on");
      render();
    });
    posPills.appendChild(b);
  });

  document.querySelectorAll("#modeTog button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      state.mode = btn.getAttribute("data-mode");
      state.sortKey = state.mode === "fp" ? "dfp" : "dtd";
      state.sortDir = -1;
      document.querySelectorAll("#modeTog button").forEach(function (el) { el.classList.remove("on"); });
      btn.classList.add("on");
      render();
    });
  });

  document.getElementById("opp").addEventListener("input", function (e) {
    state.minOpps = Number(e.target.value);
    render();
  });
  document.getElementById("search").addEventListener("input", function (e) {
    state.q = e.target.value.trim().toLowerCase();
    render();
  });

  function luckColor(v) {
    if (v > 0.05) return "pos";
    if (v < -0.05) return "neg";
    return "";
  }
  function fmt(n, d) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    const v = Number(n);
    const s = v.toFixed(d);
    return (v > 0 && d > 0 ? "+" : "") + s;
  }
  function fmtRaw(n, d) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n).toFixed(d);
  }

  function filtered() {
    const q = state.q;
    return players.filter(function (p) {
      if (state.pos !== "ALL" && p.pos !== state.pos) return false;
      if (p.opps < state.minOpps) return false;
      if (q) {
        const hay = (p.name + " " + p.team + " " + p.pos).toLowerCase();
        if (hay.indexOf(q) === -1) return false;
      }
      return true;
    });
  }

  function sortRows(rows) {
    const k = state.sortKey;
    const dir = state.sortDir;
    return rows.slice().sort(function (a, b) {
      const av = a[k], bv = b[k];
      if (typeof av === "string") return dir * av.localeCompare(bv);
      return dir * ((av || 0) - (bv || 0));
    });
  }

  function render() {
    document.getElementById("oppVal").textContent = String(state.minOpps);
    document.getElementById("oppHint").textContent = state.pos === "QB"
      ? "Pass attempts (non-sack throws)"
      : (state.pos === "ALL"
        ? "Rushes + targets; pass attempts when the player is a QB"
        : "Rushes + targets");

    const rows = sortRows(filtered());
    document.getElementById("count").textContent = rows.length + " players  ·  click a header to sort  ·  click a row or chart point for the weekly line  ·  scroll the table sideways for yards, catch, and INT luck";

    document.getElementById("chartTitle").textContent = state.mode === "fp"
      ? "Actual vs expected PPR"
      : "Actual vs expected touchdowns";
    document.getElementById("chartSub").textContent = state.mode === "fp"
      ? "Points above the diagonal have outproduced their opportunity. Click a point to see the weekly line."
      : "Touchdowns above the diagonal are running hot relative to opportunity. Click a point for the weekly line.";

    drawScatter(rows);
    drawTable(rows);
    renderMethod();
    if (state.selected) drawWeekly(state.selected);
  }

  function drawScatter(rows) {
    const el = document.getElementById("scatter");
    if (typeof Plotly === "undefined") {
      el.innerHTML = "";
      return;
    }
    const xKey = state.mode === "fp" ? "xfp" : "xtd";
    const yKey = state.mode === "fp" ? "fp" : "td";
    const positions = ["QB", "RB", "WR", "TE"];
    const traces = positions.map(function (pos) {
      const sub = rows.filter(function (p) { return p.pos === pos; });
      const selected = sub.map(function (p) { return p.id === state.selected ? 12 : 8; });
      return {
        type: "scatter",
        mode: "markers",
        name: pos,
        x: sub.map(function (p) { return p[xKey]; }),
        y: sub.map(function (p) { return p[yKey]; }),
        customdata: sub.map(function (p) {
          return [p.id, p.name, p.team, p.pos, p.fp, p.xfp, p.dfp, p.td, p.xtd, p.dtd, p.ltd, p.lyd, p.lrec, p.lint, p.g, p.opps];
        }),
        marker: {
          color: POS_COLORS[pos],
          size: selected,
          opacity: 0.82,
          line: { width: sub.map(function (p) { return p.id === state.selected ? 2 : 0; }), color: "#16171c" }
        },
        hovertemplate: "%{customdata[1]} (%{customdata[2]}, %{customdata[3]})<br>" +
          (state.mode === "fp"
            ? "PPR %{customdata[4]:.1f} vs xPPR %{customdata[5]:.1f} (%{customdata[6]:+.1f})<br>TD %{customdata[7]:.0f} vs xTD %{customdata[8]:.1f}<br>TD luck %{customdata[10]:+.1f} · yards %{customdata[11]:+.1f} · rec %{customdata[12]:+.1f} · INT %{customdata[13]:+.1f}"
            : "TD %{customdata[7]:.1f} vs xTD %{customdata[8]:.1f} (%{customdata[9]:+.1f})<br>PPR %{customdata[4]:.1f} vs xPPR %{customdata[5]:.1f} (%{customdata[6]:+.1f})") +
          "<br>%{customdata[14]} games, %{customdata[15]} opps<extra></extra>"
      };
    });
    let lo = Infinity, hi = -Infinity;
    rows.forEach(function (p) {
      lo = Math.min(lo, p[xKey], p[yKey]);
      hi = Math.max(hi, p[xKey], p[yKey]);
    });
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    const pad = Math.max(state.mode === "fp" ? 8 : 0.5, (hi - Math.min(lo, 0)) * 0.05);
    if (lo >= 0) lo = 0;
    else lo = lo - pad;
    hi = hi + pad;
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Expected = actual",
      x: [lo, hi],
      y: [lo, hi],
      line: { color: "#8a8578", width: 1, dash: "dot" },
      hoverinfo: "skip",
      showlegend: false
    });
    const xlab = state.mode === "fp" ? "Expected PPR (xFP)" : "Expected touchdowns (xTD)";
    const ylab = state.mode === "fp" ? "Actual PPR" : "Actual touchdowns";
    const layout = {
      margin: { l: 52, r: 16, t: 10, b: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "IBM Plex Sans, sans-serif", size: 12, color: "#5b5e66" },
      xaxis: { title: xlab, range: [lo, hi], zeroline: false, gridcolor: "#ece6d8", linecolor: "#ddd6c7" },
      yaxis: { title: ylab, range: [lo, hi], zeroline: false, gridcolor: "#ece6d8", linecolor: "#ddd6c7", scaleanchor: "x", scaleratio: 1 },
      showlegend: false,
      hoverlabel: { bgcolor: "#16171c", font: { color: "#fff", size: 12 } }
    };
    Plotly.react(el, traces, layout, { responsive: true, displayModeBar: false });
    if (!el.getAttribute("data-click-bound")) {
      el.setAttribute("data-click-bound", "1");
      el.on("plotly_click", function (ev) {
        if (!ev || !ev.points || !ev.points.length) return;
        const cd = ev.points[0].customdata;
        if (!cd) return;
        const id = Array.isArray(cd) ? cd[0] : cd.id;
        if (!id) return;
        state.selected = id;
        drawWeekly(id);
        drawScatter(sortRows(filtered()));
        highlightRow(id);
      });
    }

    const legend = document.getElementById("legend");
    legend.innerHTML = positions.map(function (p) {
      return '<span class="' + p + '"><i></i>' + p + "</span>";
    }).join("");
  }

  const FP_COLS = [
    { k: "name", label: "Player", fmt: "name" },
    { k: "team", label: "Team", fmt: "text" },
    { k: "pos", label: "Pos", fmt: "pos" },
    { k: "g", label: "G", fmt: "int" },
    { k: "opps", label: "Opps", fmt: "int" },
    { k: "fp", label: "FP", fmt: "1" },
    { k: "xfp", label: "xFP", fmt: "1" },
    { k: "dfp", label: "FP−xFP", fmt: "luck1" },
    { k: "fpg", label: "FP/g", fmt: "1" },
    { k: "td", label: "TD", fmt: "0" },
    { k: "xtd", label: "xTD", fmt: "1" },
    { k: "dtd", label: "TD−xTD", fmt: "luck1" },
    { k: "ltd", label: "TD luck", fmt: "luck1" },
    { k: "lyd", label: "Yards luck", fmt: "luck1" },
    { k: "lrec", label: "Catch luck", fmt: "luck1" },
    { k: "lint", label: "INT luck", fmt: "luck1" }
  ];
  const TD_COLS = [
    { k: "name", label: "Player", fmt: "name" },
    { k: "team", label: "Team", fmt: "text" },
    { k: "pos", label: "Pos", fmt: "pos" },
    { k: "g", label: "G", fmt: "int" },
    { k: "opps", label: "Opps", fmt: "int" },
    { k: "td", label: "TD", fmt: "0" },
    { k: "xtd", label: "xTD", fmt: "1" },
    { k: "dtd", label: "TD−xTD", fmt: "luck1" },
    { k: "fp", label: "FP", fmt: "1" },
    { k: "xfp", label: "xFP", fmt: "1" },
    { k: "dfp", label: "FP−xFP", fmt: "luck1" },
    { k: "ltd", label: "TD luck", fmt: "luck1" },
    { k: "lyd", label: "Yards luck", fmt: "luck1" },
    { k: "lrec", label: "Catch luck", fmt: "luck1" },
    { k: "lint", label: "INT luck", fmt: "luck1" }
  ];

  function drawTable(rows) {
    const cols = state.mode === "fp" ? FP_COLS : TD_COLS;
    const thead = document.getElementById("thead");
    thead.innerHTML = "<tr>" + cols.map(function (c) {
      const arrow = state.sortKey === c.k ? (state.sortDir === -1 ? " ▼" : " ▲") : "";
      const cls = (c.k === "pos") ? " class='pos'" : "";
      return "<th" + cls + " data-k='" + c.k + "'>" + c.label + "<span class='arrow'>" + arrow + "</span></th>";
    }).join("") + "</tr>";
    thead.querySelectorAll("th").forEach(function (th) {
      th.addEventListener("click", function () {
        const k = th.getAttribute("data-k");
        if (state.sortKey === k) state.sortDir *= -1;
        else { state.sortKey = k; state.sortDir = (k === "name" || k === "team" || k === "pos") ? 1 : -1; }
        render();
      });
    });
    const tbody = document.getElementById("tbody");
    tbody.innerHTML = rows.map(function (p) {
      const tds = cols.map(function (c) {
        let cls = (c.k === "pos") ? "pos" : "";
        let val = p[c.k];
        let text;
        if (c.fmt === "name") text = val;
        else if (c.fmt === "text") text = val;
        else if (c.fmt === "pos") text = '<span class="pos-tag ' + p.pos + '">' + p.pos + "</span>";
        else if (c.fmt === "int") text = String(val);
        else if (c.fmt === "0") text = fmtRaw(val, 0);
        else if (c.fmt === "1") text = fmtRaw(val, 1);
        else if (c.fmt === "luck1") { text = fmt(val, 1); cls = luckColor(val); }
        else text = String(val);
        return "<td class='" + cls + "'>" + text + "</td>";
      }).join("");
      const sel = p.id === state.selected ? " sel" : "";
      return "<tr data-id='" + p.id + "' class='" + sel + "'>" + tds + "</tr>";
    }).join("");
    tbody.querySelectorAll("tr").forEach(function (tr) {
      tr.addEventListener("click", function () {
        state.selected = tr.getAttribute("data-id");
        drawWeekly(state.selected);
        drawScatter(sortRows(filtered()));
        highlightRow(state.selected);
      });
    });
  }

  function highlightRow(id) {
    document.querySelectorAll("#tbody tr").forEach(function (tr) {
      tr.classList.toggle("sel", tr.getAttribute("data-id") === id);
    });
  }

  function drawWeekly(id) {
    const p = players.find(function (x) { return x.id === id; });
    const card = document.getElementById("weeklyCard");
    if (!p) { card.classList.remove("show"); return; }
    card.classList.add("show");
    document.getElementById("wkName").textContent = p.name + "  ·  " + p.team + " " + p.pos;
    document.getElementById("wkMeta").textContent = p.g + " games  ·  " + fmt(p.dfp, 1) + " PPR  ·  " + fmt(p.dtd, 1) + " TD vs expected";
    const wk = weeksById[id] || [];
    if (typeof Plotly === "undefined") return;
    const xKey = state.mode === "fp" ? "xfp" : "xtd";
    const yKey = state.mode === "fp" ? "fp" : "td";
    const labels = wk.map(function (w) { return "W" + w.w + (w.opp ? " " + w.opp : ""); });
    const traces = [
      {
        type: "bar", name: "Expected", x: labels, y: wk.map(function (w) { return w[xKey]; }),
        marker: { color: "#c8c1b1" }, hovertemplate: "Expected: %{y:.1f}<extra></extra>"
      },
      {
        type: "bar", name: "Actual", x: labels, y: wk.map(function (w) { return w[yKey]; }),
        marker: { color: POS_COLORS[p.pos] || "#16171c" }, hovertemplate: "Actual: %{y:.1f}<extra></extra>"
      }
    ];
    const layout = {
      barmode: "group",
      margin: { l: 40, r: 10, t: 8, b: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "IBM Plex Sans, sans-serif", size: 11, color: "#5b5e66" },
      legend: { orientation: "h", y: 1.15, x: 0 },
      yaxis: { title: state.mode === "fp" ? "PPR" : "TDs", gridcolor: "#ece6d8", zerolinecolor: "#ddd6c7" },
      xaxis: { tickangle: -30 }
    };
    Plotly.react(document.getElementById("weekly"), traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderMethod() {
    const m = DATA.meta;
    const v = m.validation || {};
    const L = v.league2026 || {};
    function pair(arr, digits, label) {
      if (!arr) return "";
      const a = Number(arr[0]), e = Number(arr[1]);
      return '<div class="cal"><span>' + label + "</span><b>" + a.toFixed(digits) + " vs " + e.toFixed(digits) + "</b></div>";
    }
    const hold = v.holdout2025_by_pos || {};
    function c(x) { return x === null || x === undefined ? "—" : Number(x).toFixed(3); }
    document.getElementById("methodBody").innerHTML =
      "<p>" + m.dataThroughNote + " Play-by-play file last updated " + m.pbpUpdated + ".</p>" +
      "<p>" + m.method + "</p>" +
      "<p>" + m.scoring + "</p>" +
      "<p>Luck components add to FP − xFP: <strong>TD luck</strong> (6 points per rushing/receiving TD over expected, 4 per passing TD), <strong>yards luck</strong> (0.1 per rush or receiving yard, 0.04 per passing yard), <strong>catch luck</strong> (1 PPR point per reception over expected), and <strong>INT luck</strong> (−2 per interception over expected, so fewer interceptions than expected is positive luck).</p>" +
      "<p>2026 league totals (actual vs expected) — a well-calibrated opportunity model should be close in aggregate:</p>" +
      '<div class="cals">' +
        pair(L.rush_yd, 0, "Rush yards") +
        pair(L.rec_yd, 0, "Receiving yards") +
        pair(L.rec, 0, "Receptions") +
        pair(L.rush_td, 1, "Rush TDs") +
        pair(L.pass_td, 1, "Pass TDs") +
        pair(L.int, 1, "Interceptions") +
      "</div>" +
      "<p>Sanity check, trained on 2023–24 and evaluated on 2025 player-season totals: correlation of expected and actual PPR was " +
      c(v.holdout2025_corr_fp_xfp) + " overall, and " +
      c(hold.QB) + " / " + c(hold.RB) + " / " + c(hold.WR) + " / " + c(hold.TE) +
      " for qualified QB / RB / WR / TE. On 2026, our xFP correlates " +
      c(v.corr_ffopportunity_xfp) + " with ffopportunity expected PPR, and " +
      c(v.corr_our_xfp_actual) + " with actual PPR. Residual (luck) correlation with ffopportunity is " +
      c(v.corr_ffopportunity_luck) + ".</p>" +
      "<p>This is a first-pass public model. It does not use player identity, coverage, or blocking. Touchdown finishes have a lot of variance in a 9–10 game sample. Usage is not luck: a lot of end-zone targets is opportunity earned. Week 10 is incomplete until Monday Night Football is in the file.</p>";
  }

  render();
})();
</script>
</body>
</html>
'''

path = OUT / "fantasy_luck_2026.html"
path.write_text(html)
index = ROOT / "index.html"
index.write_text(html)
print("Wrote", path, "and", index, "bytes", path.stat().st_size)
