#!/usr/bin/env node
// Headless-Chrome UI regression runner for the dashboard.
//
// Usage: node web_ui_runner.js <base_url>
// Prints one JSON object per line:
//   { test: "...", ok: true|false, detail?: "..." }
// Final line: { "summary": { passed: N, failed: M } }
// Exit code: 0 if all pass, 1 otherwise.
//
// Each test pins a bug we hit during the parity build:
//   - tab_switch_chart_recovery — the canvas-blank-on-tab-back bug
//   - today_loads_account_on_direct_url — TODAY blank on /#today first hit
//   - variant_change_refreshes_today — switching variant didn't refresh TODAY
//   - daily_review_not_truncated — markdown was clipped under 600px iframe
//   - no_js_exceptions_during_navigation — page-load + nav health

const CDP = require('chrome-remote-interface');
const { spawn } = require('child_process');

const BASE_URL = process.argv[2] || 'http://127.0.0.1:8765';
const CHROME = process.env.CHROME_BINARY ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function emit(rec) { console.log(JSON.stringify(rec)); }

async function withChrome(fn) {
  const port = 9500 + Math.floor(Math.random() * 500);
  const userDir = `/tmp/baibot-ui-test-${Date.now()}-${process.pid}`;
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-first-run',
    '--no-default-browser-check', '--disable-features=GoogleUpdate',
    '--password-store=basic', '--use-mock-keychain',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDir}`,
    '--window-size=1400,900',
    BASE_URL,
  ], { stdio: 'pipe' });

  // Suppress chrome stderr — keychain/updater spam isn't useful and pollutes test output.
  chrome.stderr.on('data', () => {});

  await new Promise(r => setTimeout(r, 3500));

  let client;
  try {
    client = await CDP({ port });
    const { Runtime } = client;
    await Runtime.enable();

    const exceptions = [];
    Runtime.exceptionThrown(({ exceptionDetails }) => {
      exceptions.push(exceptionDetails.text + ': ' +
        (exceptionDetails.exception?.description || ''));
    });

    // Wait for the SPA's init() to settle (loadVariants → first trade selected).
    await new Promise(r => setTimeout(r, 5000));

    const ev = async expr => {
      const r = await Runtime.evaluate({ expression: expr, returnByValue: true });
      if (r.exceptionDetails) {
        return { __error: r.exceptionDetails.text };
      }
      return r.result.value;
    };

    return await fn({ ev, exceptions });
  } finally {
    if (client) try { await client.close(); } catch {}
    chrome.kill('SIGKILL');
  }
}

async function main() {
  let passed = 0, failed = 0;
  const test = async (name, fn) => {
    try {
      const detail = await fn();
      emit({ test: name, ok: true, detail: detail || '' });
      passed++;
    } catch (e) {
      emit({ test: name, ok: false, detail: e.message || String(e) });
      failed++;
    }
  };

  await withChrome(async ({ ev, exceptions }) => {

    // ── 1. No JS exceptions during initial load ────────────────────
    await test('no_js_exceptions_on_load', async () => {
      if (exceptions.length) throw new Error('JS exceptions: ' + exceptions.join(' | '));
    });

    // ── 2. Initial chart renders (TRADE is the default route) ──────
    await test('initial_chart_renders', async () => {
      const h = await ev('document.getElementById("chart").clientHeight');
      const cmax = await ev(`Math.max(...Array.from(document.querySelectorAll('#chart canvas')).map(c=>c.height), 0)`);
      if (typeof h !== 'number' || h < 100) throw new Error(`chart height ${h}`);
      if (typeof cmax !== 'number' || cmax < 100) throw new Error(`canvas max-h ${cmax}`);
      return `h=${h} cmax=${cmax}`;
    });

    // ── 3. Tab-switch recovery — leave TRADE for TODAY, return ─────
    //    Pins the autoSize-locks-zero-height bug we just shipped.
    await test('tab_switch_chart_recovery', async () => {
      await ev('document.querySelector("a[data-route=\\"today\\"]").click()');
      await new Promise(r => setTimeout(r, 600));
      // While hidden, chart should be 0 (sanity)
      const hidden_h = await ev('document.getElementById("chart").clientHeight');
      if (hidden_h !== 0) throw new Error(`expected 0 while hidden, got ${hidden_h}`);
      // Switch back
      await ev('document.querySelector("a[data-route=\\"trade\\"]").click()');
      await new Promise(r => setTimeout(r, 1500));
      const back_h = await ev('document.getElementById("chart").clientHeight');
      const back_cmax = await ev(`Math.max(...Array.from(document.querySelectorAll('#chart canvas')).map(c=>c.height), 0)`);
      if (back_h < 100) throw new Error(`chart h after tab-back: ${back_h} (regression — autoSize stuck at 0)`);
      if (back_cmax < 100) throw new Error(`canvas max-h after tab-back: ${back_cmax}`);
      return `h=${back_h} cmax=${back_cmax}`;
    });

    // ── 4. Repeated tab-switch (stress) — should remain stable ─────
    await test('tab_switch_chart_recovery_stress', async () => {
      for (let i = 0; i < 3; i++) {
        await ev('document.querySelector("a[data-route=\\"performance\\"]").click()');
        await new Promise(r => setTimeout(r, 400));
        await ev('document.querySelector("a[data-route=\\"trade\\"]").click()');
        await new Promise(r => setTimeout(r, 700));
      }
      const h = await ev('document.getElementById("chart").clientHeight');
      if (h < 100) throw new Error(`chart h after 3x stress toggles: ${h}`);
      return `h=${h}`;
    });

    // ── 5. Direct /#today URL on first load shows account values ───
    //    Pins the init-order bug (selectedVariant set after routeFromHash).
    await test('today_loads_account_on_direct_url', async () => {
      // Force-reload onto #today
      await ev('window.location.hash = "today"');
      await new Promise(r => setTimeout(r, 1500));
      // The KPI row should have at least one populated card
      const kpi = await ev('document.getElementById("today-kpi").innerHTML');
      if (typeof kpi !== 'string' || !kpi.includes('EQUITY')) {
        throw new Error('today-kpi row has no EQUITY card after direct-URL load');
      }
      // And the equity value shouldn't be the empty placeholder "—"
      const v = await ev(`(() => {
        const m = document.getElementById('today-kpi').querySelectorAll('.kpi-value');
        return Array.from(m).map(e => e.textContent).join(' | ');
      })()`);
      if (typeof v !== 'string' || !/\$\d/.test(v)) {
        throw new Error(`KPI values not populated on direct-URL TODAY: ${v}`);
      }
      return v;
    });

    // ── 6. Variant change refreshes TODAY ──────────────────────────
    await test('variant_change_refreshes_today', async () => {
      // Already on #today from previous test
      const before = await ev('document.getElementById("today-kpi").innerHTML');
      const variants = await ev(`Array.from(document.getElementById('variant-select').options).map(o => o.value)`);
      if (!Array.isArray(variants) || variants.length < 2) {
        return 'skipped — need ≥2 variants';
      }
      const sel = await ev('document.getElementById("variant-select").value');
      const next = variants.find(v => v !== sel);
      // Programmatic change: set value + dispatch change event
      await ev(`(() => {
        const el = document.getElementById('variant-select');
        el.value = ${JSON.stringify(next)};
        el.dispatchEvent(new Event('change'));
      })()`);
      await new Promise(r => setTimeout(r, 1500));
      const after = await ev('document.getElementById("today-kpi").innerHTML');
      if (after === before) {
        throw new Error('TODAY did not refresh after variant change');
      }
      return `${sel} → ${next}`;
    });

    // ── 7. Daily review markdown not truncated under chart iframe ──
    //    Pins .reviews-content overflow:hidden bug.
    await test('daily_review_not_truncated', async () => {
      // Navigate to REVIEWS, find any date that has content.
      await ev('window.location.hash = "reviews"');
      await new Promise(r => setTimeout(r, 800));
      // Try picking a few recent dates until we find one with files
      const dates = ['2026-05-01', '2026-04-30', '2026-04-29', '2026-04-28'];
      let foundDate = null;
      for (const d of dates) {
        await ev(`(() => { const el=document.getElementById('reviews-date'); el.value='${d}'; el.dispatchEvent(new Event('change')); })()`);
        await new Promise(r => setTimeout(r, 800));
        const variants = await ev('document.querySelectorAll("#reviews-variants .reviews-variant-btn").length');
        if (variants > 0) { foundDate = d; break; }
      }
      if (!foundDate) return 'skipped — no daily reviews on disk for recent dates';
      // First variant + first file should auto-select. Wait for content.
      await new Promise(r => setTimeout(r, 1200));
      // The bug was: visible markdown clipped to a few lines despite full
      // content in DOM. Test: scrollHeight of #reviews-md should match its
      // children's natural extent (no overflow:hidden squeeze).
      const dims = await ev(`(() => {
        const md = document.getElementById('reviews-md');
        const content = document.querySelector('.reviews-content');
        return JSON.stringify({
          md_scroll_h: md.scrollHeight,
          md_client_h: md.clientHeight,
          md_text_len: md.innerText.length,
          content_overflow: getComputedStyle(content).overflow,
          md_overflow_y: getComputedStyle(md).overflowY,
        });
      })()`);
      const r = JSON.parse(dims);
      if (r.md_text_len < 50) return `${foundDate}: markdown empty (md_text_len=${r.md_text_len}) — likely no content for this date`;
      // The fix removed overflow:hidden on .reviews-content and overflow-y:auto on .reviews-markdown.
      // Pin both.
      if (r.content_overflow === 'hidden') throw new Error('.reviews-content overflow:hidden returned (will clip markdown)');
      if (r.md_overflow_y === 'auto' || r.md_overflow_y === 'scroll') throw new Error(`.reviews-markdown overflow-y='${r.md_overflow_y}' returned (will squeeze under iframe)`);
      return `${foundDate}: text=${r.md_text_len}c md_h=${r.md_client_h}`;
    });

    // ── 8. IDEAS tab not in nav (we hid it) ────────────────────────
    await test('ideas_tab_hidden_from_nav', async () => {
      const present = await ev(`!!document.querySelector('nav.tabs a[data-route="ideas"]')`);
      if (present === true) throw new Error('IDEAS tab still in nav');
    });

    // ── 9. Reviews top overview cards render (parity feature) ──────
    await test('reviews_overview_cards_present', async () => {
      await ev('window.location.hash = "reviews"');
      await new Promise(r => setTimeout(r, 1500));
      const n = await ev('document.querySelectorAll("#reviews-overview .reviews-card").length');
      const totals = await ev('document.querySelectorAll("#reviews-totals .rt-cell").length');
      if (typeof n !== 'number' || n < 1) throw new Error(`reviews-overview cards: ${n}`);
      if (totals !== 4) throw new Error(`reviews-totals cells: ${totals} (want 4)`);
      return `${n} cards, ${totals} totals`;
    });

    // ── 10. Risk page sections render (parity feature, no errors) ──
    await test('risk_page_renders_all_sections', async () => {
      const exBefore = exceptions.length;
      await ev('window.location.hash = "portfolio"');
      await new Promise(r => setTimeout(r, 2500));
      const sections = await ev(`(() => ({
        corr: !!document.querySelector('#risk-corr-table table, #risk-corr-table .tb-empty'),
        recent: !!document.querySelector('#risk-corr-recent-table table, #risk-corr-recent-table .tb-empty'),
        sizing: !!document.querySelector('#risk-sizing-table table, #risk-sizing-table .tb-empty'),
        regime: !!document.querySelector('#risk-regime-table table, #risk-regime-table .tb-empty'),
        patterns: !!document.querySelector('#risk-patterns-table table, #risk-patterns-table .tb-empty'),
        ai_btn: !!document.getElementById('risk-ai-btn'),
      }))()`);
      const missing = Object.entries(sections).filter(([k, v]) => !v).map(([k]) => k);
      if (missing.length) throw new Error('missing sections: ' + missing.join(','));
      const exAfter = exceptions.length;
      if (exAfter > exBefore) throw new Error('JS exceptions thrown while rendering risk: ' +
        exceptions.slice(exBefore).join(' | '));
      return JSON.stringify(sections);
    });

    // ── 11. AI synthesis must NOT auto-fire on page load ──────────
    //    Cost protection: each click is ~$0.05. If a render path
    //    accidentally invokes triggerAiSynthesis, costs leak quietly.
    await test('ai_synthesis_does_not_auto_fire', async () => {
      // We're on #portfolio from previous test. The output area should
      // be hidden (display:none via [hidden] attribute) until the user
      // clicks the button.
      const out_hidden = await ev(`document.getElementById('risk-ai-output').hidden`);
      const out_text = await ev(`document.getElementById('risk-ai-output').innerText`);
      if (out_hidden !== true) throw new Error('AI output not hidden (auto-fired?)');
      if (out_text && out_text.trim().length > 0) throw new Error(`AI output has text without click: ${out_text.slice(0,80)}`);
      return 'hidden + empty';
    });

    // ── 12. Performance page creates four chart instances ─────────
    await test('performance_renders_four_charts', async () => {
      const exBefore = exceptions.length;
      await ev('window.location.hash = "performance"');
      await new Promise(r => setTimeout(r, 2500));
      const dims = await ev(`(() => ({
        eq: !!document.querySelector('#perf-equity-chart canvas'),
        ret: !!document.querySelector('#perf-returns-chart canvas'),
        dd: !!document.querySelector('#perf-drawdown-chart canvas'),
        act: !!document.querySelector('#perf-activity-chart canvas'),
        chips: document.querySelectorAll('#perf-variants .vf-chip').length,
      }))()`);
      const missing = ['eq', 'ret', 'dd', 'act'].filter(k => !dims[k]);
      if (missing.length) throw new Error('missing charts: ' + missing.join(','));
      if (dims.chips < 1) throw new Error(`variant filter chips: ${dims.chips}`);
      if (exceptions.length > exBefore) throw new Error('JS exceptions during performance render');
      return `chips=${dims.chips}`;
    });

    // ── 13. escapeHtml hardening — non-string inputs don't crash ──
    //    The original implementation called s.replace which throws on
    //    null/undefined/numbers. Ensure the guard stays.
    await test('escapeHtml_handles_non_strings', async () => {
      const r = await ev(`(() => {
        try {
          const a = escapeHtml(null);
          const b = escapeHtml(undefined);
          const c = escapeHtml(42);
          const d = escapeHtml({toString: () => '<x>'});
          return JSON.stringify({a, b, c, d});
        } catch (e) {
          return 'THREW: ' + e.message;
        }
      })()`);
      if (typeof r !== 'string' || r.startsWith('THREW')) throw new Error(r);
      const parsed = JSON.parse(r);
      if (parsed.c !== '42') throw new Error(`escapeHtml(42) = ${parsed.c}`);
      return r;
    });

    // ── 14. Log page wires auto-refresh + stops on deactivate ─────
    await test('log_page_renders_and_stops_on_leave', async () => {
      await ev('window.location.hash = "log"');
      await new Promise(r => setTimeout(r, 1500));
      const body = await ev('document.getElementById("log-body").innerText');
      if (typeof body !== 'string' || body.length < 5) throw new Error(`log body empty: ${body}`);
      // Switch away — _log.timer should be cleared
      await ev('window.location.hash = "trade"');
      await new Promise(r => setTimeout(r, 600));
      const timer = await ev('_log.timer');
      if (timer !== null && timer !== undefined) throw new Error(`_log.timer not cleared on deactivate: ${timer}`);
      return `body=${body.length}c`;
    });

    // ── 15. Daily review chart uses lightweight-charts (not iframe) ─
    await test('daily_review_uses_lightweight_charts', async () => {
      await ev('window.location.hash = "reviews"');
      await new Promise(r => setTimeout(r, 1500));
      // Walk recent dates until we find one with a per-trade closed review.
      // Auto-select picks the variant summary which has no chart_payload;
      // we need to click an entry from the "Closed" section explicitly.
      const dates = ['2026-05-01', '2026-04-30', '2026-04-29', '2026-04-28', '2026-04-27'];
      let clicked = false;
      for (const d of dates) {
        await ev(`(() => { const el=document.getElementById('reviews-date'); el.value='${d}'; el.dispatchEvent(new Event('change')); })()`);
        await new Promise(r => setTimeout(r, 1500));
        // Look for a non-summary, non-section <li> in the file list.
        const ok = await ev(`(() => {
          const items = Array.from(document.querySelectorAll('#reviews-files li'));
          // Skip section headers (.rfl-section) and summary entries
          // (the dataset.kind === 'summary'). Click the first 'closed' file.
          const closed = items.find(el =>
            el.dataset.kind === 'closed' && !el.classList.contains('rfl-section')
          );
          if (closed) { closed.click(); return true; }
          return false;
        })()`);
        if (ok) { clicked = true; break; }
      }
      if (!clicked) return 'skipped — no closed daily reviews on disk for recent dates';
      await new Promise(r => setTimeout(r, 2000));  // Give chart_payload fetch + render time
      const dims = await ev(`(() => {
        const canvas = document.querySelectorAll('#reviews-chart-canvas canvas').length;
        const iframe = document.querySelectorAll('#reviews-chart iframe').length;
        return JSON.stringify({ canvas, iframe });
      })()`);
      const r = JSON.parse(dims);
      if (r.canvas < 1) throw new Error(`no lightweight-charts canvas (canvas=${r.canvas} iframe=${r.iframe})`);
      const lines = await ev('_reviewsChart.fillPriceLines.length');
      if (typeof lines !== 'number' || lines < 1) throw new Error(`no fill price lines on review chart: ${lines}`);
      return `canvas=${r.canvas} fillPriceLines=${lines}`;
    });

    // ── 16. Closed-trade chart shows entry + exit price lines ─────
    //    Pin the "clicking SELL renders both buy + sell" + price-line
    //    visualization improvement.
    await test('closed_trade_shows_entry_and_exit_price_lines', async () => {
      // Navigate to TRADE
      await ev('window.location.hash = "trade"');
      await new Promise(r => setTimeout(r, 1500));
      // Find a SELL row in the trades list (status filled, side sell)
      const sellId = await ev(`(() => {
        for (const t of (state.trades || [])) {
          if ((t.side || '').toLowerCase() === 'sell') return t.trade_id;
        }
        return null;
      })()`);
      if (sellId == null) return 'skipped — no sell rows in current variant';
      // Click the matching <li>
      const clicked = await ev(`(() => {
        const el = document.querySelector('.trades-list li[data-trade-id="${sellId}"]');
        if (el) { el.click(); return true; }
        return false;
      })()`);
      if (!clicked) return `skipped — no <li> for trade_id=${sellId}`;
      await new Promise(r => setTimeout(r, 1500));
      const lines = await ev('state.fillPriceLines.length');
      const buySides = await ev(`(state.currentPayload.fills || []).map(f => f.side).filter(s => s === 'buy').length`);
      const sellSides = await ev(`(state.currentPayload.fills || []).map(f => f.side).filter(s => s === 'sell').length`);
      if (typeof lines !== 'number' || lines < 2) throw new Error(`expected ≥2 price lines, got ${lines}`);
      if (buySides < 1) throw new Error('SELL-click missing BUY fill (regression)');
      if (sellSides < 1) throw new Error('SELL-click missing SELL fill');
      return `lines=${lines} buys=${buySides} sells=${sellSides}`;
    });

    // ── 17. No JS exceptions during whole session ─────────────────
    await test('no_js_exceptions_during_session', async () => {
      if (exceptions.length) throw new Error('JS exceptions accumulated: ' + exceptions.join(' | '));
    });
  });

  emit({ summary: { passed, failed } });
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
  emit({ test: 'runner_crash', ok: false, detail: e.message || String(e) });
  emit({ summary: { passed: 0, failed: 1 } });
  process.exit(1);
});
