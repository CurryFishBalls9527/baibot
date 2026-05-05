// baibot v2 client bootstrap.
//
// Each page imports this and uses window.bb to fetch initial state and
// subscribe to live events. Two SSE streams already exist on the server:
//
//   /events/stream                 — global event log (FIRE/FILL/VETO/etc)
//   /api/variants/{v}/stream       — per-variant new-trade pushes
//
// We expose both via bb.events (the global one) and bb.subscribeVariant
// (per-variant). Pages register handlers via bb.on(type, fn).

(function () {
  'use strict';

  const ENDPOINTS = {
    state:        '/api/v2/state',
    fills:        '/api/v2/fills',
    equityDaily:  '/api/v2/equity/daily',
    calendar:     '/api/v2/calendar',
    versus:       '/api/v2/versus',
    spotlight:    (variant, sym) => `/api/v2/spotlight/${encodeURIComponent(variant)}/${encodeURIComponent(sym)}`,
    eventsStream: '/events/stream',
    variantStream: (v) => `/api/variants/${encodeURIComponent(v)}/stream`,
  };

  const handlers = new Map();   // type → Set<fn>
  const variantStreams = new Map();  // variant → EventSource

  const bb = {
    state: null,
    asof: null,

    async hydrate() {
      const res = await fetch(ENDPOINTS.state, { cache: 'no-store' });
      if (!res.ok) throw new Error(`state ${res.status}`);
      const json = await res.json();
      this.state = json;
      this.asof = json.asof;
      this._dispatch({ type: 'state.hydrated', state: json });
      return json;
    },

    on(type, fn) {
      if (!handlers.has(type)) handlers.set(type, new Set());
      handlers.get(type).add(fn);
      return () => handlers.get(type)?.delete(fn);
    },

    _dispatch(ev) {
      const set = handlers.get(ev.type);
      if (set) set.forEach(fn => { try { fn(ev); } catch (e) { console.error('handler', ev.type, e); } });
      const all = handlers.get('*');
      if (all) all.forEach(fn => { try { fn(ev); } catch (e) { console.error(e); } });
    },

    // Global event stream (events.jsonl)
    connectEvents() {
      if (this._eventsES) return;
      const es = new EventSource(ENDPOINTS.eventsStream);
      this._eventsES = es;
      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          this._dispatch({ type: 'event', event: data });
        } catch (_) { /* ignore non-JSON keepalives */ }
      };
      es.onerror = () => {
        // EventSource auto-reconnects; we just surface a status event.
        this._dispatch({ type: 'events.disconnected' });
      };
      es.onopen = () => {
        this._dispatch({ type: 'events.connected' });
      };
    },

    // Per-variant trade stream
    connectVariant(variant) {
      if (variantStreams.has(variant)) return variantStreams.get(variant);
      const es = new EventSource(ENDPOINTS.variantStream(variant));
      variantStreams.set(variant, es);
      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === 'trade') {
            this._dispatch({ type: 'trade.new', variant, trade: data.trade });
          } else if (data.type === 'snapshot') {
            this._dispatch({ type: 'variant.snapshot', variant, last_trade_id: data.last_trade_id });
          }
        } catch (_) { /* ignore */ }
      };
      es.onerror = () => this._dispatch({ type: 'variant.disconnected', variant });
      es.onopen  = () => this._dispatch({ type: 'variant.connected', variant });
      return es;
    },

    disconnectVariant(variant) {
      const es = variantStreams.get(variant);
      if (es) { es.close(); variantStreams.delete(variant); }
    },

    // Cursor-paginated helper. cursor.before / cursor.limit; returns
    // ``{ rows, next }`` where rows is the current page and next is a
    // function that fetches the page after this one (or null when
    // exhausted).
    async paginate(url, params = {}, cursorKey = 'next_before', listKey = null) {
      const qs = new URLSearchParams(params);
      const res = await fetch(url + (qs.toString() ? '?' + qs : ''), { cache: 'no-store' });
      if (!res.ok) throw new Error(`${url} ${res.status}`);
      const json = await res.json();
      const rowKey = listKey || (json.fills ? 'fills' : (json.trades ? 'trades' : 'rows'));
      const rows = json[rowKey] || [];
      const nb = json[cursorKey];
      const next = (nb == null) ? null : () =>
        bb.paginate(url, { ...params, before: nb }, cursorKey, rowKey);
      return { rows, next, raw: json };
    },

    // Fetch helpers — non-paginated.
    async fetchEquityDaily(variant, range = '30d') {
      const r = await fetch(`${ENDPOINTS.equityDaily}?variant=${encodeURIComponent(variant)}&range=${encodeURIComponent(range)}`);
      if (!r.ok) throw new Error(`equity ${r.status}`);
      return r.json();
    },
    async fetchCalendar(variant, month) {
      const r = await fetch(`${ENDPOINTS.calendar}?variant=${encodeURIComponent(variant)}&month=${encodeURIComponent(month)}`);
      if (!r.ok) throw new Error(`calendar ${r.status}`);
      return r.json();
    },
    async fetchVersus(a, b, range = '30d', before = null) {
      const params = new URLSearchParams({ a, b, range });
      if (before != null) params.set('before', String(before));
      const r = await fetch(`${ENDPOINTS.versus}?${params}`);
      if (!r.ok) throw new Error(`versus ${r.status}`);
      return r.json();
    },
    async fetchSpotlight(variant, symbol) {
      const r = await fetch(ENDPOINTS.spotlight(variant, symbol));
      if (!r.ok) throw new Error(`spotlight ${r.status}`);
      return r.json();
    },
    async fetchLive() {
      const r = await fetch('/api/v2/equity/live', { cache: 'no-store' });
      if (!r.ok) throw new Error(`live ${r.status}`);
      return r.json();
    },
  };

  // Stash the endpoint table for pages that want to build their own URLs.
  bb.endpoints = ENDPOINTS;

  // Format helpers used in multiple pages.
  bb.fmt = {
    pct(n, signed = true) {
      if (n == null || Number.isNaN(n)) return '—';
      const v = Number(n);
      const s = (signed && v >= 0 ? '+' : '') + v.toFixed(2) + '%';
      return s;
    },
    usd(n, signed = true) {
      if (n == null || Number.isNaN(n)) return '—';
      const v = Number(n);
      const sign = (signed && v >= 0) ? '+' : (v < 0 ? '−' : '');
      const abs = Math.abs(v);
      const fmt = abs >= 1000 ? `$${(abs/1000).toFixed(1)}k` : `$${abs.toFixed(0)}`;
      return sign + fmt;
    },
    time(ts) {
      if (!ts) return '—';
      // Accept "YYYY-MM-DD HH:MM:SS" or ISO. Show HH:MM:SS in local TZ.
      const d = new Date(String(ts).replace(' ', 'T'));
      if (Number.isNaN(d.getTime())) return String(ts).slice(11, 19);
      return d.toTimeString().slice(0, 8);
    },
    date(ts) {
      if (!ts) return '—';
      return String(ts).slice(0, 10);
    },
  };

  window.bb = bb;
})();
