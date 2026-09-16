/* WebMCP tools for the Tensor Atlas (progressive enhancement).
 *
 * 규격: W3C WebMCP CG Draft / Chrome 149+ origin trial.
 *   - 도구 등록: <modelContext>.registerTool({name, description, inputSchema, execute, annotations})
 *   - 표면은 브라우저 버전에 따라 document.modelContext / navigator.modelContext → 둘 다 감지.
 *   - 실행: getTools() → executeTool(tool, JSON.stringify(args))  / 테스트: navigator.modelContextTesting
 *   - 로컬 확인: chrome://flags/#enable-webmcp-testing → DevTools Application > WebMCP 또는 Inspector 확장
 *   - 미지원 브라우저에서는 아무 것도 하지 않는다(기존 페이지 동작 불변).
 *
 * 이 파일은 build.py 가 index.html 의 WebMCP 자리표시자에 인라인으로 삽입한다(외부 스크립트 아님).
 * 데이터 출처는 페이지가 이미 노출하는 window.ATLAS_DEBUG 뿐이다.
 * 벤치마크 도구를 위해 src.html 의 그 객체에 BENCH, BENCH_MODELS 식별자를 추가했다.
 * 도구 8종: get_model_overview / list_categories / search_tensors / get_tensor_detail /
 *           focus_tensor / get_view_state / list_benchmarks / compare_models
 */
(function () {
  'use strict';

  var mc = null;
  try { mc = document.modelContext || (window.navigator && window.navigator.modelContext) || null; } catch (e) { mc = null; }

  // 진단용 상태 노출 — 미지원일 때 조용히 끝나면 원인을 알 수 없으므로 상태를 남긴다.
  var supported = !!(mc && typeof mc.registerTool === 'function');
  window.__WEBMCP_STATUS = supported ? 'pending' : (mc ? 'unsupported-api-shape' : 'unsupported-browser');
  if (!supported) {
    try {
      console.info('[webmcp] document.modelContext 없음 (status=' + window.__WEBMCP_STATUS + '). ' +
        'Chrome 149+ 에서 chrome://flags/#enable-webmcp-testing 을 Enabled 로 바꾸고 브라우저를 재시작한 뒤, ' +
        'Ctrl+Shift+R 로 새로고침하세요. 콘솔에서 typeof document.modelContext 로 확인 가능.');
    } catch (e) {}
    try {
      var probe = (typeof mc.provideContext === 'function'); // API 표면이 provideContext 뿐인 빌드 대비
      if (probe) { window.__WEBMCP_STATUS = 'provideContext-only'; }
    } catch (e) {}
    if (window.__WEBMCP_STATUS !== 'provideContext-only') { return; }
  }

  var ATLAS_WAIT_MS = 12000;

  function atlas() { return window.ATLAS_DEBUG || null; }

  function waitForAtlas(timeoutMs) {
    var t0 = Date.now();
    return new Promise(function (resolve, reject) {
      (function poll() {
        var d = atlas();
        // engine 은 3D 초기화(WebGL) 실패 시 null 일 수 있다 — 데이터/벤치마크 도구는 불필요
        if (d && d.app) return resolve(d);
        if (Date.now() - t0 > (timeoutMs || ATLAS_WAIT_MS)) return reject(new Error('ATLAS_DEBUG not ready (atlas still loading)'));
        setTimeout(poll, 200);
      })();
    });
  }

  function num(n) { return (n == null ? 0 : n).toLocaleString('en-US'); }
  function billions(n) { return (n / 1e9).toFixed(2) + 'B'; }
  function bytes(n) { return (n / 1e9).toFixed(1) + ' GB'; }

  function allTensors(d) {
    var out = [];
    for (var i = 0; i < d.MODULES.length; i++) {
      var m = d.MODULES[i];
      var ws = m.ws || [];
      for (var j = 0; j < ws.length; j++) {
        out.push({
          module: m.id,
          module_label: m.label || m.id,
          layer: (m.index == null ? null : m.index),
          part: m.part || null,
          mode: m.mode || null,
          name: ws[j].name,
          shape: ws[j].shape,
          format: ws[j].format,
          cat: ws[j].cat,
          count: ws[j].count,
          params: ws[j].p,
          note: ws[j].note || null
        });
      }
    }
    return out;
  }

  function ok(payload) { return JSON.stringify(payload, null, 1); }
  function err(e) { return JSON.stringify({ error: String((e && e.message) || e) }, null, 1); }

  /* ---- model-card benchmark table (페이지가 window.ATLAS_DEBUG.BENCH 로 노출) ---- */
  var METRIC_NOTE = "Percentages or the card's 0-100 score. ZeroBench uses Pass@5, ProgramBench Almost@1, DeepSWE resolved rate (harness: mini-SWE). Do not average unlike metrics; harnesses vary per benchmark.";

  function benchTable(d) {
    var rows = d && d.BENCH, models = d && d.BENCH_MODELS;
    if (!rows || !models) throw new Error('benchmark table not exposed by this page build (window.ATLAS_DEBUG.BENCH missing)');
    return {
      models: models.slice(),
      rows: rows.map(function (r) { return { name: r[0], category: r[1], values: (r[2] || []).slice() }; })
    };
  }

  function ranked(values, models) {
    var all = [], scored = [], missing = [], i, j;
    for (i = 0; i < models.length; i++) all.push({ model: models[i], score: values[i] == null ? null : values[i] });
    for (i = 0; i < all.length; i++) (all[i].score == null ? missing : scored).push(all[i]);
    scored.sort(function (a, b) { return b.score - a.score; });
    var best = scored.length ? scored[0].score : null;
    for (i = 0; i < scored.length; i++) {
      var above = 0;
      for (j = 0; j < scored.length; j++) if (scored[j].score > scored[i].score) above++;
      scored[i].rank = above + 1;
      scored[i].delta_vs_best = best == null ? null : Math.round((scored[i].score - best) * 10) / 10;
    }
    var out = scored.concat(missing.map(function (r) { return { model: r.model, score: null, note: 'not reported' }; }));
    return { ranking: out, reported: scored.length };
  }

  function findRow(t, want) {
    var w = String(want || '').trim().toLowerCase();
    return t.rows.filter(function (r) { return r.name.toLowerCase() === w; })[0] ||
           t.rows.filter(function (r) { return r.name.toLowerCase().indexOf(w) !== -1; })[0] || null;
  }
  function findModel(ranking, want) {
    var w = String(want || '').trim().toLowerCase();
    return ranking.filter(function (r) { return r.model.toLowerCase() === w; })[0] ||
           ranking.filter(function (r) { return r.model.toLowerCase().indexOf(w) !== -1; })[0] || null;
  }

  var TOOLS = [
    {
      name: 'get_model_overview',
      description: 'Return this atlas model\'s headline facts: model_type, architectures, dtype, total parameter count, per-precision totals as offered by the page, layer count and category summary. Use this first to understand what model is on the page.',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true },
      execute: function () {
        return waitForAtlas().then(function (d) {
          var cfg = d.CFG || {};
          var cats = (d.CATEGORIES || []).map(function (c) {
            return { key: c[0], label: c[1], tensors: (c[3] || []).length };
          });
          return ok({
            model_type: cfg.model_type || null,
            architectures: cfg.architectures || null,
            dtype: cfg.dtype || null,
            transformers_version: cfg.transformers_version || null,
            quantization_config: cfg.quantization_config || null,
            total_params: d.TOTAL_P,
            total_params_label: billions(d.TOTAL_P),
            totals_by_precision: d.TOTALS || null,
            layer_modules: d.MODULES.length,
            categories: cats,
            current_precision: d.app.state.precision
          });
        }).catch(err);
      }
    },
    {
      name: 'list_categories',
      description: 'List the atlas tensor categories (e.g. routed experts, attention, vision) with their tensor counts and total parameter counts, so an agent can decide where to look.',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true },
      execute: function () {
        return waitForAtlas().then(function (d) {
          var rows = (d.CATEGORIES || []).map(function (c) {
            var ws = c[3] || [], params = 0;
            for (var i = 0; i < ws.length; i++) params += (ws[i].p || 0);
            return { key: c[0], label: c[1], color: c[2], tensors: ws.length, params: params, params_label: billions(params) };
          });
          return ok({ categories: rows, total_tensors: allTensors(d).length });
        }).catch(err);
      }
    },
    {
      name: 'search_tensors',
      description: 'Search weight tensors by name substring, category, or storage format (e.g. "experts.w1", cat "attention", format "fp4"). Returns matching tensor names with shapes, formats, parameter counts and the module/layer they belong to.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Substring to match against the tensor name (case-insensitive). Leave empty to match all.' },
          category: { type: 'string', description: 'Exact category key as returned by list_categories (e.g. attention, expert, vision).' },
          format: { type: 'string', description: 'Storage format as reported in the tensor data (e.g. fp8, fp4, bf16).' },
          module: { type: 'string', description: 'Restrict to one module id, e.g. embed or L0.' },
          limit: { type: 'integer', description: 'Max rows to return (default 20, max 200).' }
        },
        additionalProperties: false
      },
      annotations: { readOnlyHint: true },
      execute: function (a) {
        a = a || {};
        return waitForAtlas().then(function (d) {
          var q = (a.query || '').toLowerCase();
          var lim = Math.min(Math.max(parseInt(a.limit, 10) || 20, 1), 200);
          var rows = allTensors(d).filter(function (t) {
            if (q && String(t.name).toLowerCase().indexOf(q) === -1) return false;
            if (a.category && t.cat !== a.category) return false;
            if (a.format && t.format !== a.format) return false;
            if (a.module && t.module !== a.module) return false;
            return true;
          });
          return ok({
            total_matches: rows.length,
            returned: Math.min(rows.length, lim),
            tensors: rows.slice(0, lim).map(function (t) {
              return { name: t.name, module: t.module, cat: t.cat, format: t.format, shape: t.shape, count: t.count, params: t.params };
            })
          });
        }).catch(err);
      }
    },
    {
      name: 'get_tensor_detail',
      description: 'Fetch the full record for one tensor by exact name (as returned by search_tensors), including its shape, storage format, parameter count, and the explanatory note the atlas shows in its inspector panel.',
      inputSchema: {
        type: 'object',
        properties: { name: { type: 'string', description: 'Exact tensor name, e.g. "ffn.experts.w1.weight".' } },
        required: ['name'],
        additionalProperties: false
      },
      annotations: { readOnlyHint: true },
      execute: function (a) {
        a = a || {};
        return waitForAtlas().then(function (d) {
          var name = String(a.name || '');
          var hits = allTensors(d).filter(function (t) { return t.name === name; });
          if (!hits.length) {
            var near = allTensors(d).filter(function (t) { return t.name.indexOf(name) !== -1; }).slice(0, 10);
            return ok({ found: false, requested: name, suggestions: near.map(function (t) { return t.name; }) });
          }
          if (hits.length > 1) {
            return ok({ found: true, ambiguous: true, matches: hits.map(function (t) { return { name: t.name, module: t.module, cat: t.cat, shape: t.shape, format: t.format, count: t.count, params: t.params, note: t.note }; }) });
          }
          return ok({ found: true, tensor: hits[0] });
        }).catch(err);
      }
    },
    {
      name: 'focus_tensor',
      description: 'Navigate the atlas to a tensor so the user sees it highlighted in the 3D view and the inspector panel (uses the page\'s own selection logic). Use after search_tensors/get_tensor_detail. Returns the resulting view state.',
      inputSchema: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Tensor name to focus, e.g. "ffn.experts.w1.weight".' },
          module: { type: 'string', description: 'Optional module id to disambiguate (e.g. L0).' }
        },
        required: ['name'],
        additionalProperties: false
      },
      annotations: { readOnlyHint: false, consequentialHint: false },
      execute: function (a) {
        a = a || {};
        return waitForAtlas().then(function (d) {
          var name = String(a.name || '');
          var modId = a.module || null;
          var hit = null;
          var list = allTensors(d);
          for (var i = 0; i < list.length; i++) {
            if (list[i].name === name && (!modId || list[i].module === modId)) { hit = list[i]; break; }
          }
          if (!hit) return ok({ focused: false, reason: 'tensor not found', requested: name, module: modId });
          if (typeof d.app.select !== 'function') return ok({ focused: false, reason: 'view control unavailable (3D engine not initialised in this browser)' });
          d.app.select(hit.module, name);
          // React 상태 반영을 기다린 뒤 스냅샷 (즉시 읽으면 selected가 null로 나온다)
          return new Promise(function (resolve) {
            setTimeout(function () {
              var s = d.app.state;
              resolve(ok({ focused: true, module: hit.module, tensor: name, cat: hit.cat, format: hit.format, view: s.view, precision: s.precision, selected: s.selected, selected_tensor: s.tensor }));
            }, 200);
          });
        }).catch(err);
      }
    },
    {
      name: 'get_view_state',
      description: 'Report what the user is currently looking at in the atlas: view mode, precision, layout, selected module/tensor, filter, and animation state. Use it to answer "what am I looking at" or to stay in sync with the user.',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true },
      execute: function () {
        return waitForAtlas().then(function (d) {
          var s = d.app.state;
          return ok({
            view: s.view, precision: s.precision, layout: s.layout, filter: s.filter,
            selected_module: s.selected, selected_tensor: s.tensor, hover_tensor: s.hoverTensor,
            playing: s.playing, timeline_time: s.time, expert_view: s.expertView, detail_layout: s.detailLayout
          });
        }).catch(err);
      }
    },
    {
      name: 'list_benchmarks',
      description: 'List every reported evaluation on this atlas model card (reasoning, agentic, visual) with each model\'s score and rank, so an agent can choose a benchmark to compare. Use compare_models for a focused head-to-head.',
      inputSchema: {
        type: 'object',
        properties: { category: { type: 'string', description: 'Optional filter: Reasoning, Agentic or Visual.' } },
        additionalProperties: false
      },
      annotations: { readOnlyHint: true },
      execute: function (a) {
        a = a || {};
        return waitForAtlas().then(function (d) {
          var t = benchTable(d);
          var rows = t.rows.filter(function (r) {
            return !a.category || String(r.category).toLowerCase() === String(a.category).toLowerCase();
          });
          return ok({
            models: t.models,
            count: rows.length,
            metric_note: METRIC_NOTE,
            benchmarks: rows.map(function (r) {
              return { name: r.name, category: r.category, ranking: ranked(r.values, t.models).ranking };
            })
          });
        }).catch(err);
      }
    },
    {
      name: 'compare_models',
      description: 'Compare models on one reported benchmark of this model card. Give a benchmark name (e.g. "DeepSWE v1.1") and optionally the models to compare (e.g. ["Qwen3.8 27B (this atlas)", "GLM-5.3"]); returns each score with rank and delta vs the best reported score, plus the metric caveat.',
      inputSchema: {
        type: 'object',
        properties: {
          benchmark: { type: 'string', description: 'Benchmark name from list_benchmarks, e.g. "DeepSWE v1.1" (partial match allowed).' },
          models: { type: 'array', items: { type: 'string' }, description: 'Optional subset of models to focus the comparison on (partial names allowed).' }
        },
        required: ['benchmark'],
        additionalProperties: false
      },
      annotations: { readOnlyHint: true },
      execute: function (a) {
        a = a || {};
        return waitForAtlas().then(function (d) {
          var t = benchTable(d);
          if (!a.benchmark) return ok({ error: 'benchmark is required', available: t.rows.map(function (r) { return r.name; }) });
          var row = findRow(t, a.benchmark);
          if (!row) return ok({ error: 'benchmark not found: ' + a.benchmark, available: t.rows.map(function (r) { return r.name; }) });
          var r = ranked(row.values, t.models);
          var focus = null;
          if (a.models && a.models.length) {
            focus = a.models.map(function (m) {
              var hit = findModel(r.ranking, m);
              return hit || { model: m, error: 'model not in table' };
            });
            if (focus.length === 2 && !focus[0].error && !focus[1].error && focus[0].score != null && focus[1].score != null) {
              var hi = focus[0].score >= focus[1].score ? focus[0] : focus[1];
              var lo = focus[0].score >= focus[1].score ? focus[1] : focus[0];
              focus.push({
                head_to_head: hi.model + ' vs ' + lo.model,
                delta: Math.round((hi.score - lo.score) * 10) / 10,
                relative_percent: Math.round(((hi.score - lo.score) / lo.score) * 1000) / 10,
                leader: hi.model
              });
            }
          }
          return ok({
            benchmark: row.name,
            category: row.category,
            reported_models: r.reported,
            ranking: r.ranking,
            selected: focus,
            metric_note: METRIC_NOTE
          });
        }).catch(err);
      }
    }
  ];

  function register(list) {
    // registerTool 우선, 없으면 provideContext({tools}) 로 폴백 (API 표면 변동 대응)
    if (typeof mc.registerTool !== 'function' && typeof mc.provideContext === 'function') {
      try {
        mc.provideContext({ tools: list });
        window.__WEBMCP_STATUS = 'registered (provideContext)';
        try { console.log('[webmcp] atlas tools provided via provideContext:', list.length); } catch (e) {}
      } catch (e) {
        window.__WEBMCP_STATUS = 'provideContext-failed';
        try { console.warn('[webmcp] provideContext failed', e); } catch (e2) {}
      }
      return;
    }
    var i = 0;
    (function next() {
      if (i >= list.length) {
        window.__WEBMCP_STATUS = 'registered (' + list.length + ' tools)';
        try { console.log('[webmcp] atlas tools registered:', list.length, list.map(function (t) { return t.name; })); } catch (e) {}
        return;
      }
      var tool = list[i++];
      try {
        var p = mc.registerTool(tool);
        if (p && typeof p.then === 'function') { p.then(next, next); } else { next(); }
      } catch (e) {
        try { console.warn('[webmcp] registerTool failed for', tool.name, e); } catch (e2) {}
        next();
      }
    })();
  }

  // 디버깅/자동검증용 진입점 (inspector 확장 없이도 호출 가능)
  window.__WEBMCP_ATLAS = {
    tools: TOOLS,
    call: function (name, args) {
      for (var i = 0; i < TOOLS.length; i++) {
        if (TOOLS[i].name === name) return Promise.resolve(TOOLS[i].execute(args || {}));
      }
      return Promise.reject(new Error('unknown tool: ' + name));
    }
  };

  register(TOOLS);
})();
