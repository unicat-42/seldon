// Досье страны. Порядок секций фиксирован: C1 шапка, C2 покрытие, C3 профиль, C4 история,
// C5 факторы, C6 происхождение.
import { S, load, release, releaseList, esc, pct, num, signed, dateRu, href, nameOf, state, stateText, stateDetail, estCell, scaleMax,
  interval, ctxLine, oddsLine, strip, niceMax, factorBar, sparkline, ownWarEvent, setQuery, download, toCSV, DOMAIN_FULL, low, featValue } from "../lib.js";
import { carry, readState } from "./forecasts.js";

const FACTOR_TABS = [["war25", "Война, от 25", "war"], ["war1000", "Большая война, от 1000", "war"], ["fxa", "Обвал курса за год", "fx"],
  ["fxm", "Скачок за месяц", "fx"], ["fxm_live", "Скачок за месяц, новая модель", "fx"], ["fxa_tree", "Обвал за год, вторая модель", "fx"], ["bank", "Банки", "bank"]];

export async function country(el, parts, q) {
  const iso = (parts[0] ?? "").toUpperCase();
  if (!S.byIso.has(iso)) throw new Error(`Страны с кодом «${iso}» нет среди 174 государств витрины.`);
  const st = readState(q), rel = st.rel, row = rel.rows.get(iso);
  const d = await load(`country/${iso}.json`);
  const cur = release("current"), crow = cur.rows.get(iso);
  const back = carry(st);
  const warEv = ownWarEvent(row), war1000Ev = row.est.war1000_onset ? ownWarEvent(row, "war1000") : null;
  const max = (ev) => scaleMax(rel, ev);
  const est = (ev, opt) => (rel.contexts[ev] ? estCell(row.est[ev], max(ev), opt) : `<span class="st st-text">В этом выпуске не публиковалось</span>`);
  // простой слой — шансы и обычный уровень; точный (интервал, место, определение события) — под «Подробнее»
  const more = (ev, extra = "", int = null) => { const e = row.est[ev];
    if (e?.p === undefined) return e?.status ? `<span class="st-detail">${esc(stateDetail(e.status) ?? "")}</span>` : "";
    return `<p class="odds">${esc(oddsLine(e))}</p><details class="more"><summary>Подробнее: интервал, место в списке, определение события</summary>
      <div class="small muted">${int ?? `Диапазон неопределённости: ${interval(e)}`} · место ${e.rank ?? "—"} из ${e.n}</div>${extra}</details>`; };

  // --- C2: статусы покрытия ---
  const covState = (dom) => {
    if (dom === "war") return ["Оценивается", warEv.endsWith("cont") ? "идёт конфликт — оценивается продолжение" : "конфликта нет — оценивается начало"];
    if (dom === "fx") {
      const e = row.est.fxa, m = row.est.fxm;
      if (e?.p !== undefined) return ["Оценивается", m?.p !== undefined ? "обвал за год, скачок за месяц и давление сейчас" : `обвал за год; скачок за месяц — ${low(stateText(m?.status))}`];
      if (m?.p !== undefined) return ["Оценивается частично", `скачок за месяц и давление; обвал за год — ${low(stateText(e?.status))}`];
      return [stateText(e?.status), stateDetail(e?.status)];
    }
    const e = row.est.bank;
    return e?.p !== undefined ? ["Оценивается", "кредитный цикл и кривая доходности"] : [stateText(e?.status), stateDetail(e?.status)];
  };

  // --- C3: мировые полоски ---
  const world = [[warEv, "war"], ["fxa", "fx"], ["fxm", "fx"], ["bank", "bank"]].filter(([ev]) => rel.contexts[ev] && row.est[ev]?.p !== undefined).map(([ev, dom]) => {
    const pts = [...rel.rows].filter(([, r]) => r.est[ev]?.p !== undefined).map(([id, r]) => ({ id, v: r.est[ev].p, ps: false }));
    return `<div class="strip dom-${dom}" style="margin-top:14px"><h3><span class="dot"></span>${esc(rel.contexts[ev].title)}</h3>
      ${strip(pts, { max: niceMax(Math.max(...pts.map((p) => p.v))), sel: iso, w: 420, r: 2.6 })}
      <p class="tiny faint num">место ${row.est[ev].rank} из ${row.est[ev].n} · ${esc(rel.contexts[ev].sample)}</p></div>`;
  }).join("");
  const ind = row.indicators;
  const indItem = (k, label, f) => (ind?.[k] ? `<div><span class="faint">${label}</span><b class="num">${ind[k].value === null ? "нет данных" : f(ind[k].value)}</b><span class="tiny faint">${ind[k].date ? dateRu(ind[k].date) : ""}</span></div>` : "");

  // --- C5: факторы ---
  const tabs = FACTOR_TABS.filter(([k]) => d.factors[k] || d.inputs[k]);
  const ftab = tabs.some(([k]) => k === q.f) ? q.f : tabs[0]?.[0];

  el.innerHTML = `
  <section class="section" style="padding-top:26px" id="top">
    <p class="small"><a href="${href("forecasts", { ...back, sel: iso })}">← К прогнозам</a></p>
    ${rel.archived ? `<div class="banner" style="margin-top:10px"><span>Архивный выпуск ${dateRu(rel.id)} · опубликован ${dateRu(rel.published)}</span><a href="${href("country/" + iso, { ...back, release: null })}">Перейти к последнему</a></div>` : ""}
    <div class="section-head" style="margin-top:10px">
      <div><h1>${esc(nameOf(iso))}</h1><p>Оценки модели, факторы и история проверок</p></div>
      <div class="links">
        <div class="ctl"><span>Выпуск</span><select class="field" id="rel" aria-label="Другой выпуск">${releaseList().map((r) => `<option value="${r.id}" ${r.id === rel.id ? "selected" : ""}>${esc(r.label)}</option>`).join("")}</select></div>
        <a class="btn ghost" href="${href("forecasts", { ...back, sel: iso })}">Сравнить страны</a>
      </div>
    </div>
    <div class="ctx">${esc(rel.label)}${rel.archived ? `, опубликован ${dateRu(rel.published)}` : ", ещё не заморожен как публичный выпуск"}. ${esc(S.byIso.get(iso).en)}, ${iso}</div>
  </section>
  <nav class="anchors" aria-label="Разделы досье"><a href="#est" data-jump>Оценки</a><a href="#hist" data-jump>История</a><a href="#fact" data-jump>Факторы</a><a href="#prov" data-jump>Данные</a></nav>

  <section class="section" style="padding-top:28px">
    <div class="section-head"><div><h2>Что модель оценивает для этой страны</h2></div><p class="small" style="font-weight:500">Отсутствие оценки не означает низкий риск</p></div>
    <div class="cov3">${["war", "fx", "bank"].map((dom) => { const [a, b] = covState(dom); return `<div class="dom-${dom}"><div class="eyebrow">${DOMAIN_FULL[dom]}</div><div class="state">${esc(a)}</div><div class="small muted">${esc(b ?? "")}</div></div>`; }).join("")}</div>
  </section>

  <section class="section" id="est">
    <div class="section-head"><div><h2>Вероятности в выбранном выпуске</h2></div></div>
    <div class="grid">
      <div class="c7">
        <div class="dom-block dom-war"><h3><span class="dot"></span>Вооружённые конфликты</h3>
          <div class="sub-block"><div class="t">${warEv.endsWith("cont") ? "Продолжение войны" : "Начало войны"}, от 25 погибших</div><div>${est(warEv)}${more(warEv, ctxLine(rel, warEv))}</div></div>
          ${war1000Ev ? `<div class="sub-block"><div class="t">${war1000Ev.endsWith("cont") ? "Продолжение большой войны" : "Начало большой войны"}, от 1000 погибших</div><div>${est(war1000Ev)}${more(war1000Ev, ctxLine(rel, war1000Ev))}</div></div>` : ""}
          ${warEv.endsWith("cont") ? `<p class="small muted" style="margin-top:8px">${state("other_group")} — в рейтинг начала конфликта страна не входит.</p>` : ""}
        </div>
        <div class="dom-block dom-fx"><h3><span class="dot"></span>Валютные кризисы</h3>
          <div class="sub-block"><div class="t">Обвал курса за год</div><div>${est("fxa")}${more("fxa", ctxLine(rel, "fxa") + (row.est.fxa_tree?.p !== undefined ? `<p class="small muted" style="margin-top:6px">То же событие, вторая модель: <span class="num">${pct(row.est.fxa_tree.p)}</span> · ${state("no_interval")}</p>` : ""))}</div></div>
          <div class="sub-block"><div class="t">Резкий скачок за месяц</div><div>${est("fxm")}${more("fxm", ctxLine(rel, "fxm", { asOf: row.fx_as_of }) + (row.est.fxm_live?.p !== undefined ? `<p class="small muted" style="margin-top:6px"><span class="tag">новая модель на испытании</span> <span class="num">${pct(row.est.fxm_live.p)}</span>, интервал ${interval(row.est.fxm_live)} — публикуется рядом для сравнения и основную оценку не заменяет.</p>` : ""))}</div></div>
          <div class="sub-block"><div class="t">Давление на валюту сейчас</div><div>
            ${row.est.stress?.value !== undefined ? `<div><span class="big num" style="font-size:24px">${signed(row.est.stress.value, 1)} σ</span> <span class="small muted">на ${dateRu(row.fx_as_of)} · наблюдение, не вероятность</span></div>
              ${!rel.archived && d.emp ? sparkline(d.emp) : ""}` : est("stress")}
            ${ind ? `<div class="ind">${indItem("deval12", "Курс за 12 месяцев", (v) => signed(100 * v, 1) + " %")}${indItem("reserves_g12", "Резервы за 12 месяцев", (v) => signed(100 * v, 1) + " %")}${indItem("infl12", "Инфляция за 12 месяцев", (v) => num(100 * v, 1) + " %")}
              ${indItem("peg_share12", "Привязка курса (доля спокойных месяцев)", (v) => num(v, 2))}${indItem("reer_peg", "Реальное укрепление × привязка", (v) => num(v, 3))}${indItem("peg_resfall", "Привязка × падение резервов", (v) => num(v, 3))}</div>
              <p class="tiny faint" style="margin-top:8px">Индикаторы режима курса — наблюдения, а не вероятности. Пустое значение означает отсутствие ряда, а не ноль.</p>` : ""}
          </div></div>
        </div>
        <div class="dom-block dom-bank"><h3><span class="dot"></span>Банковские кризисы</h3>
          <div class="sub-block"><div class="t">Системный кризис</div><div>${row.est.bank?.status ? state(row.est.bank.status, { detail: true }) : est("bank")}
            ${row.est.bank?.p !== undefined ? more("bank", ctxLine(rel, "bank"), state("no_interval")) : ""}</div></div>
        </div>
      </div>
      <aside class="c5"><div class="card"><h3>В мировом контексте</h3><p class="small muted" style="margin-top:4px">Положение страны среди всех стран с оценкой по тому же событию. Между событиями шкалы не сравниваются.</p>
        ${world || `<p class="muted" style="margin-top:12px">Ни по одному событию у страны нет числовой оценки в этом выпуске.</p>`}</div></aside>
    </div>
  </section>

  <section class="section" id="hist">
    <div class="section-head"><div><h2>Как менялись оценки и что происходило затем</h2>
      <p>Оценки исторической проверки вслепую: каждая рассчитана только по данным, доступным на её дату. Нажмите на точку, чтобы увидеть её окно прогноза. <span class="tag dark">Проверка на прошлом, вслепую</span></p></div></div>
    <div class="hist" id="histbox">${history(d, q.hp)}</div>
  </section>

  <section class="section" id="fact">
    <div class="section-head"><div><h2>Что повлияло на оценку модели</h2>
      <p>Вклады признаков в выбранный прогноз. Они объясняют расчёт модели и не устанавливают причины будущего события.</p></div></div>
    ${rel.archived ? `<p class="plate">В выпуске ${dateRu(rel.id)} вклады признаков не публиковались. Ниже — разбор текущего расчёта.</p>` : ""}
    ${tabs.length ? `<div class="seg" id="ftabs" role="group" aria-label="Модель" style="margin:14px 0">${tabs.map(([k, t]) => `<button data-v="${k}" aria-pressed="${k === ftab}">${t}</button>`).join("")}</div>
      <div class="dom-${tabs.find(([k]) => k === ftab)[2]}">${factors(d, ftab)}</div>` : `<p class="muted">Ни одна модель не даёт оценки для этой страны.</p>`}
  </section>

  <section class="section" id="prov">
    <div class="section-head"><div><h2>Данные и ограничения этого прогноза</h2></div></div>
    <div class="prov">
      ${[["war25_onset", "Конфликты"], ["fxa", "Валюта, обвал за год"], ["fxm", "Валюта, скачок за месяц"], ["bank", "Банки"]].filter(([k]) => cur.contexts[k]).map(([k, t]) => { const c = cur.contexts[k];
        return `<div><span>${t}: дата прогноза и данные</span><span>${k === "fxm" ? (crow.fx_as_of ? dateRu(crow.fx_as_of) : "нет даты прогноза") : dateRu(c.as_of)} · ${esc(c.source)}</span></div><div><span>Модель</span><span>${esc(c.model)}</span></div>`; }).join("")}
      <div><span>Лаг годовых макроданных</span><span>1 год: на 1 января известен только предыдущий год</span></div>
      <div><span>Лаг месячных резервов МВФ</span><span>${S.current.fx_monthly.reserves_lag} месяца</span></div>
      <div><span>Версия кода</span><span class="mono">${esc(S.manifest.code.commit.slice(0, 12))}${S.manifest.code.uncommitted_changes ? " + незакоммиченные правки" : ""}</span></div>
      <div><span>Данные собраны</span><span>${dateRu(S.manifest.generated_at.slice(0, 10))}</span></div>
    </div>
    <ul class="plain small">
      <li>Диапазон неопределённости отражает неопределённость коэффициентов модели и не учитывает ошибку её спецификации; это не интервал предсказания события.</li>
      <li>Лаги публикации месячных рядов МВФ, кроме резервов (цены, деньги, экспорт, ставки), в моделях не учтены: у таких наблюдений стоит месяц даты прогноза.</li>
      <li>История оценок — ретроспективная проверка, а не прогнозы, опубликованные в то время.</li>
    </ul>
    <div class="links" style="margin-top:18px">
      <a class="btn ghost sm" href="${href(`record/current/${warEv}/${iso}`)}">Открыть запись прогноза</a>
      <a href="data/country/${iso}.json" download="seldon_${iso}.json">Скачать данные (JSON)</a>
      <a href="#" id="csv">Скачать оценки (CSV)</a>
      <a href="${href("method")}">Метод и данные</a>
    </div>
  </section>`;

  el.querySelector("#rel").onchange = (e) => setQuery({ release: e.target.value === "current" ? null : e.target.value }, true);
  el.querySelectorAll("[data-jump]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); document.getElementById(a.getAttribute("href").slice(1))?.scrollIntoView(); }));
  el.querySelector("#ftabs")?.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setQuery({ f: b.dataset.v }); });
  el.querySelector("#histbox").onclick = (e) => { const p = e.target.closest("[data-hp]"); if (p) setQuery({ hp: p.dataset.hp === q.hp ? null : p.dataset.hp }); };
  el.querySelector("#csv").onclick = (e) => {
    e.preventDefault();
    const rows = [["iso3", "country", "release", "event", "target_period", "probability", "interval_low", "interval_high", "rank", "n", "status"]];
    for (const [ev, x] of Object.entries(row.est)) {
      if (!rel.contexts[ev]) continue;
      rows.push([iso, nameOf(iso), rel.id, ev, rel.contexts[ev].target_period, x.p ?? x.value ?? null, x.lo ?? null, x.hi ?? null, x.rank ?? null, x.n ?? null, x.status ?? (x.lo === undefined && x.p !== undefined ? "no_interval" : "ok")]);
    }
    download(`seldon_${iso}_${rel.id}.csv`, toCSV(rows), "text/csv");
  };
}

// ---------------------------------------------------------------- C4: история
function history(d, hp) {
  const H = d.history ?? {};
  const x0 = 1995, x1 = 2028, W = 1180, padL = 150, padR = 20;
  const x = (yr) => padL + ((yr - x0) / (x1 - x0)) * (W - padL - padR);
  const mx = (m) => x(+m.slice(0, 4) + (+m.slice(5, 7) - 0.5) / 12);
  const panels = [];
  const [hk, hv] = (hp ?? "").split(":");
  let info = "";
  const frame = (title, sub, dom, body, y0, hgt, max, evTrack, evLabel) => {
    const ticks = [0, max / 2, max];
    const yy = (v) => y0 + hgt - (Math.min(v, max) / max) * hgt;
    return `<g class="dom-${dom}"><text class="hp-title" x="${padL}" y="${y0 - 14}">${esc(title)}<tspan class="hp-axis" dx="10">${esc(sub)}</tspan></text>
      ${ticks.map((t) => `<line class="hp-grid" x1="${padL}" x2="${W - padR}" y1="${yy(t)}" y2="${yy(t)}"/><text class="hp-axis" x="${padL - 6}" y="${yy(t) + 4}" text-anchor="end">${num(100 * t, max < 0.1 ? 1 : 0)} %</text>`).join("")}
      ${body(yy)}
      <line class="hp-track" x1="${padL}" x2="${W - padR}" y1="${y0 + hgt + 16}" y2="${y0 + hgt + 16}"/>
      <text class="hp-axis" x="${padL - 6}" y="${y0 + hgt + 20}" text-anchor="end">${esc(evLabel)}</text>${evTrack(y0 + hgt + 16)}</g>`;
  };
  let y = 8;
  const add = (svg) => { panels.push(svg); y += 150; };
  for (const [key, title, thr] of [["war25", "Война, от 25 погибших", "от 25"], ["war1000", "Большая война, от 1000 погибших", "от 1000"]]) {
    const s = H[key];
    if (!s || (key === "war1000" && !s.some((r) => r[3] === 1) && Math.max(...s.map((r) => r[1])) < 0.02)) continue;
    const max = niceMax(Math.max(...s.map((r) => r[1])));
    add(frame(title, "начало ● · продолжение ○, не соединяются", "war", (yy) => s.map((r) => {
      const id = `${key}:${r[0]}`, on = hp === id;
      if (on) info = `<b>${esc(title)}.</b> Выпуск ${r[0]}, дата прогноза: ${dateRu(`${r[0]}-01-01`)} → период прогноза ${r[0]}: оценка ${r[2] ? "начала" : "продолжения"} — <b class="num">${pct(r[1])}</b>. Исход: ${r[3] ? "государственная война в " + r[0] + " году была" : "государственной войны в " + r[0] + " году не было"}.`;
      return `${on ? `<rect class="hp-win" x="${x(r[0])}" y="${y}" width="${x(r[0] + 1) - x(r[0])}" height="118"/>` : ""}<circle class="hp-pt ${r[2] ? "" : "open"}" data-hp="${id}" cx="${x(r[0] + 0.5)}" cy="${yy(r[1])}" r="${on ? 5 : 3.2}"><title>${r[0]}: ${pct(r[1])}</title></circle>`;
    }).join(""), y + 34, 62, max, (ty) => s.filter((r) => r[3]).map((r) => `<rect class="hp-ev" x="${x(r[0]) + 1}" y="${ty - 4}" width="${x(r[0] + 1) - x(r[0]) - 2}" height="8"/>`).join(""), `война ${thr}`));
  }
  if (H.fxa) {
    const s = H.fxa, max = niceMax(Math.max(...s.map((r) => r[1])));
    add(frame("Валюта: обвал курса за год", "девальвация ≥ 25 % в одном из двух следующих лет", "fx", (yy) => s.map((r) => {
      const id = `fxa:${r[0]}`, on = hp === id;
      if (on) info = `<b>Валюта, обвал курса за год.</b> Выпуск ${r[0] + 1}, дата прогноза: ${dateRu(`${r[0] + 1}-01-01`)} → период прогноза ${r[0] + 1}–${r[0] + 2}: оценка <b class="num">${pct(r[1])}</b>. Исход: ${r[2] ? "девальвация ≥ 25 % в периоде прогноза была" : "девальвации ≥ 25 % в периоде прогноза не было"}.`;
      return `${on ? `<rect class="hp-win" x="${x(r[0] + 1)}" y="${y}" width="${x(r[0] + 3) - x(r[0] + 1)}" height="118"/>` : ""}<circle class="hp-pt" data-hp="${id}" cx="${x(r[0] + 1)}" cy="${yy(r[1])}" r="${on ? 5 : 3.2}"><title>выпуск ${r[0] + 1}: ${pct(r[1])}</title></circle>`;
    }).join(""), y + 34, 62, max, (ty) => (H.fxa_events ?? []).map((r) => `<rect class="hp-ev" x="${x(r[0]) + 1}" y="${ty - 4}" width="${x(r[0] + 1) - x(r[0]) - 2}" height="8"><title>${r[0]}: курс +${num(r[1], 0)} % за год</title></rect>`).join(""), "девальвация ≥ 25 %"));
  }
  if (H.fxm) {
    const s = H.fxm, max = niceMax(Math.max(...s.map((r) => r[1])));
    add(frame("Валюта: скачок курса за месяц", "скачок курса ≥ 15 % за месяц в следующие 12 месяцев", "fx", (yy) => {
      // ряд месячный; там, где страна выпадала из оценки, линия рвётся — пропуски не соединяются
      let dd = "", prev = null;
      for (const r of s) { const t = +r[0].slice(0, 4) * 12 + +r[0].slice(5, 7); dd += `${prev !== null && t - prev === 1 ? "L" : "M"}${mx(r[0]).toFixed(1)} ${yy(r[1]).toFixed(1)}`; prev = t; }
      let mark = "";
      if (hk === "fxm") { const r = s.find((z) => z[0] === hv); if (r) { const end = `${+hv.slice(0, 4) + 1}-${hv.slice(5, 7)}`;
        info = `<b>Валюта, скачок курса за месяц.</b> Дата прогноза: ${dateRu(hv)} → период прогноза 12 месяцев, по ${dateRu(end)}: оценка <b class="num">${pct(r[1])}</b>. Исход: ${r[2] ? "скачок курса ≥ 15 % в окне был" : "скачка курса ≥ 15 % в окне не было"}.`;
        mark = `<rect class="hp-win" x="${mx(hv)}" y="${y}" width="${x(1) - x(0)}" height="118"/><circle class="hp-pt" cx="${mx(hv)}" cy="${yy(r[1])}" r="5"/>`; } }
      return `<path class="hp-line" d="${dd}"/>${mark}${s.filter((_, i) => i % 3 === 0).map((r) => `<rect data-hp="fxm:${r[0]}" x="${mx(r[0]) - 1.5}" y="${y + 30}" width="3.2" height="70" fill="transparent" style="cursor:pointer"><title>${dateRu(r[0])}: ${pct(r[1])}</title></rect>`).join("")}`;
    }, y + 34, 62, max, (ty) => (H.fxm_events ?? []).map((m) => `<circle class="hp-ev" cx="${mx(m)}" cy="${ty}" r="3.5"><title>${dateRu(m)}: скачок курса ≥ 15 %</title></circle>`).join(""), "скачок ≥ 15 %"));
  }
  if (H.bank) {
    const s = H.bank, max = niceMax(Math.max(...s.map((r) => r[1])));
    add(frame("Банки: системный кризис", "кризис в одном из двух следующих лет", "bank", (yy) => s.map((r) => {
      const id = `bank:${r[0]}`, on = hp === id;
      if (on) info = `<b>Банковский прогноз.</b> Выпуск ${r[0] + 1} → период прогноза ${r[0] + 1}–${r[0] + 2}: оценка <b class="num">${pct(r[1])}</b>. Исход: ${r[2] ? "год отмечен как предкризисный" : "кризиса в периоде прогноза не отмечено"}.`;
      return `${on ? `<rect class="hp-win" x="${x(r[0] + 1)}" y="${y}" width="${x(r[0] + 3) - x(r[0] + 1)}" height="118"/>` : ""}<circle class="hp-pt" data-hp="${id}" cx="${x(r[0] + 1)}" cy="${yy(r[1])}" r="${on ? 5 : 3.2}"/>`;
    }).join(""), y + 34, 62, max, (ty) => s.filter((r) => r[2]).map((r) => `<rect class="hp-ev" x="${x(r[0] + 1)}" y="${ty - 4}" width="${x(r[0] + 3) - x(r[0] + 1)}" height="8" opacity=".55"/>`).join(""), "окно кризиса"));
  }
  if (!panels.length) return `<p class="muted">Исторических оценок вслепую для этой страны нет: ни одна модель не оценивала её в прошлые годы.</p>`;
  const years = []; for (let t = 1996; t <= 2026; t += 5) years.push(t);
  const missing = [["fxa", "прогноза обвала курса за год"], ["fxm", "прогноза скачка курса за месяц"], ["bank", "банковского прогноза"]].filter(([k]) => !H[k]).map(([, t]) => t);
  return `<svg viewBox="0 0 ${W} ${y + 22}" role="img" aria-label="История оценок вслепую и зарегистрированных событий">
      ${years.map((t) => `<line class="hp-grid" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${y}" stroke-dasharray="2 4"/><text class="hp-axis" x="${x(t)}" y="${y + 14}" text-anchor="middle">${t}</text>`).join("")}
      ${panels.join("")}</svg>
    <p class="plate" style="margin-top:10px">${info || "Нажмите на точку — подсветится её окно прогноза и появится исход. Тёмные отметки на нижней дорожке — зарегистрированные события."}</p>
    ${missing.length ? `<p class="small faint" style="margin-top:8px">Нет истории: ${esc(missing.join(", "))} — модель не оценивала страну в прошлые годы.</p>` : ""}`;
}

// ---------------------------------------------------------------- C5: факторы
function factors(d, key) {
  const b = d.factors[key], t = d.inputs[key];
  if (t) {
    return `<p class="plate"><b>${esc(t.model)}.</b> ${esc(t.note)} Оценка: <span class="num">${pct(t.p)}</span> · ${state("no_interval")}</p>
      <table class="factors" style="margin-top:12px"><thead><tr><th scope="col">Входное наблюдение</th><th scope="col">Значение</th><th scope="col">Дата наблюдения</th><th scope="col">Источник</th></tr></thead><tbody>
      ${t.inputs.map((f) => `<tr class="${f.value === null ? "none" : ""}"><td>${esc(f.label)}</td><td class="num obs">${obs(f)}</td><td class="dt">${when(f)}</td><td class="dt">${esc(f.source)}</td></tr>`).join("")}</tbody></table>`;
  }
  if (b.kind !== "logit") return `<p class="plate">${esc(b.note)}</p>`;
  const fs = [...b.features].sort((a, c) => Math.abs(c.contrib) - Math.abs(a.contrib));
  const maxAbs = Math.max(...fs.map((f) => Math.abs(f.contrib)), 0.01);
  const main = fs.filter((f) => f.value !== null && !f.key.endsWith("_missing") && !f.label.startsWith("Флаг") && !(f.unit === "флаг" && !f.value) && Math.abs(f.contrib) >= 0.05).slice(0, 5);
  const side = (list) => (list.length ? `<ul class="plain">${list.map((f) => `<li><b>${esc(f.label)}</b> — ${esc(featValue(f))}${f.date ? ` <span class="tiny faint">(${dateRu(f.date)})</span>` : ""}</li>`).join("")}</ul>`
    : `<p class="small muted" style="margin-top:8px">Среди главных признаков таких нет.</p>`);
  return `<div class="two shift"><div><div class="eyebrow">Сдвигает оценку вниз</div>${side(main.filter((f) => f.contrib < 0))}</div>
      <div><div class="eyebrow">Сдвигает оценку вверх</div>${side(main.filter((f) => f.contrib > 0))}</div></div>
    <p class="small muted" style="margin-top:12px;max-width:80ch">Показаны ${main.length} из ${fs.length} признаков модели с наибольшим вкладом. Это описание расчёта, а не причин будущего события.</p>
    <details class="more" style="margin-top:14px"><summary>Все признаки и величина вклада</summary>
    <div class="ctx" style="margin:10px 0">Модель: <b>${esc(b.model)}</b>. Вероятность <b class="num">${pct(b.p)}</b>, диапазон неопределённости <span class="num">${num(100 * b.lo, 1)}–${num(100 * b.hi, 1)} %</span>. Обучение: ${b.n_train.toLocaleString("ru-RU")} наблюдений, обычный уровень ${pct(b.base_rate)}.</div>
    <table class="factors"><thead><tr><th scope="col">Признак</th><th scope="col">← снижает оценку · повышает →</th><th scope="col">Вклад, лог-шансы</th><th scope="col">Наблюдение</th><th scope="col">Дата наблюдения</th></tr></thead><tbody>
    ${fs.map((f) => `<tr class="${f.value === null ? "none" : ""}"><td>${esc(f.label)}</td><td class="bar">${factorBar(f.contrib, maxAbs)}</td><td class="num">${signed(f.contrib, 2)}</td>
      <td class="num obs">${obs(f)}</td><td class="dt">${when(f)}</td></tr>`).join("")}
    <tr><td colspan="2" class="muted">Свободный член модели</td><td class="num">${signed(b.intercept, 2)}</td><td colspan="2" class="dt">уровень при средних значениях всех признаков</td></tr>
    <tr><td colspan="2"><b>Сумма: лог-шансы прогноза</b></td><td class="num"><b>${signed(b.eta, 2)}</b></td><td colspan="2" class="dt">соответствует вероятности ${pct(b.p)}</td></tr>
    </tbody></table>
    <p class="small faint" style="margin-top:10px;max-width:90ch">Вклад — произведение коэффициента на стандартизованное значение признака, в лог-шансах, а не в процентных пунктах вероятности. Ноль означает, что значение равно среднему по обучающей выборке. Вклады складываются только на шкале лог-шансов.</p></details>`;
}
function obs(f) {
  if (f.value === null) return `<span class="st st-nodata">Нет наблюдения</span>`;
  return esc(featValue(f)) + (f.clipped !== undefined ? ` <span class="tiny faint">(в модели обрезано до ${num(f.clipped, 2)})</span>` : "");
}
function when(f) {
  if (f.value === null) return esc(f.missing ?? f.date_note ?? "нет наблюдения");
  return `${f.date ? dateRu(f.date) : "—"}${f.date_note ? `<br><span class="tiny">${esc(f.date_note)}</span>` : ""}`;
}
