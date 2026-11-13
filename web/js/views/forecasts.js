// Скринер «Прогнозы»: P1 заголовок и выпуск, P2 управление, P3 таблица и инспектор.
// Всё состояние (выпуск, событие, группа, сортировка, поиск, выбранная страна) живёт в адресе.
import { S, load, niceMax, num, release, releaseList, esc, pct, dateRu, href, nameOf, state, stateText, stateDetail, estCell, scaleMax, scaleLegend,
  statesLegend, interval, ctxLine, EVENTS, DOMAIN, domainOf, ownWarEvent, setQuery, signed } from "../lib.js";

const DEFAULT_EVENT = { war: "war25_onset", fx: "fxa", bank: "bank" };
const norm = (s) => s.toLowerCase().replace(/ё/g, "е");
let keyRow = null;    // строка, выбранная с клавиатуры — вернуть на неё фокус после перерисовки
let typing = false;  // страница перерисовывается при вводе в поиск — вернуть фокус в поле

/** Параметры экрана из адреса — с подстановкой значений по умолчанию. */
export function readState(q) {
  const rel = release(q.release);
  let event = q.event && rel.contexts[q.event] ? q.event : null;
  const domain = event ? domainOf(event) : (DOMAIN[q.domain] ? q.domain : "war");
  if (!event) event = rel.contexts[DEFAULT_EVENT[domain]] ? DEFAULT_EVENT[domain] : EVENTS[domain].find((e) => rel.contexts[e]);
  return { rel, domain, event, group: q.group === "ps" ? "ps" : "world", sort: q.sort === "name" ? "name" : "p", search: q.q ?? "", sel: q.sel ?? null };
}
/** Параметры, которые переносятся в досье страны и обратно. */
export const carry = (st) => ({ release: st.rel.id === "current" ? null : st.rel.id, event: st.event, group: st.group === "ps" ? "ps" : null, sort: st.sort === "name" ? "name" : null });

export async function forecasts(el, _parts, q, { samePage }) {
  const st = readState(q);
  const { rel, domain, event } = st;
  const ctx = rel.contexts[event];
  const thr = event.startsWith("war1000") ? "war1000" : "war25";
  const part = event.endsWith("_cont") ? "cont" : "onset";
  // колонки: в активном домене — выбранное событие, в остальных — событие по умолчанию как контекст
  const cols = {
    war: domain === "war" ? event : null,  // null → «своя группа» страны (начало или продолжение)
    fx: domain === "fx" ? event : (rel.contexts.fxa ? "fxa" : null),
    bank: domain === "bank" ? event : (rel.contexts.bank ? "bank" : null),
  };
  const max = {};
  for (const ev of Object.keys(rel.contexts)) max[ev] = scaleMax(rel, ev);

  // строки: числовой рейтинг, затем именованные группы без оценки
  const all = [...rel.rows.values()].filter((r) => st.group === "world" || r.post_soviet)
    .filter((r) => !st.search || [nameOf(r.iso3), S.byIso.get(r.iso3)?.en ?? "", r.iso3, ...(S.byIso.get(r.iso3)?.aliases ?? [])].some((s) => norm(s).includes(norm(st.search))));
  const val = (r) => { const e = r.est[event]; return e?.p ?? e?.value ?? null; };
  const ranked = all.filter((r) => val(r) !== null && r.est[event]?.status === undefined);
  ranked.sort(st.sort === "name" ? (a, b) => nameOf(a.iso3).localeCompare(nameOf(b.iso3), "ru") : (a, b) => val(b) - val(a));
  const groups = new Map();
  for (const r of all) {
    const s = r.est[event]?.status ?? (val(r) === null ? "no_estimate_period" : null);
    if (s) (groups.get(s) ?? groups.set(s, []).get(s)).push(r);
  }
  const sel = st.sel && rel.rows.get(st.sel) ? st.sel : null;
  const carried = carry(st);

  const warCell = (r) => {
    if (cols.war) {
      const e = r.est[cols.war];
      const other = `${thr}_${part === "onset" ? "cont" : "onset"}`;
      return estCell(e, max[cols.war], { link: href("forecasts", { ...carried, event: other, sel: r.iso3 }) });
    }
    const ev = ownWarEvent(r);
    const e = r.est[ev];
    return e ? estCell(e, max[ev], { sub: ev.endsWith("cont") ? "идёт конфликт · продолжение" : "начало" }) : estCell(null);
  };
  const cell = (r, d) => {
    if (d === "war") return warCell(r);
    const ev = cols[d];
    if (!ev) return `<span class="st st-text">В этом выпуске не публиковалось</span>`;
    return estCell(r.est[ev], max[ev], { observed: !!rel.contexts[ev].observed });
  };
  const rowHtml = (r, i) => `<tr class="row ${r.iso3 === sel ? "sel" : ""} dom-${domain}" data-iso="${r.iso3}" tabindex="0" aria-selected="${r.iso3 === sel}">
      <td class="rk num">${i ?? ""}</td>
      <td class="name"><a href="${href("country/" + r.iso3, carried)}">${esc(nameOf(r.iso3))}</a>${r.post_soviet && st.group === "world" ? '<span class="ps-mark" title="постсоветская страна"></span>' : ""}</td>
      ${["war", "fx", "bank"].map((d) => `<td class="dom-${d} ${d === domain ? "" : "ctx-col"}">${cell(r, d)}</td>`).join("")}</tr>
      ${r.iso3 === sel ? `<tr class="exp dom-${domain}"><td colspan="5"><div class="exp-ctx">${["war", "fx", "bank"].filter((d) => d !== domain).map((d) => `<div class="dom-${d}"><div class="eyebrow"><span class="dot"></span>${DOMAIN[d]}</div>${cell(r, d)}</div>`).join("")}</div>
        <div class="exp-ins">${inspector(rel, st, sel, max, carried)}</div></td></tr>` : ""}`;
  const head = (d) => {
    const ev = d === "war" && !cols.war ? "war25_onset" : cols[d];
    const c = ev ? rel.contexts[ev] : null;
    const t = d === "war" && !cols.war ? "начало или продолжение, от 25 погибших" : c?.short;
    return `<th class="dom-${d} ${d === domain ? "active" : "ctx-col"}" scope="col"><span class="dot"></span>${DOMAIN[d]}${c ? `<small>${esc(t)}${c.target_period ? " · " + esc(c.target_period) : ""}</small>` : ""}</th>`;
  };

  el.innerHTML = `
  <section class="section" style="padding-top:30px">
    ${rel.archived ? `<div class="banner"><span>Архивный выпуск ${dateRu(rel.id)} · опубликован ${dateRu(rel.published)}</span><a href="${href("forecasts", { ...carried, release: null })}">Перейти к последнему</a></div>` : ""}
    <div class="section-head">
      <div><h1>Прогнозы по странам</h1><p>Выберите событие для сравнения. Вероятности по другим типам кризисов останутся рядом для сравнения.</p></div>
      <div class="ctl"><span>Выпуск</span><select class="field" id="rel" aria-label="Выпуск">${releaseList().map((r) => `<option value="${r.id}" ${r.id === rel.id ? "selected" : ""}>${esc(r.label)}</option>`).join("")}</select></div>
    </div>
    <div class="ctx">${rel.archived ? `Выпуск ${dateRu(rel.id)}, опубликован ${dateRu(rel.published)}.` : `${esc(rel.label)}, ещё не заморожен как публичный выпуск. Последний публичный — ${dateRu(S.current.release.last_public_vintage)}.`}</div>

    <div class="controls" style="margin-top:20px">
      <div class="ctl"><span>Сравниваем</span><div class="seg" role="group" aria-label="Сравниваем" id="dom">${Object.keys(DOMAIN).map((d) => `<button data-v="${d}" aria-pressed="${d === domain}" ${EVENTS[d].some((e) => rel.contexts[e]) ? "" : "disabled"}>${DOMAIN[d]}</button>`).join("")}</div></div>
      <div class="ctl"><span>Группа</span><div class="seg" role="group" aria-label="Группа" id="grp"><button data-v="world" aria-pressed="${st.group === "world"}">Мир</button><button data-v="ps" aria-pressed="${st.group === "ps"}">15 постсоветских стран</button></div></div>
      <div class="ctl"><label class="vh" for="tq">Найти страну в таблице</label><input class="field" id="tq" type="search" placeholder="Найти страну" value="${esc(st.search)}"></div>
    </div>
    <div class="controls" style="margin-top:12px">${secondRow(rel, domain, event, thr, part)}</div>
    <div class="plate ${domain}" style="margin-top:16px"><b>${esc(ctx.title)}.</b> ${esc(ctx.event)}.
      ${ctx.target_period ? ` Период прогноза: <b>${esc(ctx.target_period)}</b>.` : ""} Выборка: ${esc(ctx.sample)} (${ctx.n}).
      ${ctx.interval === "none" && !ctx.observed ? " Модель на деревьях: диапазон неопределённости не рассчитывается." : ""}
      ${ctx.interval === "unpublished" ? " В этом выпуске диапазоны неопределённости не публиковались." : ""}
      ${ctx.note ? " " + esc(ctx.note) : ""}${ctx.retro ? " " + esc(ctx.retro) : ""}</div>
  </section>

  <section style="margin-top:22px" class="dom-${domain}">
    <div class="grid">
      <div class="c9">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:6px">
          <div class="small">Сортировка по: <b>${st.sort === "name" ? "названию страны" : esc(ctx.title.toLowerCase())}</b>
            · <a href="#" id="sort">${st.sort === "name" ? "сортировать по оценке" : "сортировать по названию"}</a></div>
          ${scaleLegend()}
        </div>
        <div class="tbl-scroll"><table class="tbl main">
          <thead><tr><th scope="col" class="num" style="text-align:right;padding-right:12px">${ctx.observed ? "" : "Место"}</th><th scope="col">Страна</th>${head("war")}${head("fx")}${head("bank")}</tr></thead>
          <tbody>
            ${ranked.map((r) => rowHtml(r, ctx.observed ? null : r.est[event].rank)).join("") || `<tr><td colspan="5" class="muted" style="padding:18px 10px">В этой группе нет стран с числовой оценкой по выбранному событию.</td></tr>`}
            ${[...groups].map(([s, list]) => `<tr class="grp"><td colspan="5">${esc(stateText(s))} — ${list.length} <span class="faint" style="text-transform:none;letter-spacing:0">· ${esc(stateDetail(s) ?? "")}</span></td></tr>
              ${list.sort((a, b) => nameOf(a.iso3).localeCompare(nameOf(b.iso3), "ru")).map((r) => rowHtml(r, null)).join("")}`).join("")}
          </tbody>
        </table></div>
        <p class="small faint" style="margin-top:12px">Место считается в мировой выборке события и не меняется при выборе региональной группы. Шкала в колонке общая для всех стран.</p>
      </div>
      <aside class="c3 insp-side"><div class="inspector">${inspector(rel, st, sel, max, carried)}</div></aside>
    </div>
  </section>

  ${domain === "fx" ? await p4(rel, sel, carried) : ""}

  <section class="section"><div class="section-head"><div><h2>Как читать состояния</h2><p>Причина отсутствия оценки всегда написана словами. Отсутствие оценки не означает низкий риск.</p></div></div>${statesLegend()}</section>`;

  const go = (patch) => setQuery({ sel: null, ...patch });
  el.querySelector("#rel").onchange = (e) => go({ release: e.target.value === "current" ? null : e.target.value, event: null, domain });
  el.querySelector("#dom").onclick = (e) => { const b = e.target.closest("button"); if (b && !b.disabled) go({ domain: b.dataset.v, event: null }); };
  el.querySelector("#grp").onclick = (e) => { const b = e.target.closest("button"); if (b) setQuery({ group: b.dataset.v === "ps" ? "ps" : null }); };
  el.querySelectorAll("[data-ev]").forEach((b) => (b.onclick = () => { if (!b.disabled) go({ event: b.dataset.ev, domain: null }); }));
  el.querySelector("#sort").onclick = (e) => { e.preventDefault(); setQuery({ sort: st.sort === "name" ? null : "name" }); };
  const tq = el.querySelector("#tq");
  tq.oninput = () => { clearTimeout(tq._t); tq._t = setTimeout(() => { typing = true; setQuery({ q: tq.value || null }); }, 180); };
  if (samePage && typing) { tq.focus(); tq.setSelectionRange(tq.value.length, tq.value.length); }
  typing = false;
  el.querySelector("#p4")?.addEventListener("click", (e) => { const c = e.target.closest("[data-id]"); if (c) setQuery({ sel: c.dataset.id === sel ? null : c.dataset.id }); });
  el.querySelector("tbody").onclick = (e) => {
    if (e.target.closest("a")) return;
    const tr = e.target.closest("tr.row");
    if (tr) setQuery({ sel: tr.dataset.iso === sel ? null : tr.dataset.iso });
  };
  // выбор строки с клавиатуры: Enter или пробел на строке; после перерисовки фокус возвращается на неё
  el.querySelector("tbody").onkeydown = (e) => {
    if ((e.key !== "Enter" && e.key !== " ") || !e.target.matches("tr.row")) return;
    e.preventDefault();
    keyRow = e.target.dataset.iso;
    setQuery({ sel: keyRow === sel ? null : keyRow });
  };
  if (keyRow) { el.querySelector(`tr.row[data-iso="${keyRow}"]`)?.focus(); keyRow = null; }
}

function secondRow(rel, domain, event, thr, part) {
  const btn = (ev, label) => `<button data-ev="${ev}" aria-pressed="${ev === event}" ${rel.contexts[ev] ? "" : 'disabled title="В этом выпуске не публиковалось"'}>${label}</button>`;
  if (domain === "war") {
    return `<div class="ctl"><span>Что оцениваем</span><div class="seg" role="group" aria-label="Что оцениваем">${btn(`${thr}_onset`, "Начало")}${btn(`${thr}_cont`, "Продолжение")}</div></div>
      <div class="ctl"><span>Порог</span><div class="seg" role="group" aria-label="Порог">${btn(`war25_${part}`, "от 25 погибших за год")}${btn(`war1000_${part}`, "от 1000 погибших за год")}</div></div>`;
  }
  if (domain === "fx") {
    return `<div class="ctl"><span>Вид</span><div class="seg" role="group" aria-label="Вид">${btn("fxa", "Обвал за год")}${btn("fxm", "Скачок за месяц")}${btn("stress", "Давление сейчас")}</div></div>
      <div class="ctl"><span>Другие модели</span><div class="seg" role="group" aria-label="Другие модели">${btn("fxa_tree", "Обвал за год, вторая модель")}${btn("fxm_live", "Скачок за месяц, новая модель")}</div></div>`;
  }
  const b = (ev) => btn(ev, rel.contexts[ev]?.target_period ?? (ev === "bank" ? "основной период" : "прошлый период, для проверки"));
  return `<div class="ctl"><span>Период прогноза</span><div class="seg" role="group" aria-label="Период прогноза">${b("bank")}${btn("bank_check", (rel.contexts.bank_check?.target_period ?? "") + " · проверочный")}</div></div>`;
}

function inspector(rel, st, sel, max, carried) {
  const ctx = rel.contexts[st.event];
  if (!sel) return `<p class="muted">Выберите строку таблицы, чтобы увидеть оценку подробно: диапазон неопределённости, обычный уровень и место в списке.</p>
    <p class="small faint" style="margin-top:10px">Название страны открывает её досье. Выпуск, событие, группа и сортировка сохраняются при переходе и возврате.</p>`;
  const r = rel.rows.get(sel), e = r.est[st.event];
  const asOf = ctx.as_of_per_country ? r.fx_as_of : null;
  const body = !e || e.status
    ? `<div style="margin:12px 0">${state(e?.status ?? "no_estimate_period", { detail: true, link: href("forecasts", { ...carried, event: st.event.replace(/onset|cont/, (m) => (m === "onset" ? "cont" : "onset")), sel }) })}</div>
       <p class="small faint">Отсутствие оценки не означает низкий риск.</p>`
    : ctx.observed
      ? `<div class="big num" style="margin-top:10px">${signed(e.value, 1)} σ</div><p class="small muted" style="margin-top:8px">Наблюдаемый показатель на ${dateRu(r.fx_as_of)}. Это не вероятность кризиса.</p>`
      : `<div class="big num" style="margin-top:10px">${pct(e.p)}</div>
        <dl class="kv">
          <dt>Диапазон неопределённости</dt><dd>${interval(e)}</dd>
          <dt>${esc(rel.baseLabel)}</dt><dd class="num">${e.base_rate != null ? pct(e.base_rate) : "—"}</dd>
          <dt>Место</dt><dd class="num">${e.rank ?? "—"} из ${e.n}</dd>
          ${e.rec ? `<dt>Состояние проверки</dt><dd>${e.rec.status === "resolved" ? (e.rec.outcome ? "событие наступило" : "событие не наступило") : state(e.rec.status)}</dd>` : ""}
          ${e.recent_jump ? "<dt>Пометка</dt><dd>скачок курса уже был в последние 12 месяцев</dd>" : ""}
        </dl>`;
  const recHref = rel.archived && e?.rec ? href(`record/${rel.id}/${ctx.column}/${sel}`) : href(`record/current/${st.event}/${sel}`);
  return `<div class="eyebrow">${esc(DOMAIN[st.domain])}</div>
    <h3 style="margin-top:6px">${esc(nameOf(sel))}</h3>
    <p class="small muted" style="margin-top:4px">${esc(ctx.event)}</p>
    ${body}
    ${ctxLine(rel, st.event, { asOf })}
    <div class="links" style="margin-top:14px"><a class="btn sm" href="${href("country/" + sel, carried)}">Открыть страну</a>${e?.p !== undefined ? `<a href="${recHref}">Открыть запись прогноза</a>` : ""}</div>`;
}

// ---------------------------------------------------------------- P4: структура и давление
/** Валютный вид: по горизонтали — годовая вероятность обвала, по вертикали — текущее давление.
 *  Все точки одного размера; месячная вероятность выводится текстом. У выбранной страны — след
 *  последних месяцев: давление меняется помесячно, годовой прогноз — ступенькой 1 января, без сглаживания. */
async function p4(rel, sel, carried) {
  const cx = rel.contexts.fxa, cs = rel.contexts.stress;
  if (!cx || !cs) return `<section class="section" id="p4"><div class="section-head"><div><h2>Структурная уязвимость и текущее давление</h2></div></div>
    <p class="plate">В выпуске ${dateRu(rel.id)} текущее давление не публиковалось — этот вид доступен только для текущего расчёта.</p></section>`;
  const pts = [...rel.rows.values()].filter((r) => r.est.fxa?.p !== undefined && r.est.stress?.value != null)
    .map((r) => ({ id: r.iso3, x: r.est.fxa.p, y: r.est.stress.value, ps: r.post_soviet }));
  const W = 1180, H = 440, L = 62, R = 24, T = 22, B = 50, CAP = 8;
  const maxX = niceMax(Math.max(...pts.map((p) => p.x)));
  const y0 = Math.floor(Math.min(...pts.map((p) => p.y), -1)), y1 = Math.min(Math.ceil(Math.max(...pts.map((p) => p.y))), CAP);
  const X = (v) => L + Math.sqrt(Math.min(v, maxX) / maxX) * (W - L - R), Y = (v) => T + (1 - (Math.min(v, y1) - y0) / (y1 - y0)) * (H - T - B);
  const capped = pts.filter((p) => p.y > y1).length;
  let trail = "", note = "";
  const row = sel ? rel.rows.get(sel) : null;
  if (row && row.est.fxa?.p !== undefined && row.est.stress?.value != null) {
    const d = await load(`country/${sel}.json`);
    const Ty = cx.features_year;  // оценка на 1 января Ty+1 действует в месяцах года Ty+1; годом раньше — проверочная
    const seq = (d.emp ?? []).filter((m) => m[1] !== null).map(([m, v]) => ({ m, v, x: +m.slice(0, 4) > Ty ? row.est.fxa.p : row.est.fxa_check?.p }))
      .filter((p) => p.x !== undefined);
    if (seq.length > 1) {
      trail = `<polyline fill="none" stroke="var(--ink)" stroke-width="1.2" points="${seq.map((p) => `${X(p.x).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ")}"/>
        ${seq.map((p) => `<circle cx="${X(p.x)}" cy="${Y(p.v)}" r="2.2" fill="var(--ink)"><title>${dateRu(p.m)}: ${num(p.v, 1)} σ</title></circle>`).join("")}
        <text class="sp-lbl" x="${X(seq[0].x) + 7}" y="${Y(seq[0].v) + 4}">${dateRu(seq[0].m)}</text>`;
      const step = seq.find((p, i) => i && p.x !== seq[i - 1].x);
      note = `След: ${seq.length} месяцев, ${dateRu(seq[0].m)} – ${dateRu(seq.at(-1).m)}. Давление меняется каждый месяц; годовая оценка — ступенькой${step ? ` 1 января ${Ty + 1}: с ${pct(seq[0].x)} (оценка на 1 января ${Ty}, период ${Ty}–${Ty + 1}) до ${pct(row.est.fxa.p)}` : " не менялась"}. Линия между месяцами не сглажена.`;
    }
  }
  const yt = []; for (let v = y0; v <= y1; v += Math.max(1, Math.round((y1 - y0) / 6))) yt.push(v);
  const xt = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1].filter((t) => t <= maxX + 1e-9);
  const order = [...pts].sort((a, b) => (a.id === sel) - (b.id === sel) || a.ps - b.ps);
  const s = pts.find((p) => p.id === sel);
  return `<section class="section dom-fx" id="p4">
    <div class="section-head"><div><h2>Структурная уязвимость и текущее давление</h2>
      <p>Две разные величины на одном поле: вероятность — по горизонтали, наблюдаемое давление — по вертикали. Нажмите на точку, чтобы выбрать страну и увидеть её след.</p></div></div>
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;overflow:visible" role="img" aria-label="Вероятность годового обвала курса и текущее валютное давление по странам">
      ${yt.map((v) => `<line class="hp-grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="hp-axis" x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v > 0 ? "+" : ""}${v}</text>`).join("")}
      ${xt.map((v) => `<line class="hp-grid" x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H - B}" stroke-dasharray="2 4"/><text class="hp-axis" x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${num(100 * v, 0)}</text>`).join("")}
      <line x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--rule-2)"/>
      ${order.map((p) => (p.y > y1
        ? `<path data-id="${p.id}" d="M${X(p.x) - 4.5} ${T + 5}L${X(p.x) + 4.5} ${T + 5}L${X(p.x)} ${T - 3}Z" class="${p.id === sel ? "sp-sel" : p.ps ? "sp-ps" : "sp-dot"}" style="cursor:pointer"><title>${esc(nameOf(p.id))}: ${num(p.y, 1)} σ — выше шкалы</title></path>`
        : `<circle data-id="${p.id}" class="${p.id === sel ? "sp-sel" : p.ps ? "sp-ps" : "sp-dot"}" cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.id === sel ? 5.5 : 4}" style="cursor:pointer"><title>${esc(nameOf(p.id))}</title></circle>`)).join("")}
      ${trail}
      ${s ? `<text class="sp-name" x="${Math.min(X(s.x) + 9, W - 150)}" y="${Y(s.y) - 9}">${esc(nameOf(s.id))}</text>` : ""}
      <text class="hp-axis" x="${(W + L) / 2}" y="${H - 8}" text-anchor="middle">Вероятность годового обвала курса, %, ${esc(cx.target_period)} (шкала растянута у нуля)</text>
      <text class="hp-axis" transform="translate(14 ${(H - B + T) / 2}) rotate(-90)" text-anchor="middle">Текущее валютное давление, σ · ${esc(dateRu(cs.as_of.split(" … ")[0]))} – ${esc(dateRu(cs.as_of.split(" … ").pop()))}</text>
    </svg>
    <div class="legend" style="margin-top:8px"><span><svg viewBox="0 0 12 12" style="width:12px"><circle class="sp-dot" cx="6" cy="6" r="4"/></svg>страна мира</span>
      <span><svg viewBox="0 0 12 12" style="width:12px"><circle class="sp-ps" cx="6" cy="6" r="4"/></svg>постсоветская страна</span>
      <span><svg viewBox="0 0 12 12" style="width:12px"><circle class="sp-sel" cx="6" cy="6" r="4.5"/></svg>выбранная страна</span>
      ${capped ? `<span><svg viewBox="0 0 12 12" style="width:12px"><path class="sp-dot" d="M1.5 10L10.5 10L6 2Z"/></svg>давление выше +${y1} σ (${capped}): отметка у верхней кромки</span>` : ""}
      <span>все точки одного размера — третья величина размером не кодируется</span></div>
    ${s ? `<div class="plate fx" style="margin-top:14px"><b>${esc(nameOf(s.id))}.</b> Годовой обвал ${esc(cx.target_period)}: <span class="num">${pct(s.x)}</span>; давление на ${dateRu(row.fx_as_of)}: <span class="num">${num(s.y, 1)} σ</span>;
        скачок курса за месяц, на 12 месяцев: <span class="num">${row.est.fxm?.p !== undefined ? pct(row.est.fxm.p) : "нет оценки"}</span>. ${esc(note)}
        <a href="${href("country/" + s.id, carried)}">Открыть страну</a></div>`
      : `<p class="small faint" style="margin-top:10px">На поле ${pts.length} стран: те, у кого есть и годовая оценка, и свежее давление. Давление — наблюдение в стандартных отклонениях, не вероятность.</p>`}
  </section>`;
}
