// Общая библиотека витрины: данные, форматирование, словарь состояний, графики на SVG.
// Правило: на экран попадает только то, что лежит в web/data/*.json. Пропуск — это статус
// с названной причиной (компонент `state`), а не ноль и не прочерк.

// ---------------------------------------------------------------- данные
const cache = new Map();
export function load(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(`data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`Не найден файл data/${name}. Попробуйте обновить страницу; если не помогло, данные сайта нужно собрать заново (команда «make web»).`);
      return r.json();
    }));
  }
  return cache.get(name);
}

export const S = { current: null, countries: null, archive: null, manifest: null, byIso: new Map(), extraNames: {} };

export async function boot() {
  const [current, countries, archive, manifest] = await Promise.all(
    ["current.json", "countries.json", "archive.json", "manifest.json"].map(load));
  Object.assign(S, { current, countries, archive, manifest });
  countries.forEach((c) => S.byIso.set(c.iso3, c));
  return S;
}

export const nameOf = (id) => S.byIso.get(id)?.ru ?? S.extraNames[id] ?? id;
export const isPS = (id) => !!S.byIso.get(id)?.post_soviet;

// ---------------------------------------------------------------- форматирование
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const NB = " ";
const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
const MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];

export function num(x, d = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return null;
  return Number(x).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", "−");
}
/** Вероятность 0…1 → «1,6 %». Малые значения — с двумя знаками, чтобы не превращались в «0,0 %». */
export function pct(p) {
  if (p === null || p === undefined) return null;
  const v = 100 * p;
  return `${num(v, v < 0.1 ? 2 : 1)}${NB}%`;
}
export const pctRange = (lo, hi) => `${num(100 * lo, 100 * lo < 0.1 ? 2 : 1)}–${num(100 * hi, 1)}${NB}%`;
const plural = (n, one, few, many) => (n % 10 === 1 && n % 100 !== 11 ? one : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14) ? few : many);
/** Вероятность словами: 0,016 → «1 шанс из 60». Округление грубое намеренно: это ориентир, точное число стоит рядом. */
export function odds(p) {
  if (p === null || p === undefined) return null;
  if (p >= 0.95) return "больше 9 шансов из 10";
  if (p >= 0.2) { const k = Math.round(p * 10); return `${k} ${plural(k, "шанс", "шанса", "шансов")} из 10`; }
  const n = 1 / p;
  if (n >= 1000) return "меньше 1 шанса из 1000";
  const r = n < 20 ? Math.round(n) : n < 100 ? Math.round(n / 5) * 5 : Math.round(n / 50) * 50;
  return `1 шанс из ${r}`;
}
/** Строка «много это или мало»: шансы и обычный уровень (базовая частота события), без оценочных ступеней. */
export function oddsLine(e) {
  if (e?.p === undefined) return "";
  const base = e.base_rate != null ? ` Обычный уровень — ${odds(e.base_rate)}: так часто это событие случалось в прошлых данных модели.` : "";
  return `Примерно ${odds(e.p)}.${base}`;
}
export const signed = (x, d = 2) => (x > 0 ? "+" : "") + num(x, d);

/** «2026-09» → «сентябрь 2026», «2026-01-01» → «1 января 2026», «2025» → «2025». */
export function dateRu(s) {
  if (!s) return null;
  s = String(s);
  let m;
  if ((m = s.match(/^(\d{4})-(\d{2})-(\d{2})/))) return `${+m[3]}${NB}${MONTHS_GEN[+m[2] - 1]} ${m[1]}`;
  if ((m = s.match(/^(\d{4})-(\d{2})$/))) return `${MONTHS[+m[2] - 1]} ${m[1]}`;
  return s;
}
/** Значение признака словами: флаги — «да / нет», флаги пропуска — «ряд есть / ряда нет», остальное — число с единицей. */
export function featValue(f) {
  if (f.value === null || f.value === undefined) return null;
  if (f.unit === "флаг") return f.key.endsWith("_missing") ? (f.value ? "ряда нет" : "ряд есть") : (f.value ? "да" : "нет");
  if (f.natural) return `${num(f.natural.value, Math.abs(f.natural.value) >= 100 ? 0 : 1)}${NB}${f.natural.unit}`;
  return `${num(f.value, Number.isInteger(f.value) ? 0 : 2)}${NB}${f.unit}`;
}
/** То же преобразование «по-человечески», что в экспортере (`_natural`): для сжатых исторических вкладов. */
export function natural(spec, v) {
  if (!spec || v === null || v === undefined) return null;
  const [kind, unit] = [spec.slice(0, spec.indexOf(":")), spec.slice(spec.indexOf(":") + 1)];
  const value = { expm1: Math.expm1(v), exp: Math.exp(v), exp1: 100 * Math.expm1(v), pct: 100 * v, infl: 100 * Math.expm1(v) }[kind];
  return { value, unit };
}
export const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
/** Строчная первая буква — без порчи аббревиатур («Нет данных о кредите (BIS)» → «нет данных о кредите (BIS)»). */
export const low = (s) => (s && !/^[A-ZА-ЯЁ]{2}/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);

// ---------------------------------------------------------------- адрес и состояние
export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  return { parts: path.split("/").filter(Boolean).map(decodeURIComponent), q: Object.fromEntries(new URLSearchParams(query)) };
}
export function href(path, q = {}) {
  const p = new URLSearchParams(Object.entries(q).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  const s = p.toString();
  return `#/${path}${s ? "?" + s : ""}`;
}
/** Сменить параметры адреса без записи в историю (фильтры) или с записью (переходы). */
export function setQuery(patch, push = false) {
  const { parts, q } = parseHash();
  const url = href(parts.join("/"), { ...q, ...patch });
  if (push) location.hash = url; else { history.replaceState(null, "", url); window.dispatchEvent(new Event("hashchange")); }
}

// ---------------------------------------------------------------- выпуски
export const EVENTS = {
  war: ["war25_onset", "war25_cont", "war1000_onset", "war1000_cont"],
  fx: ["fxa", "fxm", "stress", "fxa_tree", "fxm_live"],
  bank: ["bank", "bank_check"],
};
export const DOMAIN = { war: "Конфликты", fx: "Валюта", bank: "Банки" };
export const DOMAIN_FULL = { war: "Вооружённые конфликты", fx: "Валютные кризисы", bank: "Банковские кризисы" };
export const domainOf = (ev) => Object.keys(EVENTS).find((d) => EVENTS[d].includes(ev));
const ARCHIVE_MAP = [[/^p_state_war25_/, "war25"], [/^fxa_logit_/, "fxa"], [/^fxa_extree_/, "fxa_tree"], [/^fx_deval15_12m$/, "fxm"],
  [/^fx_deval15_12m_live$/, "fxm_live"], [/^bank_jst_extree_full_/, "bank"]];

/** Человеческие названия событий архивных колонок. Определение из метаданных выпуска показывается рядом как есть. */
export const ARCHIVE_EVENT = {
  war25: ["Государственный вооружённый конфликт, ≥ 25 погибших", "Государственный вооружённый конфликт: ≥ 25 погибших за год на своей территории или у соседа, где страна — сторона"],
  war1000: ["Государственный вооружённый конфликт, ≥ 1000 погибших", "Государственный вооружённый конфликт высокой интенсивности: ≥ 1000 погибших за год"],
  fxa: ["Девальвация ≥ 25 % за год", "Девальвация ≥ 25 % за год (курс к доллару, декабрь к декабрю) после ≥ 2 спокойных лет"],
  fxa_tree: ["Девальвация ≥ 25 % за год, вторая модель", "Девальвация ≥ 25 % за год (курс к доллару, декабрь к декабрю) после ≥ 2 спокойных лет"],
  fxm: ["Скачок курса ≥ 15 % за месяц", "Скачок курса к доллару ≥ 15 % за один месяц (первый после ≥ 12 спокойных месяцев)"],
  fxm_live: ["Скачок курса ≥ 15 % за месяц, новая модель на испытании", "Скачок курса к доллару ≥ 15 % за один месяц (первый после ≥ 12 спокойных месяцев)"],
  bank: ["Системный банковский кризис", "Системный банковский кризис в одном из двух лет периода прогноза (по метке Bluwstein и др.)"],
};
export const archiveKey = (column) => (/^p_state_war1000_/.test(column) ? "war1000" : ARCHIVE_MAP.find(([re]) => re.test(column))?.[1]);
/** Период прогноза архивной оценки — словами, без служебных обозначений. */
export function archiveWindow(v, m) {
  if (m.family === "fxm") return "12 месяцев после даты прогноза страны";
  return Object.values(v.records[m.column])[0]?.window ?? m.window;
}

export function releaseList() {
  const cur = S.current.release;
  return [{ id: "current", label: `Текущий расчёт · ${dateRu(cur.calculated)}`, short: "текущий расчёт" },
    ...[...S.archive.vintages].reverse().map((v) => ({ id: v.vintage, label: `Выпуск ${dateRu(v.vintage)}`, short: dateRu(v.vintage) }))];
}

/** Единая модель выпуска для экранов: строки стран с оценками и контексты событий. */
export function release(id = "current") {
  if (id === "current" || !S.archive.vintages.some((v) => v.vintage === id)) {
    const c = S.current;
    return { id: "current", archived: false, label: `Текущий расчёт · ${dateRu(c.release.calculated)}`, published: null,
      calculated: c.release.calculated, note: c.release.note, rows: new Map(c.rows.map((r) => [r.iso3, r])), contexts: c.contexts,
      baseLabel: "Обычный уровень: частота события в обучающей выборке" };
  }
  const v = S.archive.vintages.find((x) => x.vintage === id);
  const cur = S.current.contexts;
  const contexts = {}, rows = new Map();
  for (const iso of Object.keys(v.countries)) {
    const c = v.countries[iso];
    rows.set(iso, { iso3: iso, post_soviet: isPS(iso), est: {}, fx_regime: c.fx_regime, bank_status: c.bank_status, fx_as_of: c.fx_as_of,
      war_now: { state_war25: c.state_war25_now } });
  }
  for (const m of v.models) {
    const key = ARCHIVE_MAP.find(([re]) => re.test(m.column))?.[1];
    if (!key) continue;
    const recs = v.records[m.column];
    const base = m.out_of_sample?.base_rate ?? null;
    const mk = (k, n) => {
      const part = k.endsWith("_onset") ? "Начало: " : k.endsWith("_cont") ? "Продолжение: " : "";
      contexts[k] = { ...cur[k], event: part + (part ? low(ARCHIVE_EVENT[key][1]) : ARCHIVE_EVENT[key][1]), target_period: archiveWindow(v, m),
        as_of: /^\d{4}-\d{2}-\d{2}$/.test(m.as_of ?? "") ? m.as_of : cur[k].as_of, n, model: m.model, interval: "unpublished",
        column: m.column, check: m.check_status, retro: m.retrospective_note, source: null, features_year: null };
    };
    if (key === "war25") {
      const groups = { onset: [], cont: [] };
      for (const [iso, r] of Object.entries(recs)) groups[r.group === "cont" ? "cont" : "onset"].push([iso, r]);
      for (const part of ["onset", "cont"]) {
        const list = groups[part].sort((a, b) => b[1].p - a[1].p);
        list.forEach(([iso, r], i) => {
          const row = rows.get(iso);
          row.est[`war25_${part}`] = { p: r.p, rank: r.rank ?? i + 1, n: list.length, base_rate: part === "onset" ? base : null, rec: r };
          row.est[`war25_${part === "onset" ? "cont" : "onset"}`] = { status: part === "cont" ? "other_group" : "no_conflict" };
        });
        mk(`war25_${part}`, list.length);
      }
      continue;
    }
    const n = Object.keys(recs).length;
    for (const [iso, row] of rows) {
      const r = recs[iso];
      if (r) row.est[key] = { p: r.p, rank: r.rank, n, base_rate: base, rec: r };
      else if (key === "bank") row.est[key] = { status: row.bank_status === "ok" ? "no_estimate_period" : row.bank_status };
      else row.est[key] = { status: !["own", "no_imf_fx"].includes(row.fx_regime) ? `shared:${row.fx_regime}` : row.fx_regime === "no_imf_fx" ? "no_imf_fx" : "no_estimate_period" };
    }
    mk(key, n);
  }
  return { id, archived: true, label: `Выпуск ${dateRu(id)}`, published: v.published, vintage: v, rows, contexts,
    baseLabel: "Обычный уровень: частота события в проверке вслепую" };
}

/** Событие «своей группы» страны в домене конфликтов: начало или продолжение. */
export function ownWarEvent(row, thr = "war25") {
  const e = row.est[`${thr}_onset`];
  return e && e.p !== undefined ? `${thr}_onset` : `${thr}_cont`;
}

// ---------------------------------------------------------------- словарь состояний
const CUR = { EUR: "евро", USD: "доллар США", XOF: "франк КФА (Западная Африка)", XAF: "франк КФА (Центральная Африка)", XCD: "восточнокарибский доллар", AUD: "австралийский доллар", CHF: "швейцарский франк" };
export const STATE_KIND = {
  no_bis_credit: "nodata", no_yield_curve: "nodata", no_broad_money: "nodata", no_imf_fx: "nodata", no_recent_fx: "nodata",
  out_of_sample: "nodata", no_estimate_period: "nodata", other_group: "group", no_conflict: "group",
  no_interval: "text", ongoing: "text", awaiting: "text",
};
export function stateText(status) {
  if (status?.startsWith("shared:")) return `не оценивается: общая валюта (${status.slice(7)})`;
  return S.current.status_text[status] ?? status;
}
export function stateDetail(status) {
  if (status?.startsWith("shared:")) {
    const c = status.slice(7);
    return `Прогноз обвала собственной валюты неприменим: страна использует ${CUR[c] ?? c}.`;
  }
  return S.current.status_detail[status] ?? null;
}
/** Единый компонент состояния. Текст причины виден сразу, без наведения. */
export function state(status, { detail = false, link = null } = {}) {
  let html;
  if (status?.startsWith("shared:")) html = `<span class="st st-na"><b>${esc(status.slice(7))}</b>общая валюта</span>`;
  else {
    const kind = STATE_KIND[status] ?? "nodata";
    const text = esc(stateText(status));
    if (kind === "group") html = `<span class="st st-group">${link ? `<a href="${link}">${text}</a>` : text}</span>`;
    else html = `<span class="st st-${kind}">${text}</span>`;
  }
  const d = detail ? stateDetail(status) : null;
  return d ? `${html}<span class="st-detail">${esc(d)}</span>` : html;
}
export function statesLegend() {
  const items = [
    ["<span class=\"num\" style=\"font-weight:500\">1,0&nbsp;%</span>", "Низкая вероятность — это число. Она показывается точкой на шкале, как любая другая оценка."],
    [state("shared:EUR"), "Не оценивается: у страны общая или чужая валюта, прогноза обвала собственной валюты нет."],
    [state("no_bis_credit"), "Нет данных, на которых построена модель. Штриховка всегда означает отсутствие ряда, а не низкий риск."],
    [state("no_imf_fx"), "Нет курса валюты в данных МВФ."],
    [state("no_yield_curve") + " " + state("no_broad_money"), "Нет рыночных ставок или собственной денежной массы для банковской модели."],
    [state("other_group"), "Страна относится к другой условной группе и в числовой рейтинг начала не входит."],
    [state("no_interval"), "Оценка показана без диапазона: для этой модели или этого выпуска интервал не определён."],
    [state("ongoing"), "Период прогноза не закончился — исхода ещё нет."],
    [state("awaiting"), "Период закончился, но данные наблюдений ещё не вышли."],
  ];
  return `<div class="states-legend">${items.map(([a, b]) => `<div>${a}<div>${esc(b)}</div></div>`).join("")}</div>`;
}

// ---------------------------------------------------------------- оценка: число + мини-шкала
export function niceMax(v) {
  const steps = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1];
  return steps.find((s) => s >= v) ?? 1;
}
/** Общий максимум шкалы события в выпуске — чтобы мини-шкалы всех стран были сравнимы. */
export function scaleMax(rel, ev) {
  let m = 0;
  for (const r of rel.rows.values()) { const e = r.est[ev]; if (e?.p !== undefined) m = Math.max(m, e.hi ?? e.p); }
  return niceMax(m);
}
export function miniScale(e, max, w = 120) {
  const x = (v) => 2 + Math.min(1, v / max) * (w - 4);
  const ci = e.lo !== undefined ? `<line class="ms-ci" x1="${x(e.lo)}" x2="${x(e.hi)}" y1="9" y2="9"/>` : "";
  const base = e.base_rate != null ? `<line class="ms-base" x1="${x(e.base_rate)}" x2="${x(e.base_rate)}" y1="3" y2="15"/>` : "";
  return `<svg viewBox="0 0 ${w} 18" preserveAspectRatio="none" aria-hidden="true"><line class="ms-axis" x1="2" x2="${w - 2}" y1="9" y2="9"/>${ci}${base}<circle class="ms-dot" cx="${x(e.p)}" cy="9" r="3.4"/></svg>`;
}
/** Ячейка оценки: число и шкала либо состояние из словаря. */
export function estCell(e, max, { sub = "", link = null, observed = false } = {}) {
  if (!e) return `<span class="st st-nodata">Нет оценки</span>`;
  if (e.status) return state(e.status, { link });
  if (observed || e.value !== undefined) return `<span class="num" style="font-weight:500">${e.value === null ? "—" : signed(e.value, 1)}${NB}σ</span>`;
  return `<div class="est"><span class="v num">${pct(e.p)}</span>${miniScale(e, max)}${sub ? `<span class="sub">${sub}</span>` : ""}</div>`;
}
export const scaleLegend = () => `<div class="legend" aria-label="Как читать мини-шкалу">
  <span><svg viewBox="0 0 26 12"><line class="ms-axis" x1="0" x2="26" y1="6" y2="6"/><circle class="ms-dot" cx="13" cy="6" r="3.4"/></svg>вероятность</span>
  <span><svg viewBox="0 0 26 12"><line class="ms-ci" x1="2" x2="24" y1="6" y2="6"/></svg>диапазон неопределённости, если рассчитан</span>
  <span><svg viewBox="0 0 26 12"><line class="ms-axis" x1="0" x2="26" y1="6" y2="6"/><line class="ms-base" x1="13" x2="13" y1="0" y2="12"/></svg>обычный уровень</span>
</div>`;
export const interval = (e) => (e.lo !== undefined ? `<span class="num">${pctRange(e.lo, e.hi)}</span>` : state("no_interval"));

/** Постоянная строка контекста у каждого прогноза. */
export function ctxLine(rel, ev, { asOf = null } = {}) {
  const c = rel.contexts[ev];
  if (!c) return "";
  const when = asOf ?? c.as_of;
  const cut = c.as_of_per_country && !asOf ? `даты прогноза по странам: ${dateRu(when.split(" … ")[0])} – ${dateRu(when.split(" … ").pop())}` : `дата прогноза: ${dateRu(when)}`;
  const period = c.target_period ? ` · период прогноза <b>${esc(c.target_period)}</b>` : "";
  return `<div class="ctx">${esc(rel.label)} · ${esc(cut)} · событие: <b>${esc(c.event)}</b>${period}</div>`;
}

// ---------------------------------------------------------------- графики
/** Полоска распределения: каждая страна — точка, точки одного значения складываются стопкой.
 *  Шкала — квадратного корня (подписана): оценки редких событий теснятся у нуля, и на линейной
 *  оси они слились бы в один столбик. Ось общая для всех стран события. */
export function strip(points, { max, sel = null, w = 760, r = 3, label = "Вероятность, %" } = {}) {
  const padL = 10, padR = 16, x = (v) => padL + Math.sqrt(Math.min(v, max) / max) * (w - padL - padR);
  const bins = new Map();
  const sorted = [...points].sort((a, b) => (a.ps === b.ps ? 0 : a.ps ? 1 : -1));
  let top = 0;
  const dots = sorted.map((p) => {
    const b = Math.round(x(p.v) / (2 * r));
    const k = bins.get(b) ?? 0;
    bins.set(b, k + 1);
    top = Math.max(top, k);
    return { ...p, cx: b * 2 * r, k };
  });
  const base = 24 + top * 2 * r + r;
  const hgt = base + 40;
  const ticks = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1].filter((t) => t <= max + 1e-9);
  const cls = (p) => (p.id === sel ? "sp-sel" : p.ps ? "sp-ps" : "sp-dot");
  const order = [...dots].sort((a, b) => (a.id === sel) - (b.id === sel));
  const s = dots.find((d) => d.id === sel);
  const name = s ? `<text class="sp-name" x="${Math.min(Math.max(s.cx, 60), w - 70)}" y="12" text-anchor="middle">${esc(nameOf(s.id))} · ${pct(s.v)}</text>
     <line x1="${s.cx}" x2="${s.cx}" y1="16" y2="${base - s.k * 2 * r - r - 2}" stroke="var(--ink)" stroke-width="1"/>` : "";
  return `<svg viewBox="0 0 ${w} ${hgt}" role="img" aria-label="${esc(label)}">
    ${name}
    ${order.map((p) => `<circle class="${cls(p)}" data-id="${p.id}" cx="${p.cx}" cy="${base - p.k * 2 * r}" r="${p.id === sel ? r + 1.2 : r}"><title>${esc(nameOf(p.id))}</title></circle>`).join("")}
    <line class="sp-axis" x1="${padL}" x2="${w - padR}" y1="${base + r + 4}" y2="${base + r + 4}"/>
    ${ticks.map((t) => `<line class="sp-axis" x1="${x(t)}" x2="${x(t)}" y1="${base + r + 4}" y2="${base + r + 8}"/><text class="sp-lbl" x="${x(t)}" y="${base + r + 20}" text-anchor="middle">${num(100 * t, 0)}</text>`).join("")}
    <text class="sp-lbl" x="${padL}" y="${base + r + 35}">${esc(label)} · шкала растянута у нуля, общая для всех стран</text>
  </svg>`;
}
export const stripLegend = (dom = "war") => `<div class="legend dom-${dom}">
  <span><svg viewBox="0 0 12 12" style="width:12px"><circle class="sp-dot" cx="6" cy="6" r="3.4"/></svg>другая страна мира</span>
  <span><svg viewBox="0 0 12 12" style="width:12px"><circle class="sp-ps" cx="6" cy="6" r="3.4"/></svg>страна региональной группы</span>
  <span><svg viewBox="0 0 12 12" style="width:12px"><circle class="sp-sel" cx="6" cy="6" r="4"/></svg>выбранная страна (подписана)</span>
</div>`;

/** Лента рангов: строка — исторический выпуск, отметка — страна, слева — более высокие оценки. */
export function ribbon(rib, sel, { blind = false } = {}) {
  const W = 800, padL = 44, padR = 10, rowH = 15, top = 30;
  const years = rib.years;
  const H = top + years.length * rowH + 30;
  const out = [];
  years.forEach((y, i) => {
    const cy = top + i * rowH + rowH / 2;
    const step = (W - padL - padR) / y.n;
    out.push(`<text class="rb-year" x="${padL - 8}" y="${cy + 4}" text-anchor="end">${y.as_of.slice(0, 4)}</text>`);
    let ticks = "", evs = "", hits = "";
    y.rows.forEach((r, k) => {
      const x = padL + (k + 0.5) * step;
      ticks += `M${x.toFixed(1)} ${cy - 4.5}V${cy + 4.5}`;
      if (r[5] && !blind) evs += `<circle class="rb-ev" cx="${x.toFixed(1)}" cy="${cy}" r="3"/>`;
      hits += `<rect class="rb-hit" data-y="${y.release}" data-id="${r[0]}" x="${(x - step / 2).toFixed(1)}" y="${cy - rowH / 2}" width="${step.toFixed(2)}" height="${rowH}"><title>${esc(nameOf(r[0]))} · выпуск ${y.as_of.slice(0, 4)}</title></rect>`;
      if (sel && String(sel.y) === String(y.release) && sel.id === r[0]) evs += `<circle class="rb-sel" cx="${x.toFixed(1)}" cy="${cy}" r="6"/>`;
    });
    out.push(`<path class="rb-tick" d="${ticks}"/>${evs}${hits}`);
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(rib.title)}: положение стран в рейтинге каждого исторического выпуска">
    <text class="rb-axis" x="${padL}" y="14">← Более высокие оценки</text>
    <text class="rb-axis" x="${W - padR}" y="14" text-anchor="end">Более низкие оценки →</text>
    ${out.join("")}
    <text class="rb-axis" x="${(W + padL) / 2}" y="${H - 8}" text-anchor="middle">Положение в рейтинге выпуска</text>
  </svg>`;
}

/** Расходящаяся от нуля полоса вклада признака (лог-шансы). */
export function factorBar(c, maxAbs, w = 240) {
  const mid = w / 2, len = (Math.abs(c) / maxAbs) * (mid - 4);
  const x = c >= 0 ? mid : mid - len;
  return `<svg viewBox="0 0 ${w} 16" preserveAspectRatio="none" aria-hidden="true"><rect class="${c >= 0 ? "fb-pos" : "fb-neg"}" x="${x}" y="3" width="${Math.max(len, 0.5)}" height="10"/><line class="fb-zero" x1="${mid}" x2="${mid}" y1="0" y2="16"/></svg>`;
}

export function sparkline(series, { w = 300, h = 64, unit = "σ" } = {}) {
  const pts = series.filter((d) => d[1] !== null);
  if (pts.length < 2) return "";
  const vs = pts.map((d) => d[1]), lo = Math.min(0, ...vs), hi = Math.max(1, ...vs);
  const x = (i) => 4 + (i / (series.length - 1)) * (w - 44), y = (v) => 8 + (1 - (v - lo) / (hi - lo)) * (h - 26);
  const d = series.map((p, i) => (p[1] === null ? null : `${x(i).toFixed(1)} ${y(p[1]).toFixed(1)}`)).filter(Boolean).map((s, i) => (i ? "L" : "M") + s).join("");
  const last = series.length - 1;
  return `<svg viewBox="0 0 ${w} ${h}" style="max-width:${w}px;overflow:visible" role="img" aria-label="Давление по месяцам, ${unit}">
    <line class="hp-grid" x1="4" x2="${w - 40}" y1="${y(0)}" y2="${y(0)}"/>
    <path class="hp-line" d="${d}"/>
    ${series[last][1] !== null ? `<circle class="hp-pt" cx="${x(last)}" cy="${y(series[last][1])}" r="3"/><text class="hp-axis" x="${x(last) + 6}" y="${y(series[last][1]) + 4}">${signed(series[last][1], 1)} ${unit}</text>` : ""}
    <text class="hp-axis" x="4" y="${h - 2}">${dateRu(series[0][0])}</text><text class="hp-axis" x="${w - 40}" y="${h - 2}" text-anchor="end">${dateRu(series[last][0])}</text>
  </svg>`;
}

export function download(filename, text, type = "application/json") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
}
export function toCSV(rows) {
  const q = (v) => (v === null || v === undefined ? "" : /[",;\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return rows.map((r) => r.map(q).join(",")).join("\n") + "\n";
}
