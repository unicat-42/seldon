// «Архив» опубликованных выпусков и карточка записи прогноза (страна × выпуск × событие).
// Слева — что было опубликовано, справа — что произошло затем. Исход — только факт; иначе статус.
import { S, load, release, esc, pct, pctRange, num, signed, dateRu, href, nameOf, state, stateText, factorBar, download, cap, domainOf, DOMAIN, ARCHIVE_EVENT, archiveKey, archiveWindow, featValue, natural } from "../lib.js";

const FAMILY = { war: "Конфликты", fxa: "Валюта, год", fxm: "Валюта, месяц", bank: "Банки" };
const CHECK = { resolved: "Исход известен", ongoing: "Целевой период ещё идёт", awaiting: "Ожидаются данные об исходе" };

export async function archive(el, parts) {
  const open = parts[0];
  const vs = [...S.archive.vintages].reverse();
  const cur = S.current.release;
  el.innerHTML = `
  <section class="section" style="padding-top:30px">
    <div class="section-head"><div><h1>Архив опубликованных прогнозов</h1>
      <p>Выберите выпуск, чтобы увидеть опубликованные оценки. Исходы и последующие исправления показаны отдельно.</p></div></div>
    <div class="tbl-scroll"><table class="tbl">
      <thead><tr><th>Выпуск</th><th>Опубликован</th><th>Модель / схема</th><th>Целевые периоды</th><th>Состояние проверки</th></tr></thead>
      <tbody>
        ${vs.map((v) => { const st = [...new Set(v.models.flatMap((m) => m.check_status))];
          return `<tr><td class="name" style="padding-left:0"><a href="${href("archive/" + v.vintage)}">${cap(dateRu(v.vintage))}</a></td>
            <td class="num">${dateRu(v.published)}</td>
            <td>${v.models.length} оценок по ${v.rows} странам · схема ${v.schema_version}<br><span class="tiny faint mono">код ${esc(v.commit.slice(0, 10))}</span></td>
            <td class="small">${[...new Set(v.models.map((m) => `${FAMILY[m.family]}: ${archiveWindow(v, m)}`))].map(esc).join("<br>")}</td>
            <td class="small">${st.map((s) => `${CHECK[s]} — ${v.models.filter((m) => m.check_status.includes(s)).length}`).join("<br>")}</td></tr>`; }).join("")}
        <tr><td class="name faint" style="padding-left:0">Текущий расчёт</td><td class="num faint">${dateRu(cur.calculated)}</td>
          <td colspan="3" class="small muted">Не заморожен и в архив не входит. Следующий публичный выпуск появится здесь после заморозки снимка.</td></tr>
      </tbody></table></div>
  </section>
  ${vs.filter((v) => !open || v.vintage === open).map((v) => vintageBlock(v)).join("")}`;

  el.querySelectorAll("form[data-v]").forEach((f) => (f.onsubmit = (e) => {
    e.preventDefault();
    location.hash = href(`record/${f.dataset.v}/${f.col.value}/${f.iso.value}`);
  }));
}

function vintageBlock(v) {
  const withRecs = S.countries.filter((c) => v.countries[c.iso3]);
  return `<section class="section" id="v-${v.vintage}">
    <div class="section-head"><div><h2>Выпуск ${dateRu(v.vintage)}</h2>
      <p>Опубликован ${dateRu(v.published)} · ${v.rows} стран · ${esc(v.origin ?? "")}</p></div>
      <a class="btn ghost" href="${href("forecasts", { release: v.vintage })}">Открыть оценки выпуска</a></div>
    <div class="tbl-scroll"><table class="tbl">
      <thead><tr><th>Событие</th><th>Модель</th><th>Срез</th><th>Целевой период</th><th>Стран</th><th>Интервал</th><th>Состояние проверки</th></tr></thead>
      <tbody>${v.models.map((m) => `<tr><td style="max-width:38ch"><b style="font-weight:500">${esc(ARCHIVE_EVENT[archiveKey(m.column)][0])}</b>${m.role === "challenger" ? ' <span class="tag">модель-претендент</span>' : ""}
          <span class="st-detail">${esc(ARCHIVE_EVENT[archiveKey(m.column)][1])}</span></td>
        <td class="small muted">${esc(m.model)}</td><td class="small">${esc(/^\d{4}-\d{2}-\d{2}$/.test(m.as_of ?? "") ? dateRu(m.as_of) : (m.family === "fxm" ? "у каждой страны свой: последний месяц с курсом МВФ" : "1 января года целевого периода"))}</td>
        <td class="small num">${esc(archiveWindow(v, m))}</td><td class="num">${m.n_countries}</td><td>${state("no_interval")}</td>
        <td class="small">${m.check_status.map((s) => (s === "resolved" ? "Исход известен" : state(s))).join(" ")}${m.retrospective_note ? `<span class="st-detail">${esc(m.retrospective_note)}</span>` : ""}</td></tr>`).join("")}</tbody>
    </table></div>
    <div class="grid" style="margin-top:24px">
      <div class="c6"><div class="card"><h3>Открыть запись прогноза</h3>
        <p class="small muted" style="margin-top:4px">Одна запись — одна страна, одно событие, один выпуск.</p>
        <form data-v="${v.vintage}" class="controls" style="margin-top:12px">
          <select class="field" name="iso" aria-label="Страна">${withRecs.map((c) => `<option value="${c.iso3}" ${c.iso3 === "KAZ" ? "selected" : ""}>${esc(c.ru)}</option>`).join("")}</select>
          <select class="field" name="col" aria-label="Событие">${v.models.map((m) => `<option value="${m.column}">${esc(ARCHIVE_EVENT[archiveKey(m.column)][0])} · ${esc(archiveWindow(v, m))}</option>`).join("")}</select>
          <button class="btn sm">Открыть</button>
        </form></div></div>
      <div class="c6"><div class="card"><h3>Неизменность выпуска</h3>
        <dl class="kv"><dt>Файл оценок</dt><dd class="mono">${esc(v.files.predictions)}</dd>
          <dt>sha256 оценок</dt><dd class="mono">${esc(v.predictions_sha256)}</dd>
          <dt>Код</dt><dd class="mono">коммит ${esc(v.commit.slice(0, 12))}${v.uncommitted_changes ? " + незакоммиченные правки" : ""}; sha256 исходников ${esc(v.src_sha256.slice(0, 16))}…</dd>
          <dt>Версии данных</dt><dd class="small">${Object.entries(v.raw_versions).map(([k, x]) => `${esc(k)} ${esc(x)}`).join(" · ")}</dd></dl></div></div>
    </div>
    ${v.schema_notes.length ? `<div class="plate" style="margin-top:18px"><b>Сравнимость с другими выпусками.</b>
      <ul class="plain">${v.schema_notes.map((n) => `<li>${esc(n.note)}</li>`).join("")}</ul></div>` : ""}
  </section>`;
}

// ---------------------------------------------------------------- запись прогноза
export async function record(el, parts) {
  const [kind] = parts;
  let r;
  if (kind === "oos") r = await oosRecord(parts[1], +parts[2], parts[3]);
  else if (kind === "current") r = await currentRecord(parts[1], parts[2]);
  else r = await vintageRecord(parts[0], parts[1], parts[2]);

  el.innerHTML = `
  <section class="section dom-${r.dom}" style="padding-top:26px">
    <p class="small"><a href="${r.back[1]}">← ${esc(r.back[0])}</a></p>
    <div class="section-head" style="margin-top:10px"><div>
      <div class="eyebrow"><span class="dot"></span>Запись прогноза · ${esc(r.kindLabel)}</div>
      <h1 style="margin-top:6px">${esc(nameOf(r.iso))} · ${esc(r.title)}</h1>
      <p>${esc(r.event)}</p></div></div>
    ${r.banner ? `<div class="plate" style="margin-bottom:18px">${esc(r.banner)}</div>` : ""}
    <div class="two">
      <div><div class="eyebrow">${esc(r.leftTitle)}</div>
        <div class="big num" style="margin-top:10px">${pct(r.p)}</div>
        <dl class="kv">
          <dt>Интервал оценки</dt><dd>${r.lo != null ? `<span class="num">${pctRange(r.lo, r.hi)}</span>` : state("no_interval")}</dd>
          ${r.rank ? `<dt>Место</dt><dd class="num">${r.rank}${r.n ? ` из ${r.n}` : ""}</dd>` : ""}
          <dt>Целевой период</dt><dd class="num">${esc(r.window)}</dd>
          <dt>Информационная граница</dt><dd>${esc(r.asOf)}</dd>
          <dt>Модель</dt><dd class="small">${esc(r.model)}</dd>
          ${r.proof.map(([k, v]) => `<dt>${esc(k)}</dt><dd class="mono">${esc(v)}</dd>`).join("")}
        </dl>
        <h3 style="font:500 15px var(--sans);margin:18px 0 8px">Факторы</h3>
        ${r.factors ? factorsMini(r.factors) : `<p class="small muted">${esc(r.noFactors)}</p>`}
      </div>
      <div><div class="eyebrow">Что произошло затем</div>
        ${r.outcome ? `<div class="big" style="margin-top:10px;font-size:26px">${esc(r.outcome.text)}</div>
          <dl class="kv"><dt>Источник</dt><dd>${esc(r.outcome.source)}</dd><dt>Версия наблюдений</dt><dd>${esc(r.outcome.observed)}</dd><dt>Дата проверки</dt><dd>${esc(r.outcome.checked)}</dd></dl>
          ${r.outcome.note ? `<p class="plate small">${esc(r.outcome.note)}</p>` : ""}`
        : `<div style="margin-top:14px;font-size:19px;font-family:var(--serif)">${esc(stateText(r.status))}</div>
          <p class="small muted" style="margin-top:8px">${r.status === "ongoing" ? `Целевой период (${esc(r.window)}) не закончился. Исход появится здесь после его окончания и выхода данных наблюдений — с источником и версией.` : "Целевой период закончился, но данные наблюдений об исходе ещё не опубликованы."}</p>
          <p class="small faint" style="margin-top:8px">Пока исход неизвестен, здесь нет ни нуля, ни «не наступило».</p>`}
      </div>
    </div>
    <div class="links" style="margin-top:26px">
      <button class="btn ghost sm" id="link">Постоянная ссылка</button>
      <button class="btn ghost sm" id="json">Скачать запись</button>
      <button class="btn ghost sm" id="card">Сохранить карточку прогноза</button>
      <a href="${href("country/" + r.iso)}">Открыть страну</a>
    </div>
  </section>`;

  const data = { country: nameOf(r.iso), iso3: r.iso, record: r.kindLabel, release: r.title, event: r.event, probability: r.p, interval: r.lo != null ? [r.lo, r.hi] : null,
    interval_status: r.lo != null ? "ok" : "Интервал не рассчитан", rank: r.rank ?? null, n: r.n ?? null, target_period: r.window, as_of: r.asOf, model: r.model,
    outcome: r.outcome ?? null, status: r.outcome ? "resolved" : stateText(r.status), proof: Object.fromEntries(r.proof), url: location.href };
  el.querySelector("#link").onclick = async (e) => { try { await navigator.clipboard.writeText(location.href); e.target.textContent = "Ссылка скопирована"; } catch { prompt("Постоянная ссылка", location.href); } };
  el.querySelector("#json").onclick = () => download(`seldon_record_${r.iso}_${r.slug}.json`, JSON.stringify(data, null, 2));
  el.querySelector("#card").onclick = () => download(`seldon_card_${r.iso}_${r.slug}.svg`, cardSvg(r), "image/svg+xml");
}

function factorsMini(b) {
  const fs = [...b.features].sort((a, c) => Math.abs(c.contrib) - Math.abs(a.contrib)).slice(0, 8);
  const maxAbs = Math.max(...fs.map((f) => Math.abs(f.contrib)), 0.01);
  return `<table class="factors"><tbody>${fs.map((f) => `<tr class="${f.value === null ? "none" : ""}"><td>${esc(f.label)}<br><span class="dt">${f.value === null ? "нет наблюдения" : `${esc(featValue(f))}${f.date ? " · " + dateRu(f.date) : ""}`}</span></td>
    <td class="bar" style="min-width:120px">${factorBar(f.contrib, maxAbs, 160)}</td><td class="num">${signed(f.contrib, 2)}</td></tr>`).join("")}</tbody></table>
    <p class="tiny faint" style="margin-top:8px">Восемь наибольших по модулю вкладов, в лог-шансах (не в процентных пунктах). Вклады объясняют расчёт модели и не устанавливают причин.</p>`;
}

async function vintageRecord(vid, col, iso) {
  const v = S.archive.vintages.find((x) => x.vintage === vid);
  const m = v?.models.find((x) => x.column === col), rec = v?.records[col]?.[iso];
  if (!m) throw new Error("Такой записи в архиве нет.");
  const base = { iso, dom: m.family === "war" ? "war" : m.family === "bank" ? "bank" : "fx", kindLabel: "опубликованный выпуск", title: `выпуск ${dateRu(vid)}`,
    event: ARCHIVE_EVENT[archiveKey(col)][1], back: [`Выпуск ${dateRu(vid)}`, href("archive/" + vid)], slug: `${vid}_${col}`, leftTitle: "Опубликованный прогноз",
    model: m.model, noFactors: m.factors, factors: null, banner: null,
    proof: [["Выпуск опубликован", dateRu(v.published)], ["Код", `коммит ${v.commit.slice(0, 12)}`], ["sha256 оценок", v.predictions_sha256]] };
  if (!rec) {
    const c = v.countries[iso];
    const why = m.family === "bank" ? c.bank_status : (!["own", "no_imf_fx"].includes(c.fx_regime) ? `shared:${c.fx_regime}` : c.fx_regime === "no_imf_fx" ? "no_imf_fx" : "no_estimate_period");
    throw new Error(`В выпуске ${dateRu(vid)} по этому событию для страны «${nameOf(iso)}» оценки нет: ${stateText(why)}.`);
  }
  const asOf = /^\d{4}-\d{2}-\d{2}$/.test(m.as_of ?? "") ? dateRu(m.as_of) : (v.countries[iso].fx_as_of ? `данные по ${dateRu(v.countries[iso].fx_as_of)}` : m.as_of);
  const out = rec.status === "resolved" ? { text: rec.outcome ? "Событие наступило" : "Событие не наступило", source: m.outcome_source.source, observed: m.outcome_source.observed,
    checked: dateRu(m.outcome_source.checked_at), note: m.retrospective_note } : null;
  return { ...base, p: rec.p, lo: null, hi: null, rank: rec.rank, n: m.family === "war" ? null : m.n_countries, window: rec.window, asOf, outcome: out, status: rec.status };
}

async function currentRecord(ev, iso) {
  const rel = release("current"), c = rel.contexts[ev], e = rel.rows.get(iso)?.est[ev];
  if (!c || !e || e.p === undefined) throw new Error("По этому событию у страны нет числовой оценки в текущем расчёте.");
  const d = await load(`country/${iso}.json`);
  const key = ev.startsWith("war25") ? "war25" : ev.startsWith("war1000") ? "war1000" : ev;
  const row = rel.rows.get(iso);
  return { iso, dom: domainOf(ev), kindLabel: "текущий расчёт, ещё не опубликованный выпуск", title: `текущий расчёт от ${dateRu(rel.calculated)}`, event: c.event,
    back: [DOMAIN[domainOf(ev)] + " · прогнозы", href("forecasts", { event: ev, sel: iso })], slug: `current_${ev}`, leftTitle: "Текущая оценка",
    p: e.p, lo: e.lo ?? null, hi: e.hi ?? null, rank: e.rank, n: e.n, window: c.target_period, asOf: c.as_of_per_country ? `данные по ${dateRu(row.fx_as_of)}` : dateRu(c.as_of),
    model: c.model, factors: d.factors[key]?.kind === "logit" ? d.factors[key] : null, noFactors: "Модель на деревьях: линейных вкладов признаков у неё нет. Входные наблюдения — в досье страны.",
    banner: "Это текущий расчёт, а не опубликованный выпуск: он не заморожен и может измениться при пересчёте. В архив попадают только замороженные выпуски.",
    proof: [["Код", `коммит ${S.manifest.code.commit.slice(0, 12)}`], ["Данные собраны", dateRu(S.manifest.generated_at.slice(0, 10))]], outcome: null, status: "ongoing" };
}

async function oosRecord(contour, year, id) {
  const rib = await load(`oos/${contour}.json`);
  Object.assign(S.extraNames, rib.extra_names ?? {});
  const yr = rib.years.find((y) => y.release === year), r = yr?.rows.find((x) => x[0] === id);
  if (!r) throw new Error("В этом выпуске исторической проверки такой страны нет.");
  const det = await load(`oos/${contour}/${year}.json`), dr = det.rows[id];
  const feats = det.features.map((f, i) => ({ ...f, value: dr.miss.includes(i) ? null : dr.v[i], natural: dr.miss.includes(i) ? null : natural(f.nat, dr.v[i]), date: dr.d[i], contrib: dr.c[i] }));
  return { iso: id, dom: rib.domain, kindLabel: "ретроспективная OOS-проверка", title: `исторический выпуск ${yr.as_of.slice(0, 4)}`, event: rib.event,
    back: ["Главная · проверка прошлым", href("", { oos: contour, y: year, id })], slug: `oos_${contour}_${year}`, leftTitle: "Оценка по протоколу проверки вслепую",
    p: r[1], lo: r[2], hi: r[3], rank: r[4], n: yr.n, window: yr.target, asOf: dateRu(yr.as_of), model: rib.model,
    factors: { features: feats }, banner: "Исторические оценки рассчитаны по протоколу проверки на будущих для обучения наблюдениях. Они не были публичными прогнозами того времени.",
    proof: [["Обучение", `${det.n_train.toLocaleString("ru-RU")} наблюдений, базовая частота ${pct(det.base_rate)}`]],
    outcome: { text: r[5] ? "Событие наступило" : "Событие не наступило", source: rib.source, observed: `целевой период ${yr.target}`, checked: dateRu(S.manifest.today),
      note: "Исход взят из текущей версии наблюдений; он использован для проверки, но не для обучения этого выпуска." }, status: "resolved" };
}

function cardSvg(r) {
  const t = (s) => esc(s);
  const wrap = (s, n) => { const w = String(s).split(" "), lines = [""]; for (const x of w) { if ((lines.at(-1) + " " + x).length > n) lines.push(x); else lines[lines.length - 1] += (lines.at(-1) ? " " : "") + x; } return lines.slice(0, 4); };
  const ev = wrap(r.event, 78);
  const col = { war: "#b0563a", fx: "#456a8a", bank: "#746689" }[r.dom];
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" font-family="Georgia, serif">
  <rect width="1000" height="560" fill="#f6f1e6"/><rect width="1000" height="8" fill="${col}"/>
  <text x="56" y="70" font-size="20" fill="#57524a" font-family="Arial, sans-serif">Селдон · запись прогноза · ${t(r.kindLabel)}</text>
  <text x="56" y="130" font-size="44" font-weight="600" fill="#25231f">${t(nameOf(r.iso))} · ${t(r.title)}</text>
  ${ev.map((l, i) => `<text x="56" y="${176 + i * 26}" font-size="19" fill="#57524a" font-family="Arial, sans-serif">${t(l)}</text>`).join("")}
  <text x="56" y="350" font-size="96" font-weight="600" fill="#25231f">${t(pct(r.p))}</text>
  <text x="56" y="396" font-size="20" fill="#57524a" font-family="Arial, sans-serif">${r.lo != null ? "Интервал оценки: " + t(pctRange(r.lo, r.hi)) : "Интервал не рассчитан"}${r.rank ? ` · место ${r.rank}${r.n ? " из " + r.n : ""}` : ""}</text>
  <text x="56" y="430" font-size="20" fill="#57524a" font-family="Arial, sans-serif">Целевой период: ${t(r.window)} · информационная граница: ${t(r.asOf)}</text>
  <line x1="56" x2="944" y1="462" y2="462" stroke="#c3b8a1"/>
  <text x="56" y="500" font-size="22" fill="#25231f" font-family="Arial, sans-serif">${r.outcome ? "Что произошло затем: " + t(r.outcome.text.toLowerCase()) + " (" + t(r.outcome.source) + ")" : t(stateText(r.status))}</text>
  <text x="56" y="532" font-size="15" fill="#857f73" font-family="Arial, sans-serif">${t(r.model)}</text>
</svg>`;
}
