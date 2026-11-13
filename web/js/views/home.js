// Главная: H1 введение, H2 лента проверки прошлым, H3 текущие оценки, H4 региональная линза, H5 метод.
import { S, load, release, esc, pct, pctRange, dateRu, href, nameOf, isPS, state, stateText, strip, stripLegend, ribbon,
  niceMax, DOMAIN_FULL, setQuery, low } from "../lib.js";

const RIBBONS = { war25: "Начало конфликта", fxa: "Девальвация ≥ 25 %" };
const DOM = { war25: "war", fxa: "fx" };

export async function home(el, _parts, q) {
  const contour = RIBBONS[q.oos] ? q.oos : "war25";
  const rib = await load(`oos/${contour}.json`);
  Object.assign(S.extraNames, rib.extra_names ?? {});
  const sel = q.y && q.id ? { y: +q.y, id: q.id } : null;
  const rel = release("current");
  const cur = S.current, cov = cur.coverage;
  const lastV = cur.release.last_public_vintage;
  const lens = q.lens && isPS(q.lens) ? q.lens : null;

  el.innerHTML = `
  <section class="hero">
    <div class="hero-row">
      <div>
        <div class="eyebrow">Открытое исследование кризисов</div>
        <h1>Прогнозы кризисов, которые можно проверить</h1>
        <p class="lede">Вооружённые конфликты, валютные и банковские кризисы. Вероятности для стран мира, объяснения модели и проверка на прошлых событиях.</p>
      </div>
      <div>
        <div class="links"><a class="btn" href="${href("forecasts")}">Исследовать прогнозы</a><a href="#how">Как мы проверяем модель</a></div>
        <p class="note">Мировой охват · отдельный фокус на 15 постсоветских странах</p>
      </div>
    </div>
  </section>

  <section class="section dom-${DOM[contour]}" id="past" style="padding-top:30px">
    <div class="section-head">
      <div>
        <h2>Где находились будущие кризисы в прогнозах модели</h2>
        <p>Каждая строка — исторический выпуск. Страны расположены от более высокой оценки к более низкой; тёмные отметки показывают наступившие события.</p>
      </div>
      <div class="seg" role="group" aria-label="Событие">${Object.entries(RIBBONS).map(([k, t]) => `<button data-oos="${k}" aria-pressed="${k === contour}">${t}</button>`).join("")}</div>
    </div>
    <div class="grid">
      <div class="c8 ribbon">
        <div class="ctx" style="margin-bottom:8px"><span class="tag dark">Ретроспективная OOS-проверка</span>
          &nbsp; событие: <b>${esc(rib.event)}</b> · горизонт ${esc(rib.horizon)} · выпуски ${rib.years[0].as_of.slice(0, 4)}–${rib.years.at(-1).as_of.slice(0, 4)}</div>
        ${ribbon(rib, sel)}
        <p class="small faint" style="margin-top:6px">Положение в рейтинге не равно вероятности события. Год слева — год выпуска: оценка сделана на 1 января этого года по данным, доступным к этой дате.</p>
      </div>
      <aside class="c4" id="rb-side">${side(rib, contour, sel)}</aside>
    </div>
    <p class="small muted" style="margin-top:14px;max-width:80ch">Вероятность 25 % допускает отсутствие события. Качество вероятностей проверяется по совокупности прогнозов. <a href="${href("check")}">Открыть полную проверку</a></p>
    <div class="cases"><span class="small faint">Разобрать исторический случай:</span>
      ${cur.cases.map((c) => `<a class="chip" href="${href("", { oos: c.contour, y: c.release, id: c.iso3 })}" aria-pressed="${sel && c.contour === contour && sel.y === c.release && sel.id === c.iso3}">${esc(c.title)}</a>`).join("")}
    </div>
  </section>

  <section class="section" id="now">
    <div class="section-head">
      <div><h2>Что модель оценивает сейчас</h2><p>В каждом разделе — своё событие, горизонт и покрытие. Единого индекса кризиса здесь нет.</p></div>
      <div class="ctx" style="text-align:right">Последний публичный выпуск: <b>${lastV ? dateRu(lastV) : "нет"}</b><br>Текущий расчёт: ${dateRu(cur.release.calculated)}, ещё не заморожен</div>
    </div>
    ${nowRow("war", "Начало и продолжение оцениваются отдельно. Два порога интенсивности.",
      `Оценены ${cov.war.estimated} стран · неприменимо ${cov.war.not_applicable} · нет оценки ${cov.war.no_estimate}`,
      `начало — ${cov.war.at_risk} стран без войны, продолжение — ${cov.war.at_war} воюющих`,
      [["Смотреть начало конфликта", { domain: "war", event: "war25_onset" }], ["Продолжение", { domain: "war", event: "war25_cont" }]])}
    ${nowRow("fx", "Структурный прогноз, риск резкого месячного обвала и текущее давление.",
      `Оценены ${cov.fx.estimated} стран · неприменимо ${cov.fx.not_applicable} · нет оценки ${cov.fx.no_estimate}`,
      `структурный прогноз; месячный обвал и давление — ${cov.fx.monthly} стран`,
      [["Смотреть валютные оценки", { domain: "fx", event: "fxa" }]])}
    ${nowRow("bank", "Прогноз на основе кредитного цикла и кривой доходности.",
      `Оценены ${cov.bank.estimated} стран · неприменимо ${cov.bank.not_applicable} · нет оценки ${cov.bank.no_estimate}`,
      Object.entries(cov.bank.by_status).filter(([k]) => k !== "ok").map(([k, n]) => `${low(stateText(k))} — ${n}`).join(", "),
      [["Смотреть банковские оценки", { domain: "bank", event: "bank" }]])}
  </section>

  <section class="section" id="lens">
    <div class="section-head">
      <div><h2>15 постсоветских стран на фоне мира</h2><p>Сравните страны региона с мировой выборкой. Шкала остаётся общей для всех стран в пределах одного события.</p></div>
      <a class="btn ghost" href="${href("forecasts", { group: "ps" })}">Открыть все 15 стран</a>
    </div>
    <div class="cases" style="margin:0 0 4px"><span class="small faint">Подписать страну:</span>
      ${S.countries.filter((c) => c.post_soviet).map((c) => `<button class="chip" data-lens="${c.iso3}" aria-pressed="${c.iso3 === lens}">${esc(c.ru)}</button>`).join("")}
    </div>
    ${lensStrip(rel, "war25_onset", "war", lens)}
    ${lensStrip(rel, "fxa", "fx", lens)}
    ${lensStrip(rel, "bank", "bank", lens)}
    <div style="margin-top:18px">${stripLegend("war")}</div>
  </section>

  <section class="section" id="how">
    <div class="section-head"><h2>Как устроена проверка</h2></div>
    <div class="method3">
      <p><b>Обучение на прошлом.</b>Для каждого исторического среза обучение ограничено информацией, допустимой по протоколу на эту дату, включая лаги публикации источников.</p>
      <p><b>Проверка вероятностей.</b>Мы рассматриваем все прогнозы, калибровку и сравнение с наивной базой.</p>
      <p><b>Отдельный публичный архив.</b>Опубликованные прогнозы сохраняются как самостоятельные выпуски; исходы добавляются с указанием источника и версии.</p>
    </div>
    <div class="links" style="margin-top:22px"><a href="${href("method")}">Прочитать метод</a><a href="${href("method", { s: "limits" })}">Посмотреть ограничения</a><a href="${href("method", { s: "data" })}">Открыть данные</a></div>
  </section>`;

  el.querySelector(".seg").onclick = (e) => { const b = e.target.closest("button"); if (b) setQuery({ oos: b.dataset.oos, y: null, id: null }); };
  el.querySelector(".ribbon").onclick = (e) => { const t = e.target.closest(".rb-hit"); if (t) setQuery({ oos: contour, y: t.dataset.y, id: t.dataset.id }); };
  el.querySelector("#lens").onclick = (e) => {
    const b = e.target.closest("[data-lens]"), d = e.target.closest("circle[data-id]");
    if (b) setQuery({ lens: b.dataset.lens === lens ? null : b.dataset.lens });
    else if (d && isPS(d.dataset.id)) setQuery({ lens: d.dataset.id });
  };
}

function side(rib, contour, sel) {
  if (!sel) return `<div class="card"><p class="muted">Хорошее ранжирование собирает больше последующих событий в левой части графика. Высокие оценки без события и события при низких оценках остаются видимыми.</p>
    <p class="small faint" style="margin-top:12px">Нажмите на отметку страны, чтобы увидеть её оценку в этом выпуске.</p></div>`;
  const yr = rib.years.find((y) => y.release === sel.y), r = yr?.rows.find((x) => x[0] === sel.id);
  if (!r) return `<div class="card"><p class="muted">В этом выпуске страна не входила в выборку модели.</p></div>`;
  const [id, p, lo, hi, rank, out] = r;
  return `<div class="card">
    <div class="eyebrow">Исторический выпуск · проверка вслепую</div>
    <h3 style="margin-top:6px">${esc(nameOf(id))} · выпуск ${yr.as_of.slice(0, 4)}</h3>
    <p class="small muted">срез ${dateRu(yr.as_of)}</p>
    <div class="big num" style="margin-top:10px">${pct(p)}</div>
    <dl class="kv">
      <dt>Прогноз</dt><dd class="num">${pct(p)}</dd>
      <dt>Интервал</dt><dd>${lo != null ? `<span class="num">${pctRange(lo, hi)}</span>` : state("no_interval")}</dd>
      <dt>Место</dt><dd class="num">${rank} из ${yr.n}</dd>
      <dt>Целевой период</dt><dd class="num">${esc(yr.target)}</dd>
      <dt>Исход</dt><dd>${out ? "<b>событие наступило</b>" : "событие не наступило"}</dd>
    </dl>
    <a class="btn ghost sm" href="${href(`record/oos/${contour}/${yr.release}/${id}`)}">Разобрать этот прогноз</a>
    <p class="tiny faint" style="margin-top:12px">Исторические оценки рассчитаны по протоколу проверки на будущих для обучения наблюдениях. Они не были публичными прогнозами того времени.</p>
  </div>`;
}

function nowRow(dom, text, cov, covNote, links) {
  return `<div class="now-row dom-${dom}">
    <h3><span class="dot"></span>${DOMAIN_FULL[dom]}</h3>
    <div><p>${text}</p><p class="cov num">${cov}<br><span class="faint">${esc(covNote)}</span></p></div>
    <div class="links">${links.map(([t, q], i) => `<a class="${i ? "" : "btn ghost sm"}" href="${href("forecasts", q)}">${t}</a>`).join("")}</div>
  </div>`;
}

function lensStrip(rel, ev, dom, lens) {
  const c = rel.contexts[ev];
  const pts = [], ps = { ok: [], na: [], no: [], grp: [] };
  for (const [iso, r] of rel.rows) {
    const e = r.est[ev];
    if (e?.p !== undefined) pts.push({ id: iso, v: e.p, ps: r.post_soviet });
    if (r.post_soviet) (e?.p !== undefined ? ps.ok : e?.status?.startsWith("shared:") ? ps.na : e?.status === "other_group" ? ps.grp : ps.no).push([iso, e?.status]);
  }
  const max = niceMax(Math.max(...pts.map((p) => p.v)));
  const why = [...ps.na, ...ps.no].map(([iso, st]) => `${nameOf(iso)} — ${low(stateText(st))}`).join("; ");
  const lensState = lens && !pts.some((p) => p.id === lens) ? rel.rows.get(lens).est[ev]?.status : null;
  return `<div class="strip dom-${dom}">
    <h3><span class="dot"></span>${esc(c.title)}</h3>
    <div class="ctx">событие: <b>${esc(c.event)}</b> · срез ${dateRu(c.as_of)} · целевой период <b>${esc(c.target_period)}</b> · в мировой выборке ${pts.length} стран</div>
    ${strip(pts, { max, sel: lens })}
    ${lensState ? `<p class="small" style="margin-top:4px">${esc(nameOf(lens))}: ${state(lensState, { link: href("forecasts", { domain: "war", event: "war25_cont" }) })}</p>` : ""}
    <p class="small muted num" style="margin-top:6px">С оценкой: ${ps.ok.length} из 15 · неприменимо: ${ps.na.length} · нет оценки: ${ps.no.length}${why ? ` <span class="faint">(${esc(why)})</span>` : ""}${ps.grp.length ? ` · идёт конфликт, оценивается продолжение: ${ps.grp.length} <span class="faint">(${esc(ps.grp.map(([i]) => nameOf(i)).join(", "))})</span>` : ""}</p>
  </div>`;
}
