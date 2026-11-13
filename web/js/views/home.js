// Главная: H1 введение, H2 лента проверки прошлым, H3 текущие оценки, H4 региональная линза, H5 метод.
import { S, load, release, esc, pct, pctRange, dateRu, href, nameOf, isPS, state, stateText, strip, stripLegend, ribbon, odds,
  niceMax, DOMAIN_FULL, setQuery, low } from "../lib.js";

const RIBBONS = { war25: "Начало войны", fxa: "Обвал курса за год" };
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

  <section class="section" id="now" style="padding-top:30px">
    <div class="section-head">
      <div><h2>Что модель оценивает сейчас</h2><p>По три страны с самой высокой вероятностью в каждом событии. У каждого события своя модель и своя шкала. Единого индекса кризиса здесь нет.</p></div>
      <div class="ctx" style="text-align:right">Последний публичный выпуск: <b>${lastV ? dateRu(lastV) : "нет"}</b><br>Текущий расчёт: ${dateRu(cur.release.calculated)}, ещё не заморожен</div>
    </div>
    <div class="now4">
      ${nowCard(rel, "war25_onset", "war", "Начало войны", `<a href="${href("forecasts", { domain: "war", event: "war25_cont" })}">Продолжение идущих войн (${cov.war.at_war})</a>`)}
      ${nowCard(rel, "fxa", "fx", "Обвал курса за год")}
      ${nowCard(rel, "fxm", "fx", "Резкий скачок курса за месяц")}
      ${nowCard(rel, "bank", "bank", "Банковский кризис")}
    </div>
    <p class="small muted" style="margin-top:14px;max-width:86ch">Высокая вероятность — не предсказание: даже при 30 % событие чаще не случается, чем случается. Страны без числа не оценены, и это не значит «спокойно»: всего в витрине ${S.countries.length} государств.</p>
  </section>

  <section class="section dom-${DOM[contour]}" id="past">
    <div class="section-head">
      <div>
        <h2>Можно ли этому верить?</h2>
        <p>Так модель расставляла страны в прошлые годы. Каждая строка — год, каждая отметка — страна. Чёрная точка — кризис случился; чем она левее, тем выше была вероятность.</p>
      </div>
      <div class="seg" role="group" aria-label="Событие">${Object.entries(RIBBONS).map(([k, t]) => `<button data-oos="${k}" aria-pressed="${k === contour}">${t}</button>`).join("")}</div>
    </div>
    <div class="grid">
      <div class="c8 ribbon">
        <div class="ctx" style="margin-bottom:8px"><span class="tag dark">Проверка на прошлом, вслепую</span>
          &nbsp; Событие: <b>${esc(rib.event)}</b>. Прогноз на ${esc(rib.horizon)} вперёд, выпуски ${rib.years[0].as_of.slice(0, 4)}–${rib.years.at(-1).as_of.slice(0, 4)}.</div>
        ${ribbon(rib, sel)}
        <p class="small faint" style="margin-top:6px">Положение в рейтинге не равно вероятности события. Год слева — год выпуска: оценка сделана на 1 января этого года по данным, доступным к этой дате.</p>
      </div>
      <aside class="c4" id="rb-side">${side(rib, contour, sel)}</aside>
    </div>
    <p class="small muted" style="margin-top:14px;max-width:80ch">Вероятность 25 % допускает отсутствие события. Качество вероятностей проверяется по совокупности прогнозов. <a href="${href("check")}">Открыть полную проверку</a></p>
    <div class="cases"><span class="small faint">Разобрать исторический случай:</span>
      ${cur.cases.map((c) => `<a class="chip" href="${href("", { oos: c.contour, y: c.release, id: c.iso3 })}" ${sel && c.contour === contour && sel.y === c.release && sel.id === c.iso3 ? 'aria-current="true"' : ""}>${esc(c.title)}</a>`).join("")}
    </div>
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
      <p><b>Обучение на прошлом.</b>Для каждой исторической даты прогноза обучение ограничено информацией, допустимой по протоколу на эту дату, включая лаги публикации источников.</p>
      <p><b>Проверка вероятностей.</b>Мы рассматриваем все прогнозы, калибровку и сравнение с простым правилом.</p>
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
    <p class="small muted">дата прогноза: ${dateRu(yr.as_of)}</p>
    <div class="big num" style="margin-top:10px">${pct(p)}</div>
    <dl class="kv">
      <dt>Прогноз</dt><dd class="num">${pct(p)}</dd>
      <dt>Интервал</dt><dd>${lo != null ? `<span class="num">${pctRange(lo, hi)}</span>` : state("no_interval")}</dd>
      <dt>Место</dt><dd class="num">${rank} из ${yr.n}</dd>
      <dt>Период прогноза</dt><dd class="num">${esc(yr.target)}</dd>
      <dt>Исход</dt><dd>${out ? "<b>событие наступило</b>" : "событие не наступило"}</dd>
    </dl>
    <a class="btn ghost sm" href="${href(`record/oos/${contour}/${yr.release}/${id}`)}">Разобрать этот прогноз</a>
    <p class="tiny faint" style="margin-top:12px">Исторические оценки рассчитаны по протоколу проверки на будущих для обучения наблюдениях. Они не были публичными прогнозами того времени.</p>
  </div>`;
}

/** Карточка «сейчас»: три страны с самой высокой вероятностью, обычный уровень и первая из постсоветских. */
function nowCard(rel, ev, dom, title, extra = "") {
  const c = rel.contexts[ev];
  if (!c) return "";
  const rows = [...rel.rows].filter(([, r]) => r.est[ev]?.p !== undefined).map(([iso, r]) => ({ iso, e: r.est[ev], ps: r.post_soviet })).sort((a, b) => a.e.rank - b.e.rank || b.e.p - a.e.p);
  const line = (x, rank = false) => `<li><a href="${href("country/" + x.iso)}">${esc(nameOf(x.iso))}</a><span class="num">${pct(x.e.p)}${rank ? ` <span class="faint">· ${x.e.rank}-е место</span>` : ""}</span></li>`;
  const base = rows.find((x) => x.e.base_rate != null)?.e.base_rate;
  const ps = rows.find((x) => x.ps);
  const period = /^\d/.test(c.target_period ?? "") && !/месяц/.test(c.target_period) ? c.target_period : `на ${c.horizon}`;
  return `<div class="now-card dom-${dom}">
    <div class="eyebrow">${esc(DOMAIN_FULL[dom])}</div>
    <h3>${esc(title)} <span class="period num">· ${esc(period)}</span></h3>
    <ol class="top">${rows.slice(0, 3).map((x) => line(x)).join("")}</ol>
    ${base != null ? `<p class="small muted">Обычный уровень — ${odds(base)}</p>` : ""}
    <div class="ps"><div class="small muted">Из 15 постсоветских выше всех:</div>
      ${ps ? `<ol class="top">${line(ps, true)}</ol>` : `<p class="small">ни одна не оценена: для них нет данных, на которых построена модель</p>`}</div>
    <div class="links"><a class="btn ghost sm" href="${href("forecasts", { domain: dom, event: ev })}">Весь список (${rows.length})</a>${extra}</div>
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
    <div class="ctx">Событие: <b>${esc(c.event)}</b>. Период прогноза: <b>${esc(c.target_period)}</b>, дата прогноза: ${dateRu(c.as_of)}. В мировой выборке ${pts.length} стран.</div>
    ${strip(pts, { max, sel: lens })}
    ${lensState ? `<p class="small" style="margin-top:4px">${esc(nameOf(lens))}: ${state(lensState, { link: href("forecasts", { domain: "war", event: "war25_cont" }) })}</p>` : ""}
    <p class="small muted num" style="margin-top:6px">С оценкой ${ps.ok.length} из 15 стран.${why ? ` Без оценки ${ps.na.length + ps.no.length}: <span class="faint">${esc(why)}.</span>` : ""}${ps.grp.length ? ` Идёт конфликт, оценивается продолжение: <span class="faint">${esc(ps.grp.map(([i]) => nameOf(i)).join(", "))}.</span>` : ""}</p>
  </div>`;
}
