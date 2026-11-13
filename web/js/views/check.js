// Экран «Проверка»: все проверяемые оценки, калибровка, сравнение с простым правилом, верхние k.
// Фирменное взаимодействие — просмотр выпуска без знания исходов: слой исходов снимается целиком,
// остаётся только то, что было допустимо на дату среза.
import { S, load, esc, num, pct, pctRange, dateRu, href, nameOf, state, ribbon, setQuery, niceMax, ARCHIVE_EVENT, archiveKey, archiveWindow } from "../lib.js";

const ORDER = ["war25", "fxa", "fxm", "fxm_live", "bank"];
const SHORT = { war25: "Начало войны", fxa: "Обвал курса за год", fxm: "Скачок за месяц", fxm_live: "Скачок за месяц, новая модель", bank: "Банки" };
const RIBBON = { war25: "war25", fxa: "fxa", fxm: "fxm", bank: "bank" };

export async function check(el, _parts, q) {
  const [ck, sc] = await Promise.all([load("check.json"), load("scorecard.json")]);
  const mode = q.mode === "pub" ? "pub" : "retro";
  const key = ck.contours[q.event] ? q.event : "war25";
  const c = ck.contours[key];
  const sample = c.samples[q.sample] ? q.sample : "world";
  const period = q.period === "2010" && c.cells[`${sample}|2010`] ? "2010" : "all";
  const cell = c.cells[`${sample}|${period}`];
  const blind = q.blind === "1";
  const rib = RIBBON[key] ? await load(`oos/${RIBBON[key]}.json`) : null;
  if (rib) Object.assign(S.extraNames, rib.extra_names ?? {});
  const sel = q.y && q.id ? { y: isNaN(+q.y) ? q.y : +q.y, id: q.id } : null;

  const head = `
  <section class="section dom-${c.domain}" style="padding-top:30px">
    <div class="section-head"><div><h1>Проверка прогнозов</h1>
      <p>Все проверяемые оценки, включая высокие вероятности без события и события при низких вероятностях.</p></div>
      <div class="seg" id="mode" role="group" aria-label="Вид проверки"><button data-v="retro" aria-pressed="${mode === "retro"}">Ретроспектива</button><button data-v="pub" aria-pressed="${mode === "pub"}">Публичные выпуски</button></div></div>`;
  if (mode === "pub") { el.innerHTML = head + publicBlock() + "</section>"; bind(el, {}); return; }

  el.innerHTML = head + `
    <div class="controls">
      <div class="ctl"><span>Событие</span><div class="seg" role="group" aria-label="Событие" id="ev">${ORDER.map((k) => `<button data-v="${k}" aria-pressed="${k === key}">${SHORT[k]}</button>`).join("")}</div></div>
    </div>
    <div class="controls" style="margin-top:12px">
      <div class="ctl"><span>Выборка</span><div class="seg" role="group" aria-label="Выборка" id="smp">${Object.entries(c.samples).map(([k, t]) => `<button data-v="${k}" aria-pressed="${k === sample}" ${c.cells[`${k}|all`] ? "" : "disabled"}>${esc(t.split(":")[0])}</button>`).join("")}</div></div>
      <div class="ctl"><span>Период</span><div class="seg" role="group" aria-label="Период" id="per"><button data-v="all" aria-pressed="${period === "all"}">Все годы</button><button data-v="2010" aria-pressed="${period === "2010"}" ${c.cells[`${sample}|2010`] ? "" : "disabled"}>С 2010 года</button></div></div>
      <div class="ctl"><span>Горизонт</span><span class="small">${esc(c.horizon)}</span></div>
    </div>
    <div class="plate ${c.domain}" style="margin-top:16px"><span class="tag dark">Проверка на прошлом, вслепую</span>&nbsp; <b>${esc(c.title)}.</b>
      ${esc(c.samples[sample])} · ${esc(cell.years)} · ${cell.n.toLocaleString("ru-RU")} оценок${blind ? "" : `, из них с наступившим событием — ${cell.events.toLocaleString("ru-RU")} (${pct(cell.base_rate)})`}. Модель: ${esc(c.model)}.
      ${key === "fxm_live" ? " Новая модель на испытании: проверяется рядом с основной и её не заменяет." : ""}</div>
  </section>

  <section class="section dom-${c.domain}" style="padding-top:28px">
    <div class="section-head"><div><h2>Лента прогнозов</h2>
      <p>${blind ? "Выпуски показаны так, как они выглядели на дату прогноза: оценки и места есть, исходов нет." : "Каждая строка — исторический выпуск, отметка — страна; слева более высокие оценки, тёмные точки — наступившие события."}</p></div>
      <button class="btn ${blind ? "" : "ghost"}" id="blind">${blind ? "Показать исходы" : "Посмотреть выпуск без знания исходов"}</button></div>
    ${rib ? `<div class="grid"><div class="c8 ribbon">${ribbon(rib, sel, { blind })}
        <p class="small faint" style="margin-top:6px">Лента показывает мировую выборку события${c.monthly ? "; для месячной модели — январские прогнозы каждого года" : ""}. Положение в рейтинге не равно вероятности события.
        ${blind ? "" : `Исходы: ${esc(rib.source)}, текущая версия наблюдений.`}</p></div>
      <aside class="c4">${card(rib, RIBBON[key], sel, blind)}</aside></div>`
    : `<p class="plate">Для новой модели на испытании отдельная лента не выгружается: её прогнозы проверяются на тех же строках, что у основной месячной модели.</p>`}
  </section>

  ${blind ? `<section class="section"><div class="plate"><b>Слой исходов скрыт.</b> Калибровка, сравнение с базой и доля пойманных событий считаются только по исходам, поэтому сейчас не показаны.
      На экране осталось то, что было допустимо на дату прогноза по протоколу: оценки, места и состав выборки. <a href="#" id="unblind">Показать исходы</a></div></section>` : `
  <section class="section dom-${c.domain}">
    <div class="grid">
      <div class="c6"><h2>Соответствуют ли вероятности наблюдаемым частотам?</h2>
        <p class="fact">${relLead(cell.reliability)}</p>
        <p class="muted small" style="margin:8px 0 10px">Оценки разбиты на ${cell.reliability.length} групп равного размера. Диагональ — совпадение вероятности и частоты. Отрезки — 95 % интервал частоты.</p>
        <div class="chart">${reliability(cell.reliability)}</div>
        <details style="margin-top:8px"><summary class="small">Группы таблицей</summary>
          <table class="factors" style="margin-top:6px"><thead><tr><th scope="col">Оценки в группе</th><th scope="col">Средняя оценка</th><th scope="col">Наблюдаемая частота</th><th scope="col">Событий / оценок</th></tr></thead><tbody>
          ${cell.reliability.map((b) => `<tr><td class="num">${pctRange(b.p_min, b.p_max)}</td><td class="num">${pct(b.p)}</td><td class="num">${pct(b.freq)} <span class="dt">[${pctRange(b.lo, b.hi)}]</span></td><td class="num">${b.events} / ${b.n}</td></tr>`).join("")}</tbody></table></details>
      </div>
      <div class="c6"><h2>Лучше ли модель простого правила?</h2>
        <p class="fact">${brierLead(cell.brier)}</p>
        <details class="more"><summary>Показать расчёт: разность Brier score</summary>
        <p class="muted small" style="margin:8px 0 10px">Brier score — средний квадрат ошибки вероятности. Показана разность базы и модели. Выше нуля — меньшая ошибка у модели. Отрезок — 95 % интервал (бутстрэп по годам).</p>
        <div class="chart">${brier(cell.brier)}</div></details>
        <h2 style="margin-top:34px">Сколько событий попадает в поле внимания?</h2>
        <p class="fact">${topLead(cell.top_k)}</p>
        <p class="muted small" style="margin:8px 0 10px">Доля наступивших событий среди первых стран списка каждого выпуска (${cell.releases} выпусков). Вертикальная отметка — что дал бы случайный порядок.</p>
        <div class="chart">${topk(cell.top_k)}</div>
        <p class="tiny faint" style="margin-top:6px">Исследовательский порог. Он не меняет исторически опубликованные сигналы.${c.monthly ? " Для месячной модели выпуск — каждый месяц." : ""}</p>
      </div>
    </div>
    <details style="margin-top:28px"><summary>Дополнительные метрики</summary>
      <dl class="kv" style="max-width:720px">
        <dt>AUC сквозной</dt><dd class="num">${num(cell.auc.pooled, 3)}${cell.auc.lo != null ? ` [${num(cell.auc.lo, 3)}; ${num(cell.auc.hi, 3)}] — 95 % блочный бутстрэп по годам` : " — интервал для этой выборки в сводке качества не считался"}</dd>
        <dt>AUC внутри года</dt><dd class="num">${num(cell.auc.within_year, 3)} — среднее по годам, где определён</dd>
        <dt>Выборка</dt><dd>${esc(c.samples[sample])}; ${esc(cell.years)}; ${cell.n.toLocaleString("ru-RU")} оценок, ${cell.events} событий</dd>
      </dl>
      <p class="small muted" style="max-width:90ch">AUC — доля пар «событие / не событие», которые модель упорядочила правильно: 0,5 — монетка, 1 — идеально. ${esc(ck.method)}</p>
      <div class="tbl-scroll" style="margin-top:12px"><table class="tbl"><thead><tr><th scope="col">Событие</th><th scope="col">Выборка</th><th scope="col">Модель</th><th scope="col">Оценок</th><th scope="col">Событий</th><th scope="col">AUC сквозной [95 %]</th><th scope="col">Внутри года</th><th scope="col">Brier</th></tr></thead><tbody>
        ${sc.rows.map((r) => `<tr><td class="small">${esc(r.block)}</td><td class="small">${esc(r.sample)}</td><td class="mono">${esc(r.model)}</td><td class="num">${r.n.toLocaleString("ru-RU")}</td><td class="num">${r.positives}</td>
          <td class="num">${num(r.auc, 3)} [${num(r.auc_lo, 3)}; ${num(r.auc_hi, 3)}]</td><td class="num">${num(r.auc_within_year, 3)}</td><td class="num">${num(r.brier, 3)}</td></tr>`).join("")}</tbody></table></div>
      <p class="tiny faint" style="margin-top:6px">Полная сводка качества: <span class="mono">${esc(sc.file)}</span>.</p>
    </details>
  </section>`}`;
  bind(el, { key, rib: RIBBON[key] });
}

function bind(el, { key }) {
  const seg = (id, fn) => el.querySelector(id)?.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) fn(b.dataset.v); });
  seg("#mode", (v) => setQuery({ mode: v === "pub" ? "pub" : null }));
  seg("#ev", (v) => setQuery({ event: v, sample: null, period: null, y: null, id: null }));
  seg("#smp", (v) => setQuery({ sample: v === "world" ? null : v, period: null }));
  seg("#per", (v) => setQuery({ period: v === "2010" ? "2010" : null }));
  const flip = (e) => { e.preventDefault(); setQuery({ blind: new URLSearchParams(location.hash.split("?")[1] ?? "").get("blind") === "1" ? null : "1" }); };
  el.querySelector("#blind")?.addEventListener("click", flip);
  el.querySelector("#unblind")?.addEventListener("click", flip);
  el.querySelector(".ribbon")?.addEventListener("click", (e) => { const t = e.target.closest(".rb-hit"); if (t) setQuery({ event: key, y: t.dataset.y, id: t.dataset.id }); });
}

function card(rib, contour, sel, blind) {
  if (!sel) return `<div class="card"><p class="muted">Нажмите на отметку страны, чтобы увидеть её оценку в этом выпуске.</p>
    <p class="small faint" style="margin-top:10px">Высокие оценки без события и события при низких оценках остаются на ленте — их не убирают.</p></div>`;
  const yr = rib.years.find((y) => String(y.release) === String(sel.y)), r = yr?.rows.find((x) => x[0] === sel.id);
  if (!r) return `<div class="card"><p class="muted">В этом выпуске страна не входила в выборку модели.</p></div>`;
  const [id, p, lo, hi, rank, out] = r;
  return `<div class="card"><div class="eyebrow">Исторический выпуск · проверка вслепую</div>
    <h3 style="margin-top:6px">${esc(nameOf(id))} · выпуск ${esc(String(yr.as_of).slice(0, 4))}</h3><p class="small muted">дата прогноза: ${dateRu(yr.as_of)}</p>
    <div class="big num" style="margin-top:10px">${pct(p)}</div>
    <dl class="kv"><dt>Интервал</dt><dd>${lo != null ? `<span class="num">${pctRange(lo, hi)}</span>` : state("no_interval")}</dd>
      <dt>Место</dt><dd class="num">${rank} из ${yr.n}</dd><dt>Период прогноза</dt><dd class="num">${esc(yr.target)}</dd>
      <dt>Исход</dt><dd>${blind ? '<span class="st st-text">скрыт: выпуск показан без знания исходов</span>' : out ? "<b>событие наступило</b>" : "событие не наступило"}</dd></dl>
    ${rib.details === false ? "" : `<a class="btn ghost sm" href="${href(`record/oos/${contour}/${yr.release}/${id}`)}">Разобрать этот прогноз</a>`}</div>`;
}

// Строки-выводы над графиками: одно предложение простыми словами из тех же чисел, что на графике.
function relLead(bins) {
  const hi = bins.at(-1), lo = bins[0];
  return `Когда модель давала самые высокие оценки (в среднем ${pct(hi.p)}), событие наступало в ${pct(hi.freq)} случаев: ${hi.events} из ${hi.n}. При самых низких (в среднем ${pct(lo.p)}) — в ${pct(lo.freq)}: ${lo.events} из ${lo.n}.`;
}
function brierLead(list) {
  if (!list.length) return "Простое правило для сравнения по этой выборке не сохранено.";
  const b = list[0], rule = b.base === "base_rate" ? "«всем странам — средняя частота прошлых лет»" : `«${esc(b.label)}»`;
  if (b.lo > 0) return `Модель ошибается меньше, чем простое правило ${rule}. Разница больше погрешности.`;
  if (b.hi < 0) return `Модель ошибается больше, чем простое правило ${rule}. Разница больше погрешности.`;
  return `Отличие модели от простого правила ${rule} не установлено: разница в пределах погрешности.`;
}
function topLead(list) {
  const t = list.find((x) => x.k === 10) ?? list[0];
  return `Из ${t.events} наступивших событий ${t.caught} пришлись на ${t.k} стран с самыми высокими оценками своего выпуска. При случайном порядке туда попало бы около ${Math.max(1, Math.round(t.random * t.events))}.`;
}

function reliability(bins) {
  const W = 520, H = 330, L = 50, B = 40, T = 12, R = 14;
  const max = niceMax(Math.max(...bins.map((b) => Math.max(b.p, b.hi))));
  // обе оси — на одной шкале квадратного корня: группы редких событий не слипаются у нуля, диагональ остаётся диагональю
  const u = (v) => Math.sqrt(Math.min(Math.max(v, 0), max) / max);
  const x = (v) => L + u(v) * (W - L - R), y = (v) => H - B - u(v) * (H - B - T);
  const ticks = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1].filter((t) => t <= max + 1e-9 && (t === 0 || t >= max / 60));
  const f = (v) => num(100 * v, v > 0 && v < 0.01 ? 1 : 0);
  return `<svg viewBox="0 0 ${W} ${H}" style="max-width:${W}px;width:100%" role="img" aria-label="Калибровка: средняя оценка группы и наблюдаемая частота событий">
    ${ticks.map((t) => `<line class="hp-grid" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/><text class="hp-axis" x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${f(t)}</text>
      <text class="hp-axis" x="${x(t)}" y="${H - B + 16}" text-anchor="middle">${f(t)}</text>`).join("")}
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}" stroke="var(--ink-3)" stroke-dasharray="4 4"/>
    ${bins.map((b) => `<line x1="${x(b.p)}" x2="${x(b.p)}" y1="${y(b.lo)}" y2="${y(b.hi)}" stroke="var(--dom)" stroke-width="2" opacity=".45"/>
      <circle cx="${x(b.p)}" cy="${y(b.freq)}" r="4.5" fill="var(--dom)"><title>оценка ${pct(b.p)}, частота ${pct(b.freq)}, ${b.events} из ${b.n}</title></circle>`).join("")}
    <text class="hp-axis" x="${(W + L) / 2}" y="${H - 6}" text-anchor="middle">Средняя оценка в группе, % · обе оси растянуты у нуля</text>
    <text class="hp-axis" transform="translate(12 ${(H - B) / 2}) rotate(-90)" text-anchor="middle">Наблюдаемая частота, %</text>
    <text class="hp-axis" x="${W - R}" y="${T + 10}" text-anchor="end">в каждой группе ≈ ${bins[0].n} оценок</text>
  </svg>`;
}

function brier(list) {
  if (!list.length) return `<p class="muted">Для этой выборки базовая оценка не сохранена.</p>`;
  const W = 520, L = 14, R = 14, rowH = 46;
  const m = Math.max(...list.flatMap((b) => [Math.abs(b.lo), Math.abs(b.hi), Math.abs(b.diff)]), 1e-6) * 1.15;
  const x = (v) => L + ((v + m) / (2 * m)) * (W - L - R);
  return `<svg viewBox="0 0 ${W} ${list.length * rowH + 30}" style="max-width:${W}px;width:100%" role="img" aria-label="Разность Brier базы и модели">
    <line x1="${x(0)}" x2="${x(0)}" y1="0" y2="${list.length * rowH + 6}" stroke="var(--ink)"/>
    ${list.map((b, i) => { const cy = i * rowH + 30;
      return `<text class="hp-title" x="${L}" y="${cy - 14}">${esc(b.label)}</text>
      <line x1="${x(b.lo)}" x2="${x(b.hi)}" y1="${cy}" y2="${cy}" stroke="var(--dom)" stroke-width="2" opacity=".45"/><circle cx="${x(b.diff)}" cy="${cy}" r="5" fill="var(--dom)"/>
      <text class="hp-axis" x="${x(b.diff)}" y="${cy + 17}" text-anchor="middle">${b.diff > 0 ? "+" : ""}${num(b.diff, 4)} [${num(b.lo, 4)}; ${num(b.hi, 4)}]</text>`; }).join("")}
    <text class="hp-axis" x="${L}" y="${list.length * rowH + 26}">← ошибка меньше у простого правила</text><text class="hp-axis" x="${W - R}" y="${list.length * rowH + 26}" text-anchor="end">ошибка меньше у модели →</text>
  </svg>
  <p class="tiny faint">Brier модели ${num(list[0].brier_model, 4)}; у простого правила — ${list.map((b) => num(b.brier_base, 4)).join(" и ")}. Интервал, накрывающий ноль, означает, что отличие от простого правила не установлено.</p>`;
}

function topk(list) {
  const W = 520, L = 92, R = 70, rowH = 30;
  const x = (v) => L + v * (W - L - R);
  return `<svg viewBox="0 0 ${W} ${list.length * rowH + 22}" style="max-width:${W}px;width:100%" role="img" aria-label="Доля событий среди первых стран списка выпуска">
    ${list.map((t, i) => { const cy = i * rowH + 14;
      return `<text class="hp-title" x="0" y="${cy + 4}">первые ${t.k}</text><rect x="${L}" y="${cy - 7}" width="${W - L - R}" height="14" fill="var(--paper-3)"/>
        <rect x="${L}" y="${cy - 7}" width="${x(t.share) - L}" height="14" fill="var(--dom)"/><line x1="${x(t.random)}" x2="${x(t.random)}" y1="${cy - 11}" y2="${cy + 11}" stroke="var(--ink)" stroke-width="1.5"/>
        <text class="hp-axis" x="${W - R + 8}" y="${cy + 4}">${pct(t.share)}</text>`; }).join("")}
    <text class="hp-axis" x="${L}" y="${list.length * rowH + 18}">${list.map((t) => `первые ${t.k}: ${t.caught} из ${t.events}, случайно — ${pct(t.random)}`).join(" · ")}</text>
  </svg>`;
}

function publicBlock() {
  const vs = [...S.archive.vintages].reverse();
  return `<div class="plate" style="margin-bottom:18px"><b>Проспективной проверки пока нет.</b> Публичные выпуски начали копиться с ${dateRu(S.archive.vintages[0].vintage)}; периоды прогноза почти всех оценок ещё открыты.
      Исходы будут добавляться сюда по мере закрытия периодов — с источником и версией наблюдений. До тех пор здесь стоят статусы, а не нули.</div>
    <div class="tbl-scroll"><table class="tbl"><thead><tr><th scope="col">Выпуск</th><th scope="col">Событие</th><th scope="col">Период прогноза</th><th scope="col">Оценок</th><th scope="col">Состояние проверки</th></tr></thead><tbody>
    ${vs.flatMap((v) => v.models.map((m) => `<tr><td class="num"><a href="${href("archive/" + v.vintage)}">${dateRu(v.vintage)}</a></td><td>${esc(ARCHIVE_EVENT[archiveKey(m.column)][0])}</td>
      <td class="num small">${esc(archiveWindow(v, m))}</td><td class="num">${m.n_countries}</td>
      <td class="small">${m.check_status.map((s) => (s === "resolved" ? "Исход известен" : state(s))).join(" ")}${m.retrospective_note ? `<span class="st-detail">${esc(m.retrospective_note)}</span>` : ""}</td></tr>`)).join("")}
    </tbody></table></div>
    <div class="links" style="margin-top:18px"><a href="${href("archive")}">Открыть архив выпусков</a><a href="${href("check")}">Ретроспективная проверка</a></div>`;
}
