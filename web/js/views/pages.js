// «Метод и данные» и временная страница «Проверка» (полный экран — чекпоинт 5.3).
import { S, load, esc, num, href, dateRu, statesLegend } from "../lib.js";

export async function check(el) {
  const sc = await load("scorecard.json");
  const f = (x) => (x === null ? "—" : num(x, 3));
  el.innerHTML = `
  <section class="section" style="padding-top:30px">
    <div class="section-head"><div><h1>Проверка прогнозов</h1>
      <p>Все проверяемые оценки, включая высокие вероятности без события и события при низких вероятностях.</p></div></div>
    <div class="plate"><b>Экран в работе.</b> Калибровка, сравнение с наивной базой, доля событий в верхних k странах и просмотр выпуска без знания исходов появятся в следующем чекпоинте. Пока здесь — сводная таблица качества из проверки вслепую, как она есть в <span class="mono">${esc(sc.file)}</span>.</div>
    <p class="small muted" style="margin:14px 0 6px;max-width:90ch">${esc(sc.method)} AUC — доля пар «событие / не событие», которые модель упорядочила правильно: 0,5 — монетка, 1 — идеально.</p>
    <div class="tbl-scroll"><table class="tbl">
      <thead><tr><th>Событие</th><th>Выборка</th><th>Модель</th><th>Прогнозов</th><th>Событий</th><th>Годы</th><th>AUC сквозной [95 %]</th><th>AUC внутри года</th><th>Brier</th><th>Δ AUC к базе [95 %]</th></tr></thead>
      <tbody>${sc.rows.map((r) => `<tr><td class="small">${esc(r.block)}</td><td class="small">${esc(r.sample)}</td><td class="mono">${esc(r.model)}</td>
        <td class="num">${r.n.toLocaleString("ru-RU")}</td><td class="num">${r.positives}</td><td class="num small">${esc(r.years)}</td>
        <td class="num">${f(r.auc)} [${f(r.auc_lo)}; ${f(r.auc_hi)}]</td><td class="num">${f(r.auc_within_year)}</td><td class="num">${f(r.brier)}</td>
        <td class="num small">${r.auc_vs_ref === null ? "—" : `${r.auc_vs_ref > 0 ? "+" : ""}${f(r.auc_vs_ref)} [${f(r.auc_vs_ref_lo)}; ${f(r.auc_vs_ref_hi)}] к ${esc(r.ref)}`}</td></tr>`).join("")}</tbody>
    </table></div>
    <p class="small faint" style="margin-top:12px">У баз <span class="mono">decay</span> и <span class="mono">persist</span> балл — не вероятность: их Brier с остальными не сравнивается. На постсоветской подвыборке месячная валютная модель не лучше монетки — это видно в таблице и не скрывается.</p>
    <div class="links" style="margin-top:18px"><a href="${href("")}">Лента проверки прошлым на главной</a><a href="${href("archive")}">Публичные выпуски</a></div>
  </section>`;
}

export async function method(el, _p, q) {
  const m = await load("meta.json");
  const cov = S.current.coverage;
  el.innerHTML = `
  <section class="section" style="padding-top:30px">
    <div class="section-head"><div><h1>Метод и данные</h1><p>Как считаются оценки, что известно модели на дату прогноза и чего она не умеет.</p></div></div>
    <div class="grid">
      <div class="c7">
        <h2 id="method">Протокол</h2>
        <ul class="plain">
          <li><b>Точка отсчёта.</b> Каждый признак и каждый параметр на дату прогноза считается только по данным, доступным к этой дате. ${esc(m.origin)}</li>
          <li><b>Метка известна не сразу.</b> Если событие относится к будущему окну, обучаться можно только на тех наблюдениях, чьё окно уже закрылось.</li>
          <li><b>Проверка вслепую.</b> Для каждого исторического года модель обучается на прошлом и оценивает период, которого не видела.</li>
          <li><b>Единого индекса кризиса нет.</b> У каждого события — своя модель, свой горизонт и своя выборка стран.</li>
        </ul>
        <h2 style="margin-top:32px">${esc(m.interval_method.title)}</h2>
        <p class="muted" style="margin-top:8px">${esc(m.interval_method.text)}</p>
        <p class="muted" style="margin-top:8px">${esc(m.interval_method.no_interval)}</p>
        <h2 style="margin-top:32px">Вклады признаков</h2>
        <p class="muted" style="margin-top:8px">${esc(m.factors_method)}</p>
        <h2 style="margin-top:32px" id="limits">Ограничения</h2>
        <ul class="plain">
          <li>События редкие: высокая оценка чаще всего не сбывается, и это нормальное свойство вероятностного прогноза.</li>
          <li>Банковский прогноз покрывает ${cov.bank.estimated} стран из 174; для остальных нет рядов, на которых построена модель.</li>
          <li>На постсоветской подвыборке месячная валютная модель не лучше монетки; годовая — заметно слабее, чем на мире. См. <a href="${href("check")}">«Проверку»</a>.</li>
          <li>Внезапные политические решения и внешние шоки (Украина-2014, Россия-2014) модели не видят заранее.</li>
          <li>Лаги публикации месячных рядов МВФ, кроме резервов, не учтены. Ревизии источников задним числом не покрыты.</li>
          <li>Интервал оценки не учитывает ошибку спецификации модели.</li>
        </ul>
      </div>
      <aside class="c5">
        <div class="card"><h3>Когда данные становятся известны</h3>
          <table class="factors" style="margin-top:8px"><tbody>
            ${m.lags.annual.map((l) => `<tr><td>${esc(l.label)}</td><td class="dt">${esc(l.rule)}</td></tr>`).join("")}
            ${m.lags.monthly.map((l) => `<tr><td>${esc(l.label)}</td><td class="dt">${esc(l.rule)}</td></tr>`).join("")}
          </tbody></table>
          <p class="tiny faint" style="margin-top:8px">Лаги консервативные и одинаковые для всех стран: это допущение, а не расписание релизов. Источник правил — <span class="mono">${esc(m.lags.file)}</span>.</p></div>
        <div class="card" style="margin-top:26px" id="data"><h3>Данные</h3>
          <dl class="kv">${m.sources.map((s) => `<dt class="mono">${esc(s.key)}</dt><dd>${esc(s.version ?? "актуальная версия на дату загрузки")}</dd>`).join("")}</dl>
          <ul class="plain small">${m.downloads.map((d) => `<li>${esc(d.label)}: <span class="mono">${esc(d.path)}</span></li>`).join("")}
            <li>Данные сайта (JSON): <a href="data/current.json">текущий срез</a>, <a href="data/scorecard.json">сводка качества</a>, <a href="data/archive.json">архив</a>, <a href="data/manifest.json">манифест сборки</a></li></ul>
          <p class="tiny faint" style="margin-top:8px">Данные сайта собраны ${dateRu(S.manifest.generated_at.slice(0, 10))} из таблиц с контрольными суммами, записанными в манифесте.</p></div>
      </aside>
    </div>
  </section>
  <section class="section"><div class="section-head"><div><h2>Словарь состояний</h2><p>Причины пропусков, единицы и даты на сайте всегда написаны словами.</p></div></div>${statesLegend()}</section>`;
  if (q.s) document.getElementById(q.s)?.scrollIntoView();
}
