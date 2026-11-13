// Каркас сайта: загрузка данных, маршруты по хэшу, поиск страны в шапке.
import { S, boot, parseHash, href, esc, dateRu, nameOf } from "./lib.js";
import { home } from "./views/home.js";
import { forecasts } from "./views/forecasts.js";
import { country } from "./views/country.js";
import { archive, record } from "./views/archive.js";
import { method } from "./views/pages.js";
import { check } from "./views/check.js";

const main = document.getElementById("main");
const ROUTES = { "": home, forecasts, country, archive, record, check, method };
let lastPath = null;
const SITE = "Селдон";
const TITLES = { "": "Селдон — прогнозы кризисов, которые можно проверить", forecasts: "Прогнозы по странам", check: "Проверка прогнозов",
  archive: "Архив выпусков", country: "Страна не найдена", record: "Запись прогноза", method: "Метод и данные" };
/** Заголовок вкладки называет экран: смену маршрута слышит скринридер и видно в истории браузера. */
function pageTitle(parts) {
  if (parts[0] === "country" && S.byIso.has((parts[1] ?? "").toUpperCase())) return `${nameOf(parts[1].toUpperCase())} — ${SITE}`;
  const t = TITLES[ROUTES[parts[0] ?? ""] ? parts[0] ?? "" : ""];
  return parts[0] && ROUTES[parts[0]] ? `${t} — ${SITE}` : TITLES[""];
}

async function render() {
  const { parts, q } = parseHash();
  const view = ROUTES[parts[0] ?? ""] ?? home;
  const path = parts.join("/");
  const samePage = path === lastPath;
  document.querySelectorAll(".nav a").forEach((a) => {
    const on = a.dataset.nav === (parts[0] === "record" ? "archive" : parts[0]);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  try {
    await view(main, parts.slice(1), q, { samePage });
  } catch (e) {
    console.error(e);
    main.innerHTML = `<section class="section"><h2>Не удалось показать страницу</h2><p class="lede" style="margin-top:10px">${esc(e.message)}</p></section>`;
  }
  document.title = pageTitle(parts);
  if (!samePage) {
    window.scrollTo(0, 0);
    // при переходе между экранами фокус уходит в начало содержимого (не при первой загрузке)
    if (lastPath !== null) main.focus({ preventScroll: true });
    lastPath = path;
  }
}

// ---------------------------------------------------------------- поиск страны
function setupSearch() {
  const input = document.getElementById("q"), list = document.getElementById("q-list");
  const norm = (s) => s.toLowerCase().replace(/ё/g, "е").replace(/[’'`\-–—]/g, " ").replace(/\s+/g, " ").trim();
  const index = S.countries.map((c) => ({ c, keys: [c.ru, c.en, c.iso3, ...c.aliases].map(norm) }));
  let active = 0, found = [];
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); };
  const draw = () => {
    const v = norm(input.value);
    if (!v) { close(); return; }
    found = index.map(({ c, keys }) => ({ c, score: Math.min(...keys.map((k) => (k === v ? 0 : k.startsWith(v) ? 1 : k.includes(v) ? 2 : 9))) }))
      .filter((x) => x.score < 9).sort((a, b) => a.score - b.score || a.c.ru.localeCompare(b.c.ru, "ru")).slice(0, 12).map((x) => x.c);
    active = Math.min(active, Math.max(found.length - 1, 0));
    list.innerHTML = found.length
      ? found.map((c, i) => `<li class="${i === active ? "on" : ""}" role="option" id="q-opt-${i}" aria-selected="${i === active}"><a tabindex="-1" href="${href("country/" + c.iso3)}"><span>${esc(c.ru)}</span><span class="faint">${esc(c.en)} · ${c.iso3}</span></a></li>`).join("")
      : `<li class="empty" role="option" aria-disabled="true" aria-selected="false">Страна не найдена. Попробуйте русское или английское название либо код ISO3.</li>`;
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    if (found.length) input.setAttribute("aria-activedescendant", `q-opt-${active}`); else input.removeAttribute("aria-activedescendant");
  };
  input.addEventListener("input", () => { active = 0; draw(); });
  input.addEventListener("focus", draw);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { active = Math.min(active + 1, found.length - 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); draw(); e.preventDefault(); }
    else if (e.key === "Enter" && found[active]) { location.hash = href("country/" + found[active].iso3); input.value = ""; close(); input.blur(); }
    else if (e.key === "Escape") { close(); input.blur(); }
  });
  list.addEventListener("click", () => { input.value = ""; close(); });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) close(); });
}

// «К содержанию»: якорь #main совпал бы с маршрутом по хэшу и увёл бы на главную — переносим фокус сами
document.querySelector(".skip").addEventListener("click", (e) => { e.preventDefault(); main.focus(); main.scrollIntoView(); });

boot().then(() => {
  setupSearch();
  document.getElementById("foot-build").textContent =
    `Данные собраны ${dateRu(S.manifest.generated_at.slice(0, 10))} · код ${S.manifest.code.commit.slice(0, 7)}`;
  window.addEventListener("hashchange", render);
  render();
}).catch((e) => {
  main.innerHTML = `<section class="section"><h2>Данные не загружены</h2><p class="lede" style="margin-top:10px">${esc(e.message)}</p></section>`;
});
