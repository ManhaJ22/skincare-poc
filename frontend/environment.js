/*
  Dermalis climate and environment: the optional climate card in the chat, the saved profile,
  the seasonal refresh check, and the lifestyle-module scaffold.

  A plain script (not a module), like camera.js. Exposes window.DermalisEnvironment.
  Privacy: the profile lives only in this browser's localStorage. Lookups go to our own backend
  (which asks Open-Meteo); nothing here logs places or coordinates.
*/
(() => {
  const ENV_KEY = "dermalis.environment"; // the EnvironmentProfile, rounded coordinates only
  const SETTINGS_KEY = "dermalis.settings"; // { lifestyle: bool, seasonDismissedAt: ISO date }
  const STALE_DAYS = 60;      // ask for a refresh when the saved profile is older than this
  const DISMISS_DAYS = 30;    // after "Not now", stay quiet at least this long
  const TROPICS_LAT = 23.5;   // mirrors environment_rules.TROPICS_LAT
  const DAY_MS = 24 * 60 * 60 * 1000;

  const CLIMATES = [
    ["hot_humid", "Hot and humid"],
    ["hot_dry", "Hot and dry"],
    ["mild", "Mild / temperate"],
    ["cold_dry", "Cold and dry"],
    ["cold_humid", "Cold and damp"],
  ];
  const CLIMATE_TEXT = { hot_humid: "hot and humid", hot_dry: "hot and dry", mild: "mild", cold_dry: "cold and dry", cold_humid: "cold and damp" };
  const DRY = new Set(["hot_dry", "cold_dry"]);

  // The only extra questions: quick single-tap choices. Values match the backend's EnvironmentProfile.
  const QUESTIONS = [
    { id: "indoor", legend: "Indoors, the air is usually…", options: [["heated", "Mostly heated"], ["ac", "Mostly air-conditioned"], ["mixed", "Neither / mixed"]] },
    { id: "outdoor_time", legend: "Time outdoors on a typical day", options: [["under_30m", "Under 30 min"], ["30m_2h", "30 min to 2 hours"], ["over_2h", "Over 2 hours"]] },
    { id: "sunscreen_habit", legend: "Do you wear sunscreen?", options: [["daily", "Daily"], ["sometimes", "Sometimes"], ["rarely", "Rarely or never"]] },
    { id: "travels", legend: "Travel often between very different climates?", options: [["yes", "Yes"], ["no", "No"]] },
  ];

  /*
    Lifestyle modules (the "Deeper lifestyle questions" setting, off by default).

    To add one, push an object here:
      {
        id: "sleep",                       // key under environment.lifestyle in requests
        title: "Sleep",                    // shown above its questions
        questions: [{ id, legend, options: [[value, label], ...] }],  // same shape as QUESTIONS
        toContext: (answers) => "Usually sleeps about 6 hours.",       // one plain sentence for the consultant
      }
    Only modules with questions are asked, and only while the setting is on. Their answers are sent as
    environment.lifestyle[id] = { answers, context }; the backend adds `context` to the consultant's prompt.
    Keep questions optional, single-tap, and free of medical or diagnostic wording.
  */
  const LIFESTYLE_MODULES = [
    { id: "placeholder", title: "Coming later", questions: [], toContext: () => "" }, // never asked
  ];

  let apiBase = "";
  let onChange = () => {};

  // --- Storage ------------------------------------------------------------------------
  function read(key) {
    try {
      return JSON.parse(localStorage.getItem(key)) || null;
    } catch {
      return null;
    }
  }
  function write(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Storage unavailable (private mode etc.); the profile still works for this page load.
    }
  }

  let memoryProfile = null; // used when storage is unavailable
  const load = () => read(ENV_KEY) || memoryProfile;
  const settings = () => read(SETTINGS_KEY) || {};
  const saveSettings = (patch) => write(SETTINGS_KEY, { ...settings(), ...patch });

  function save(profile) {
    memoryProfile = profile;
    write(ENV_KEY, profile);
    onChange();
  }

  // "Clear my location and environment data": the profile, lifestyle answers and the refresh timer.
  function clear() {
    memoryProfile = null;
    try {
      localStorage.removeItem(ENV_KEY);
    } catch {}
    const { seasonDismissedAt, ...rest } = settings();
    write(SETTINGS_KEY, rest);
    onChange();
  }

  // What goes to /analyze and /chat. Place and coordinates stay in the browser: the backend's
  // rules only need the climate facts and answers.
  function requestFields() {
    const env = load();
    if (!env) return {};
    const { place, lat, lon, lifestyle, ...facts } = env;
    const enabled = settings().lifestyle ? lifestyle : null;
    return { environment: enabled ? { ...facts, lifestyle: enabled } : facts };
  }

  // --- Helpers ------------------------------------------------------------------------
  function seasonFor(lat, now = new Date()) {
    if (lat == null || Math.abs(lat) < TROPICS_LAT) return null;
    const north = ["winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter"][now.getMonth()];
    if (lat >= 0) return north;
    return { winter: "summer", summer: "winter", spring: "autumn", autumn: "spring" }[north];
  }

  const uvWord = (uv) => (uv < 3 ? "low" : uv < 6 ? "moderate" : uv < 8 ? "high" : "very high");

  function describe(env) {
    if (!env) return "";
    const parts = [env.climate_label ? `${CLIMATE_TEXT[env.climate_label]}${env.source === "manual" ? " (you chose)" : ""}` : ""];
    if (env.uv_index_max != null) parts.push(`UV ${uvWord(env.uv_index_max)}`);
    if (env.place) parts.push(env.place);
    return parts.filter(Boolean).join(" · ");
  }

  async function lookup(body) {
    let res;
    try {
      res = await fetch(`${apiBase}/environment/lookup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      return { error: "unavailable" };
    }
    if (res.status === 404) return { error: "not_found" };
    if (!res.ok) return { error: "unavailable" };
    try {
      return { data: await res.json() };
    } catch {
      return { error: "unavailable" };
    }
  }

  function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    node.append(...children.filter((c) => c != null && c !== false));
    return node;
  }

  let uid = 0;
  // A row of single-tap radio chips: arrow keys move within it, and the checked one shows a ✓.
  function chipGroup(question, selected, onPick) {
    const name = `env-${question.id}-${++uid}`;
    return h("fieldset", { class: "env-group" },
      h("legend", {}, question.legend),
      h("div", { class: "env-chips" }, ...question.options.map(([value, label]) =>
        h("label", { class: "env-chip" },
          h("input", { type: "radio", name, value, checked: value === selected, onchange: () => onPick(value) }),
          h("span", {}, label)))));
  }

  // --- The climate card in the chat ---------------------------------------------------------
  // onDone(profile) after Save; onSkip() after Skip. Starts from the saved profile when editing.
  function buildCard({ onDone, onSkip }) {
    const saved = load();
    const draft = saved ? { ...saved } : {};
    const card = h("section", { class: "env-card", "aria-label": "Your climate (optional)" });

    const render = (stage, message = "") => {
      card.replaceChildren(h("h3", {}, "Your climate", h("span", { class: "env-optional" }, " (optional)")));
      if (stage === "ask") renderAsk(message);
      else renderAnswers();
    };

    const climateChips = () => h("div", { class: "env-manual" },
      chipGroup({ id: "climate", legend: "Choose your climate", options: CLIMATES }, draft.source === "manual" ? draft.climate_label : null,
        (value) => {
          // A manual choice replaces the looked-up climate but keeps the general place, if any.
          // The measured UV still applies; temperature and humidity would contradict the chosen label.
          Object.assign(draft, { climate_label: value, source: "manual", avg_temp_c: null, avg_humidity: null });
          if (draft.lat == null) draft.season = null;
          render("answers");
        }));

    function renderAsk(message) {
      const input = h("input", { type: "text", id: `env-query-${++uid}`, autocomplete: "postal-code", maxlength: "100", required: true, placeholder: "e.g. Denver or 80202" });
      const status = h("p", { class: "env-status", "aria-live": "polite" }, message);
      const lookupBtn = h("button", { type: "submit" }, "Look up");
      const form = h("form", {
        class: "env-lookup",
        onsubmit: async (e) => {
          e.preventDefault();
          const query = input.value.trim();
          if (query.length < 2) return;
          lookupBtn.disabled = true;
          status.textContent = "Looking up your climate…";
          const { data, error } = await lookup({ query });
          lookupBtn.disabled = false;
          if (error === "not_found") {
            status.textContent = "We couldn't find that place. Try a nearby city, or choose your climate below.";
            input.focus();
            return;
          }
          if (error) {
            // Never block onboarding: the manual choice is right there.
            status.textContent = "Climate data isn't available right now. Choose your climate below instead.";
            card.querySelector(".env-manual input")?.focus();
            return;
          }
          Object.assign(draft, {
            place: data.place, lat: data.lat, lon: data.lon, season: data.season, climate_label: data.climate_label,
            avg_temp_c: data.avg_temp_c, avg_humidity: data.avg_humidity, uv_index_max: data.uv_index_max, source: "lookup",
            summary: data.summary,
          });
          render("answers");
        },
      },
        h("label", { for: input.id }, "City or ZIP code"),
        h("div", { class: "env-row" }, input, lookupBtn));
      card.append(
        h("p", {}, "Dermalis uses your general location to adapt advice to your climate and the season. We never need your exact address."),
        form, status, climateChips(),
        h("div", { class: "env-actions" }, h("button", { type: "button", class: "secondary", onclick: skip }, "Skip")));
    }

    function renderAnswers() {
      const detected = draft.source === "lookup"
        ? (draft.summary || `Currently ${CLIMATE_TEXT[draft.climate_label]} where you are.`)
        : `You chose: ${CLIMATE_TEXT[draft.climate_label]}.`;
      const correct = h("details", { class: "env-correct" },
        h("summary", {}, draft.source === "lookup" ? "Not right? Choose your climate" : "Change climate"), climateChips());
      const lifestyle = settings().lifestyle
        ? LIFESTYLE_MODULES.filter((m) => m.questions.length).map((m) => h("div", { class: "env-module" },
          h("h4", {}, m.title),
          ...m.questions.map((q) => chipGroup(q, draft.lifestyle?.[m.id]?.answers?.[q.id], (value) => {
            const answers = { ...(draft.lifestyle?.[m.id]?.answers || {}), [q.id]: value };
            draft.lifestyle = { ...(draft.lifestyle || {}), [m.id]: { answers, context: m.toContext(answers) } };
          }))))
        : [];
      card.append(
        h("p", { class: "env-detected" }, detected, draft.place ? h("span", { class: "env-place" }, ` ${draft.place}`) : null),
        correct,
        ...QUESTIONS.map((q) => {
          const current = q.id === "travels" ? (draft.travels == null ? null : draft.travels ? "yes" : "no") : draft[q.id];
          return chipGroup(q, current, (value) => (draft[q.id] = q.id === "travels" ? value === "yes" : value));
        }),
        ...lifestyle,
        h("p", { class: "env-fine" }, "All optional. These are factors that can affect skin; they shape suggestions, not a diagnosis."),
        h("div", { class: "env-actions" },
          h("button", { type: "button", onclick: done }, "Save"),
          h("button", { type: "button", class: "secondary", onclick: skip }, saved ? "Cancel" : "Skip")));
      card.querySelector(":scope > .env-group input")?.focus({ preventScroll: true });
    }

    function done() {
      const { summary, ...rest } = draft;
      const profile = { ...rest, updated_at: new Date().toISOString() };
      save(profile);
      card.replaceChildren(h("p", { class: "env-saved" }, h("strong", {}, "Climate saved: "), describe(profile)));
      onDone?.(profile);
    }

    function skip() {
      card.remove();
      onSkip?.();
    }

    render(draft.climate_label ? "answers" : "ask");
    return card;
  }

  // --- Seasonal refresh ---------------------------------------------------------------------
  // Resolves to null, or { message, apply() } for one gentle prompt. Only calls our backend when
  // the season turned over or the profile is stale, never on every visit.
  async function seasonalCheck(now = new Date()) {
    const env = load();
    if (!env?.updated_at) return null;
    const dismissed = settings().seasonDismissedAt;
    if (dismissed && now - new Date(dismissed) < DISMISS_DAYS * DAY_MS) return null;
    const stale = now - new Date(env.updated_at) > STALE_DAYS * DAY_MS;
    const season = seasonFor(env.lat, now);
    const seasonChanged = env.source === "lookup" && season && env.season && season !== env.season;
    if (!stale && !seasonChanged) return null;

    if (env.source !== "lookup" || env.lat == null) {
      return { message: "It's been a while since you set your climate. Want to check it's still right?", edit: true };
    }
    const { data } = await lookup({ lat: env.lat, lon: env.lon });
    if (!data) return null; // try again next visit
    const changes = [];
    if (DRY.has(data.climate_label) && !DRY.has(env.climate_label)) changes.push("drier");
    else if (env.avg_humidity != null && env.avg_humidity - data.avg_humidity >= 15) changes.push("drier");
    else if (env.avg_humidity != null && data.avg_humidity - env.avg_humidity >= 15) changes.push("more humid");
    if (env.avg_temp_c != null && env.avg_temp_c - data.avg_temp_c >= 8) changes.push("colder");
    else if (env.avg_temp_c != null && data.avg_temp_c - env.avg_temp_c >= 8) changes.push("warmer");
    const climateChanged = data.climate_label !== env.climate_label || changes.length;
    if (!seasonChanged && !climateChanged && !stale) return null;

    const lead = data.season && data.season !== env.season ? `${data.season[0].toUpperCase()}${data.season.slice(1)}'s here` : "The weather's shifted";
    const how = changes.length ? ` and it's gotten ${changes.join(" and ")} where you are` : "";
    return {
      message: `${lead}${how}. Want a seasonal routine update?`,
      apply() {
        const { place, lat, lon } = env;
        save({ ...env, ...data, place, lat, lon, summary: undefined, updated_at: new Date().toISOString() });
      },
    };
  }

  const dismissSeasonal = () => saveSettings({ seasonDismissedAt: new Date().toISOString() });

  window.DermalisEnvironment = {
    init(options) {
      apiBase = options.apiBase;
      onChange = options.onChange || onChange;
    },
    load,
    clear,
    describe,
    requestFields,
    buildCard,
    seasonalCheck,
    dismissSeasonal,
    lifestyleEnabled: () => Boolean(settings().lifestyle),
    setLifestyle: (on) => saveSettings({ lifestyle: Boolean(on) }),
  };
})();
