// Progress page: the predicted timeline for each product in the saved routine, plus progress photos.
// Data comes from the backend's /sessions endpoints (backend/services/progress_store.py).
(() => {
  const API_BASE = location.pathname.startsWith("/app/") ? location.origin : "http://localhost:8000";
  const LAST_SESSION_KEY = "dermalis.lastSessionId";
  const MAX_PHOTO_SIDE = 1280;
  const MIN_WEEKS_SHOWN = 24;
  const DAY_MS = 24 * 60 * 60 * 1000;

  const main = document.getElementById("main");
  let session = null;

  // --- Small helpers ----------------------------------------------------------------------------
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "style") node.style.cssText = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  let toastTimer = null;
  function toast(text) {
    const box = document.getElementById("toast");
    box.textContent = text;
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (box.hidden = true), 6000);
  }

  async function api(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
      const err = new Error(typeof detail === "string" ? detail : `Request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return res.status === 204 ? null : res.json();
  }

  function parseDate(iso) {
    const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
    return new Date(y, m - 1, d);
  }
  function todayIso() {
    const now = new Date();
    return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  }
  function weekOf(iso) {
    return Math.floor((parseDate(iso) - parseDate(session.start_date)) / (7 * DAY_MS));
  }
  function niceDate(iso) {
    return parseDate(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }
  function range([lo, hi]) {
    return lo === hi ? `week ${lo}` : `weeks ${lo}–${hi}`;
  }
  function photoUrl(photo) {
    return `${API_BASE}/sessions/${encodeURIComponent(session.session_id)}/photos/${photo.photo_id}`;
  }

  // --- Loading ----------------------------------------------------------------------------------
  function requestedSessionId() {
    const fromUrl = new URLSearchParams(location.search).get("session");
    if (fromUrl) return fromUrl;
    try {
      return localStorage.getItem(LAST_SESSION_KEY);
    } catch {
      return null;
    }
  }

  async function load() {
    const id = requestedSessionId();
    try {
      session = id ? await api(`/sessions/${encodeURIComponent(id)}`).catch((err) => {
        if (err.status === 404 || err.status === 400) return api("/sessions/latest");
        throw err;
      }) : await api("/sessions/latest");
    } catch (err) {
      main.replaceChildren(err.status === 404 ? noSession() : loadError(err));
      return;
    }
    try {
      localStorage.setItem(LAST_SESSION_KEY, session.session_id);
    } catch {
      // Storage unavailable; the page still works from the URL.
    }
    render();
  }

  function noSession() {
    return el("section", { class: "panel empty" },
      el("span", { class: "hand" }, "Build your routine first"),
      el("p", {}, "Once Dermalis has put your routine together, you can track how it's going here."),
      el("a", { class: "link-btn", href: "./" }, "Start a consultation"),
    );
  }

  function loadError(err) {
    return el("section", { class: "panel" },
      el("h2", {}, "Couldn't load your progress"),
      el("p", {}, `${err.message}. Is the backend running on ${API_BASE}?`),
      el("button", { type: "button", onclick: () => location.reload() }, "Try again"),
    );
  }

  // --- Rendering --------------------------------------------------------------------------------
  function render() {
    main.replaceChildren(
      el("div", { class: "row" }, startPanel(), nowPanel()),
      timelinePanel(),
      photosPanel(),
    );
  }

  function currentWeek() {
    return weekOf(todayIso());
  }

  function startPanel() {
    const week = currentWeek();
    const badge = week < 0
      ? el("div", { class: "week-badge" }, "Soon", el("small", {}, `starts ${niceDate(session.start_date)}`))
      : el("div", { class: "week-badge" }, `Week ${week}`, el("small", {}, "of your routine"));
    const input = el("input", { type: "date", id: "start-date", value: session.start_date, max: todayIso() });
    input.addEventListener("change", async () => {
      if (!input.value) return;
      try {
        session = await api(`/sessions/${encodeURIComponent(session.session_id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ start_date: input.value }),
        });
        render();
        toast("Start date updated.");
      } catch (err) {
        toast(`Couldn't update the date: ${err.message}`);
        input.value = session.start_date;
      }
    });
    return el("section", { class: "panel" },
      el("h2", {}, "Your routine"),
      session.conversation_summary ? el("p", {}, session.conversation_summary) : null,
      el("div", { class: "start" },
        badge,
        el("div", {},
          el("label", { for: "start-date" }, "I started on"),
          input,
          el("div", { class: "muted" }, "Change this if you started later than the day you built the routine."),
        ),
      ),
    );
  }

  // Where each product is today, in plain words.
  function status(m, week) {
    const [firstLo, firstHi] = m.first_signs_weeks;
    const [fullLo, fullHi] = m.full_effect_weeks;
    if (m.kind === "protective") return "Every day: it's working in the background.";
    if (week < firstLo) return `First signs usually show in ${range(m.first_signs_weeks)}. ${firstLo - week} week${firstLo - week === 1 ? "" : "s"} to go.`;
    if (week <= firstHi) return "First signs usually show around now. Compare with your earliest photo.";
    if (week < fullLo) return `Keep going: fuller results usually by ${range(m.full_effect_weeks)}.`;
    if (week <= fullHi) return "Fuller results usually show around now.";
    return "Should be close to its full effect by now.";
  }

  function nowPanel() {
    const week = Math.max(0, currentWeek());
    const active = session.timeline.filter((m) => m.kind !== "protective");
    const inWindow = active.filter((m) => week >= m.first_signs_weeks[0] && week <= m.full_effect_weeks[1]);
    const upcoming = active
      .filter((m) => week < m.first_signs_weeks[0])
      .sort((a, b) => a.first_signs_weeks[0] - b.first_signs_weeks[0])[0];
    const label = (m) => [m.ingredient || m.product_name, m.concern].filter(Boolean).join(" for ");
    const items = [
      ...inWindow.map((m) => el("li", {}, el("b", {}, label(m)), ": ", status(m, week))),
      upcoming ? el("li", { class: "next" }, el("b", {}, `Next up, ${label(upcoming)}`), ": ", status(upcoming, week)) : null,
    ].filter(Boolean);
    return el("section", { class: "panel" },
      el("h2", {}, "Around now"),
      session.timeline_error ? el("p", { class: "muted" }, session.timeline_error) : null,
      items.length
        ? el("ul", { class: "now-list" }, items)
        : el("p", { class: "muted" }, session.timeline.length ? "Nothing new this week; keep going!" : "No timeline yet."),
      el("p", { class: "muted" }, "Typical timings from research. Everyone's skin is different; consistency matters most."),
    );
  }

  function timelinePanel() {
    const timeline = session.timeline;
    if (!timeline.length) {
      return el("section", { class: "panel" },
        el("h2", {}, "When to expect results"),
        el("p", { class: "muted" }, session.timeline_error
          ? "The timeline couldn't be predicted just now. Go back and press “Track your progress” again to retry."
          : "There are no products in this routine to track."),
      );
    }
    const week = currentWeek();
    const maxWeek = Math.ceil(Math.max(MIN_WEEKS_SHOWN, week + 2, ...timeline.map((m) => m.full_effect_weeks[1])) / 4) * 4;
    const pct = (w) => `${(Math.min(Math.max(w, 0), maxWeek) / maxWeek) * 100}%`;
    const span = ([lo, hi]) => `left:${pct(lo)};width:calc(${pct(hi)} - ${pct(lo)} + 6px)`;

    const rows = timeline.map((m) => {
      const bars = m.kind === "protective"
        ? [el("span", { class: "bar", style: "left:0;right:0", title: "Every day" })]
        : [
            el("span", { class: "bar full", style: span(m.full_effect_weeks), title: `Fuller results: ${range(m.full_effect_weeks)}` }),
            el("span", { class: "bar first", style: span(m.first_signs_weeks), title: `First signs: ${range(m.first_signs_weeks)}` }),
          ];
      return el("div", { class: `tl-row kind-${m.kind}` },
        el("div", { class: "tl-name" }, m.product_name, el("span", { class: "ing" }, [m.ingredient, m.concern].filter(Boolean).join(" · "))),
        el("div", { class: "track", role: "img",
          "aria-label": m.kind === "protective" ? `${m.product_name}: protects every day`
            : `${m.product_name}: first signs ${range(m.first_signs_weeks)}, fuller results ${range(m.full_effect_weeks)}` }, bars),
        m.what_to_look_for ? el("div", { class: "tl-look" }, m.what_to_look_for) : null,
      );
    });

    const ticks = [];
    for (let w = 0; w <= maxWeek; w += 4) ticks.push(el("span", { class: "tick", style: `left:${pct(w)}` }, w === 0 ? "Start" : `wk ${w}`));
    const dots = session.photos
      .filter((p) => weekOf(p.taken_at) >= 0)
      .map((p) => el("span", { class: "dot", style: `left:${pct(weekOf(p.taken_at))}`, title: `Photo, ${niceDate(p.taken_at)}` }));

    return el("section", { class: "panel" },
      el("h2", {}, "When to expect results"),
      el("p", { class: "legend" },
        el("span", {}, el("i", { style: "background:var(--terracotta-300)" }), "first signs"),
        el("span", {}, el("i", { style: "background:var(--terracotta-500)" }), "fuller results"),
        el("span", {}, el("i", { style: "background:var(--sage-300)" }), "comfort / barrier"),
        el("span", {}, el("i", { style: "background:var(--mustard-300)" }), "daily protection"),
        el("span", {}, el("i", { style: "background:var(--navy-500);border-radius:50%;width:10px" }), "your photos"),
      ),
      el("div", { class: "tl" },
        rows,
        el("div", { class: "axis" }, el("span"), el("div", { class: "axis-track" }, ticks, dots)),
        week >= 0 && week <= maxWeek
          ? el("div", { class: "tl-lines", "aria-hidden": "true" }, el("div", { class: "today", style: `left:${pct(week)}` }, el("span", {}, "today")))
          : null,
      ),
    );
  }

  function photosPanel() {
    const fileInput = el("input", { type: "file", id: "photo-input", accept: "image/*", capture: "user" });
    const dateInput = el("input", { type: "date", id: "photo-date", value: todayIso(), max: todayIso() });
    const noteInput = el("input", { type: "text", id: "photo-note", maxlength: "200", placeholder: "e.g. morning light, no makeup" });
    const addBtn = el("button", { type: "button", onclick: () => fileInput.click() }, "Add a photo");
    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      fileInput.value = "";
      if (!file) return;
      addBtn.disabled = true;
      addBtn.textContent = "Uploading…";
      try {
        await uploadPhoto(file, dateInput.value || todayIso(), noteInput.value.trim());
        noteInput.value = "";
        toast("Photo saved.");
      } catch (err) {
        toast(`Couldn't save the photo: ${err.message}`);
      } finally {
        addBtn.disabled = false;
        addBtn.textContent = "Add a photo";
      }
    });

    const photos = [...session.photos].sort((a, b) => a.taken_at.localeCompare(b.taken_at));
    return el("section", { class: "panel" },
      el("h2", {}, "Progress photos"),
      el("p", { class: "muted" }, "Same spot, same light, every couple of weeks makes changes easier to see. Photos are stored on this demo's server."),
      el("div", { class: "upload" },
        el("label", { class: "field" }, "Taken on", dateInput),
        el("label", { class: "field" }, "Note (optional)", noteInput),
        addBtn,
        fileInput,
      ),
      photos.length ? null : el("div", { class: "empty" },
        el("span", { class: "hand" }, "Take a baseline photo today"),
        "It's the one you'll be glad you have in a couple of months.",
      ),
      photos.length >= 2 ? comparePanel(photos) : null,
      photos.length ? el("div", { class: "gallery" }, photos.map(photoCard)) : null,
    );
  }

  function weekLabel(photo) {
    const w = weekOf(photo.taken_at);
    return w < 0 ? "Before" : `Week ${w}`;
  }

  function photoCard(photo) {
    const del = el("button", { type: "button", class: "secondary small", "aria-label": `Delete photo from ${niceDate(photo.taken_at)}` }, "Delete");
    del.addEventListener("click", async () => {
      // Two clicks to delete, without a blocking confirm() dialog.
      if (del.dataset.armed !== "1") {
        del.dataset.armed = "1";
        del.textContent = "Sure?";
        setTimeout(() => {
          del.dataset.armed = "";
          del.textContent = "Delete";
        }, 3000);
        return;
      }
      try {
        await api(`/sessions/${encodeURIComponent(session.session_id)}/photos/${photo.photo_id}`, { method: "DELETE" });
        session.photos = session.photos.filter((p) => p.photo_id !== photo.photo_id);
        render();
        toast("Photo deleted.");
      } catch (err) {
        toast(`Couldn't delete it: ${err.message}`);
      }
    });
    return el("figure", { class: "shot" },
      el("img", { src: photoUrl(photo), alt: `Progress photo, ${weekLabel(photo)}`, loading: "lazy" }),
      el("figcaption", {},
        el("span", {}, el("span", { class: "wk" }, weekLabel(photo)), el("span", { class: "note" }, niceDate(photo.taken_at)),
          photo.note ? el("span", { class: "note" }, photo.note) : null),
        del,
      ),
    );
  }

  function comparePanel(photos) {
    const side = (initial) => {
      const img = el("img", { alt: "" });
      const select = el("select", { "aria-label": "Choose a photo to compare" },
        photos.map((p) => el("option", { value: p.photo_id }, `${weekLabel(p)} · ${niceDate(p.taken_at)}`)));
      const show = () => {
        const photo = photos.find((p) => p.photo_id === select.value);
        img.src = photoUrl(photo);
        img.alt = `Progress photo, ${weekLabel(photo)}`;
      };
      select.value = initial.photo_id;
      select.addEventListener("change", show);
      show();
      return el("figure", {}, select, img);
    };
    return el("div", {},
      el("h3", { style: "margin:4px 0 0;font-size:16px" }, "Compare"),
      el("div", { class: "compare" }, side(photos[0]), side(photos[photos.length - 1])),
      el("hr", { style: "border:0;border-top:2px dotted var(--sage-300);margin:18px 0" }),
    );
  }

  // --- Upload: downscale to a JPEG first, like the main page does before /analyze -------------
  function resizeToJpeg(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, MAX_PHOTO_SIDE / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(img.src);
        canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not process that image"))), "image/jpeg", 0.88);
      };
      img.onerror = () => reject(new Error("Could not read that image. Try a JPEG or PNG"));
      img.src = URL.createObjectURL(file);
    });
  }

  async function uploadPhoto(file, takenAt, note) {
    const form = new FormData();
    form.append("file", await resizeToJpeg(file), "progress.jpg");
    form.append("taken_at", takenAt);
    if (note) form.append("note", note);
    const photo = await api(`/sessions/${encodeURIComponent(session.session_id)}/photos`, { method: "POST", body: form });
    session.photos.push(photo);
    render();
  }

  load();
})();
