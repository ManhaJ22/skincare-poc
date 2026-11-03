/*
  Dermalis in-app camera: live preview with a face oval, a lighting check and a review step.

  A plain script (not a module) so the page still works when opened straight from disk.
  Exposes window.DermalisCamera = { supported, insecure, open({ onAccept, onLibrary }) }.
  Photos stay in memory only: nothing is written to storage and no image data is logged.
*/
(() => {
  // --- Lighting thresholds ------------------------------------------------------------
  // Starting points, NOT validated values. Before relying on them, check them on faces across
  // the full Monk Skin Tone Scale (1-10) in good and poor light: open the page with
  // ?lightdebug=1 and note the live numbers for each person.
  //
  // Fairness: brightness is the mean over the whole face oval, so deeper skin in good light
  // reads lower than lighter skin in the same light (the Monk 10 reference swatch #292420 has
  // a luma of about 37). MIN_BRIGHTNESS is therefore set low, so a deep skin tone is never told
  // "too dark" just for being deep; only a genuinely underexposed frame should trip it.
  // The same holds at the light end: the Monk 1 swatch #f6ede4 has a luma of about 238, so
  // MAX_BRIGHTNESS sits above that and blown-out light is caught by the clipped-white share.
  const MIN_BRIGHTNESS = 25;          // mean luma (0-255) inside the oval below this: "too dark"
  const MAX_BRIGHTNESS = 245;         // mean luma above this: "too bright"
  const MAX_SIDE_DIFF = 0.35;         // |left - right| / mean above this: "one side in shadow".
                                      // Relative, so deep and light skin are judged alike.
  const MAX_CLIPPED_FRACTION = 0.15;  // share of oval pixels that are clipped (near white or near black)
  const CLIP_HIGH = 250;              // luma above this counts as clipped white
  const CLIP_LOW = 10;                // luma below this counts as clipped black
  const SAMPLE_MS = 400;              // how often the live preview is measured
  const SMOOTH_SAMPLES = 4;           // readings averaged so the message doesn't flicker
  const SAMPLE_MAX_SIDE = 160;        // the offscreen sample canvas is about 160x120

  // Face oval as fractions of the 3:4 preview frame. The guide SVG is drawn from these too,
  // so what the user lines up with is exactly what gets measured.
  const OVAL = { cx: 0.5, cy: 0.47, rx: 0.32, ry: 0.33 };

  const MESSAGES = {
    dark: "Too dark: move closer to a window or turn on a light",
    bright: "Too bright: move away from direct light",
    uneven: "One side of your face is in shadow: turn toward the light",
    good: "Lighting looks good",
    starting: "Starting camera…",
    denied:
      "Camera access is turned off for this site. To turn it back on, click the camera or lock icon " +
      "in the address bar and allow the camera (on iPhone: Settings › Apps › Safari › Camera), then " +
      "try again. Or choose a photo from your library.",
    notFound: "We couldn't find a camera on this device. You can choose a photo from your library instead.",
    inUse: "Your camera is being used by another app. Close that app (a video call, for example) and try again.",
    insecure:
      "The camera only works when this page is opened over HTTPS or on localhost. " +
      "You can choose a photo from your library instead.",
    unsupported: "This browser can't open the camera here. You can choose a photo from your library instead.",
    failed: "The camera didn't start. Try again, or choose a photo from your library.",
    captureFailed: "That photo didn't save. Please try again.",
  };

  const ICONS = {
    good: '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><path d="M7 12.5l3.2 3.2L17 9" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    warn: '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M12 3L2.5 20h19z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4.5" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><circle cx="12" cy="17.3" r="1.3" fill="currentColor"/></svg>',
    info: '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="8" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="16" cy="12" r="1.4" fill="currentColor"/></svg>',
  };

  const DEBUG = new URLSearchParams(location.search).get("lightdebug") === "1";
  const insecure = window.isSecureContext === false;
  const supported = !insecure && !!navigator.mediaDevices?.getUserMedia;

  const $ = (id) => document.getElementById(id);
  const dialog = $("camera-dialog");
  const video = $("cam-video");
  const photo = $("cam-photo");
  const captureBtn = $("cam-capture");
  const flipBtn = $("cam-flip");
  const debugBox = $("cam-debug");
  const debugText = $("cam-debug-text");
  const debugCanvas = $("cam-debug-canvas");
  const screens = ["intro", "error", "live", "review"];

  const cam = {
    stream: null,
    facing: "user",
    token: 0,          // bumped whenever a pending getUserMedia result should be thrown away
    timer: null,
    readings: [],
    blob: null,        // the captured photo, until it is accepted or retaken
    photoUrl: null,    // object URL shown in the review step; revoked as soon as it's not shown
    onAccept: null,
    onLibrary: null,
  };

  const sampleCanvas = document.createElement("canvas");
  const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });

  // Draw the guide from OVAL: a dimmed surround with an oval cut-out and a dashed outline.
  for (const id of ["cam-oval-hole", "cam-oval-line"]) {
    const e = $(id);
    e.setAttribute("cx", OVAL.cx * 300);
    e.setAttribute("cy", OVAL.cy * 400);
    e.setAttribute("rx", OVAL.rx * 300);
    e.setAttribute("ry", OVAL.ry * 400);
  }
  debugBox.hidden = !DEBUG;

  function show(name) {
    for (const s of screens) $(`cam-${s}`).hidden = s !== name;
  }

  function setStatus(el, tone, text) {
    el.dataset.tone = tone;
    el.innerHTML = ICONS[tone];
    el.append(Object.assign(document.createElement("span"), { textContent: text }));
  }

  // --- Opening and closing -----------------------------------------------------------
  async function open({ onAccept, onLibrary }) {
    cam.onAccept = onAccept;
    cam.onLibrary = onLibrary;
    show("intro");
    if (!dialog.open) dialog.showModal();
    if (!supported) return showError(insecure ? MESSAGES.insecure : MESSAGES.unsupported, false);
    $("cam-start").focus();
    // The explanation is shown before the browser asks; once access is granted there's nothing to ask.
    if ((await permissionState()) === "granted" && dialog.open && !$("cam-intro").hidden) start();
  }

  async function permissionState() {
    try {
      return (await navigator.permissions.query({ name: "camera" })).state;
    } catch {
      return "prompt"; // not supported (older Safari, Firefox): show the explanation
    }
  }

  function close() {
    cleanup();
    if (dialog.open) dialog.close();
  }

  // Camera off, photo released, back to the start.
  function cleanup() {
    cam.token++;
    stopStream();
    revokePhoto();
    cam.blob = null;
    show("intro");
  }

  // Esc also closes the dialog. The event is async: if the camera was reopened in the meantime,
  // leave the new session alone (close() already cleaned up the old one).
  dialog.addEventListener("close", () => {
    if (!dialog.open) cleanup();
  });

  // --- The stream --------------------------------------------------------------------
  async function start() {
    stopStream(); // the old stream's tracks are stopped before a new one is requested
    const token = ++cam.token;
    show("live");
    $("cam-paused").hidden = true;
    setStatus($("cam-status"), "info", MESSAGES.starting);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: cam.facing, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      // Closed, hidden or flipped again while the browser was asking: don't keep this camera on.
      if (token !== cam.token || !dialog.open || document.hidden) return stopTracks(stream);
      cam.stream = stream;
      const facing = stream.getVideoTracks()[0]?.getSettings().facingMode || cam.facing;
      video.classList.toggle("mirrored", facing !== "environment");
      video.srcObject = stream;
      video.play().catch(() => {}); // autoplay covers this; play() rejects if a flip interrupts it
      updateFlip();
    } catch (err) {
      if (token !== cam.token) return;
      const [message, canRetry, libraryFirst] = errorMessage(err);
      showError(message, canRetry, libraryFirst);
    }
  }

  // [message, can retry, library is the main way forward]
  function errorMessage(err) {
    switch (err?.name) {
      case "NotAllowedError":
      case "SecurityError":
        // Retrying only helps after the user changes the setting, so the library comes first.
        return [MESSAGES.denied, true, true];
      case "NotFoundError":
      case "OverconstrainedError":
        return [MESSAGES.notFound, false];
      case "NotReadableError":
      case "AbortError":
        return [MESSAGES.inUse, true];
      default:
        return [MESSAGES.failed, true];
    }
  }

  function showError(message, canRetry, libraryFirst = !canRetry) {
    show("error");
    $("cam-error-text").textContent = message;
    const retry = $("cam-retry");
    const library = $("cam-error-library");
    retry.hidden = !canRetry;
    retry.classList.toggle("secondary", libraryFirst);
    library.classList.toggle("secondary", !libraryFirst);
    const [primary, secondary] = libraryFirst ? [library, retry] : [retry, library];
    $("cam-error-actions").replaceChildren(primary, secondary);
    primary.focus();
  }

  const stopTracks = (stream) => stream.getTracks().forEach((t) => t.stop());

  function stopStream() {
    stopSampling();
    if (cam.stream) stopTracks(cam.stream);
    cam.stream = null;
    video.srcObject = null;
    captureBtn.disabled = true;
  }

  async function updateFlip() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      flipBtn.hidden = devices.filter((d) => d.kind === "videoinput").length < 2;
    } catch {
      flipBtn.hidden = false;
    }
  }

  video.addEventListener("playing", () => {
    if (!cam.stream) return;
    captureBtn.disabled = false;
    startSampling();
  });

  // Hidden tab or app: the camera light goes off; the user resumes it when they come back.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden || !dialog.open || $("cam-live").hidden) return;
    cam.token++;
    stopStream();
    $("cam-paused").hidden = false;
    setStatus($("cam-status"), "info", "Camera paused");
  });
  window.addEventListener("pagehide", () => {
    cam.token++;
    stopStream();
  });

  // --- Lighting check ----------------------------------------------------------------
  function startSampling() {
    stopSampling();
    cam.readings = [];
    sample();
    cam.timer = setInterval(sample, SAMPLE_MS);
  }

  function stopSampling() {
    clearInterval(cam.timer);
    cam.timer = null;
  }

  // Map the on-screen oval through object-fit: cover into video pixels, so only the face area
  // the user lined up is measured. The oval is centred, so mirroring doesn't move it.
  function ovalInVideo() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const fw = video.clientWidth || 3;
    const fh = video.clientHeight || 4;
    const scale = Math.max(fw / vw, fh / vh);
    const ox = (fw - vw * scale) / 2;
    const oy = (fh - vh * scale) / 2;
    return {
      cx: (OVAL.cx * fw - ox) / scale,
      cy: (OVAL.cy * fh - oy) / scale,
      rx: (OVAL.rx * fw) / scale,
      ry: (OVAL.ry * fh) / scale,
    };
  }

  // Mean luma, left/right halves and clipped share over every pixel inside the oval.
  // `region` is in source pixels; the source is scaled down to about 160x120 first.
  function measure(source, width, height, region) {
    const k = SAMPLE_MAX_SIDE / Math.max(width, height);
    const w = Math.round(width * k);
    const h = Math.round(height * k);
    sampleCanvas.width = w;
    sampleCanvas.height = h;
    sampleCtx.drawImage(source, 0, 0, w, h);
    const { data } = sampleCtx.getImageData(0, 0, w, h);
    const cx = region.cx * k;
    const cy = region.cy * k;
    const rx = region.rx * k;
    const ry = region.ry * k;
    let sum = 0, n = 0, left = 0, nLeft = 0, right = 0, nRight = 0, high = 0, low = 0;
    for (let y = Math.max(0, Math.floor(cy - ry)); y < Math.min(h, Math.ceil(cy + ry)); y++) {
      for (let x = Math.max(0, Math.floor(cx - rx)); x < Math.min(w, Math.ceil(cx + rx)); x++) {
        const dx = (x + 0.5 - cx) / rx;
        const dy = (y + 0.5 - cy) / ry;
        if (dx * dx + dy * dy > 1) continue;
        const i = (y * w + x) * 4;
        const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]; // Rec. 601
        sum += luma;
        n++;
        if (x + 0.5 < cx) { left += luma; nLeft++; } else { right += luma; nRight++; }
        if (luma > CLIP_HIGH) high++;
        else if (luma < CLIP_LOW) low++;
      }
    }
    if (!n) return null;
    return { mean: sum / n, left: left / (nLeft || 1), right: right / (nRight || 1), high: high / n, low: low / n };
  }

  function average(readings) {
    const valid = readings.filter(Boolean);
    if (!valid.length) return null;
    const avg = {};
    for (const key of Object.keys(valid[0])) avg[key] = valid.reduce((s, r) => s + r[key], 0) / valid.length;
    return avg;
  }

  const sideDiff = (m) => Math.abs(m.left - m.right) / Math.max(m.mean, 1);

  function judge(m) {
    if (!m) return null;
    const clipped = m.high + m.low;
    if (m.mean < MIN_BRIGHTNESS || (clipped > MAX_CLIPPED_FRACTION && m.low >= m.high)) return "dark";
    if (m.mean > MAX_BRIGHTNESS || clipped > MAX_CLIPPED_FRACTION) return "bright";
    if (sideDiff(m) > MAX_SIDE_DIFF) return "uneven";
    return "good";
  }

  function sample() {
    if (!video.videoWidth || document.hidden) return;
    const region = ovalInVideo();
    cam.readings.push(measure(video, video.videoWidth, video.videoHeight, region));
    if (cam.readings.length > SMOOTH_SAMPLES) cam.readings.shift();
    const m = average(cam.readings);
    const verdict = judge(m);
    if (!verdict) return;
    setStatus($("cam-status"), verdict === "good" ? "good" : "warn", MESSAGES[verdict]);
    // Poor light changes the button's look and label but never blocks the photo.
    const poor = verdict !== "good";
    captureBtn.classList.toggle("poor", poor);
    $("cam-capture-hint").hidden = !poor;
    captureBtn.setAttribute("aria-label", poor ? "Take photo anyway (lighting could be better)" : "Take photo");
    renderDebug(m, region);
  }

  function renderDebug(m, region) {
    if (!DEBUG) return;
    const mirrored = video.classList.contains("mirrored");
    // Report sides as they appear on screen.
    const [l, r] = mirrored ? [m.right, m.left] : [m.left, m.right];
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    debugText.textContent = [
      `mean   ${m.mean.toFixed(1)}  (min ${MIN_BRIGHTNESS}, max ${MAX_BRIGHTNESS})`,
      `left   ${l.toFixed(1)}  right ${r.toFixed(1)}`,
      `side   ${sideDiff(m).toFixed(2)}  (max ${MAX_SIDE_DIFF})`,
      `clip   ${pct(m.high + m.low)}  white ${pct(m.high)} black ${pct(m.low)}  (max ${pct(MAX_CLIPPED_FRACTION)})`,
      `verdict ${judge(m)}  ·  ${cam.readings.length}/${SMOOTH_SAMPLES} samples  ·  ${video.videoWidth}x${video.videoHeight}`,
    ].join("\n");
    // The scaled-down frame that was measured, with the measured oval drawn on it.
    debugCanvas.width = sampleCanvas.width;
    debugCanvas.height = sampleCanvas.height;
    const ctx = debugCanvas.getContext("2d");
    ctx.drawImage(sampleCanvas, 0, 0);
    const k = sampleCanvas.width / video.videoWidth;
    ctx.strokeStyle = "#EFD58F";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(region.cx * k, region.cy * k, region.rx * k, region.ry * k, 0, 0, Math.PI * 2);
    ctx.stroke();
    debugCanvas.classList.toggle("mirrored", mirrored);
  }

  // --- Capture and review --------------------------------------------------------------
  function capture() {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    // Native resolution, un-mirrored, no filters, no overlay: raw colour feeds the skin-tone analysis.
    canvas.getContext("2d").drawImage(video, 0, 0, w, h);
    const region = ovalInVideo();
    stopStream();
    canvas.toBlob((blob) => {
      if (!blob) return showError(MESSAGES.captureFailed, true);
      // The same check, on the photo itself.
      const verdict = judge(measure(canvas, w, h, region));
      canvas.width = canvas.height = 0;
      showReview(blob, verdict);
    }, "image/jpeg", 0.92);
  }

  function showReview(blob, verdict) {
    revokePhoto();
    cam.blob = blob;
    cam.photoUrl = URL.createObjectURL(blob);
    photo.src = cam.photoUrl;
    show("review");
    const good = verdict === "good" || !verdict;
    setStatus($("cam-review-status"), good ? "good" : "warn", good ? MESSAGES.good : MESSAGES[verdict]);
    const use = $("cam-use");
    const retake = $("cam-retake");
    use.textContent = good ? "Use this photo" : "Use anyway";
    // Good light: using the photo is the main action. Poor light: retaking is.
    use.classList.toggle("secondary", !good);
    retake.classList.toggle("secondary", good);
    const primary = good ? use : retake;
    $("cam-review-actions").replaceChildren(primary, good ? retake : use);
    primary.focus();
  }

  function revokePhoto() {
    if (cam.photoUrl) URL.revokeObjectURL(cam.photoUrl);
    cam.photoUrl = null;
    photo.removeAttribute("src");
  }

  function accept() {
    const blob = cam.blob;
    const onAccept = cam.onAccept;
    cam.blob = null;
    close();
    if (blob) onAccept?.(blob);
  }

  function retake() {
    revokePhoto();
    cam.blob = null;
    start();
  }

  function chooseLibrary() {
    const onLibrary = cam.onLibrary;
    close();
    onLibrary?.(); // still inside the click, so the browser lets it open the file picker
  }

  $("cam-start").addEventListener("click", start);
  $("cam-retry").addEventListener("click", start);
  $("cam-resume").addEventListener("click", start);
  $("cam-close").addEventListener("click", close);
  for (const btn of dialog.querySelectorAll("[data-cam-library]")) btn.addEventListener("click", chooseLibrary);
  captureBtn.addEventListener("click", capture);
  flipBtn.addEventListener("click", () => {
    cam.facing = cam.facing === "user" ? "environment" : "user";
    start();
  });
  $("cam-use").addEventListener("click", accept);
  $("cam-retake").addEventListener("click", retake);
  // A click on the backdrop (the dialog element itself, outside its content) closes it.
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) close();
  });

  window.DermalisCamera = { supported, insecure, open };
})();
