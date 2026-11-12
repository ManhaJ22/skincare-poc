"""
UTKFace test harness: run faces from every UTKFace race group through the local backend and
render one HTML dashboard of what came back.

  1. Reads UTKFace images from a local flat folder (--utk-dir; filenames are
     age_gender_race_timestamp.jpg, race 0=White 1=Black 2=Asian 3=Indian 4=Other) and screens
     them in random order with a MediaPipe face pre-filter (scripts/face_prefilter.py): exactly one
     face, at least 15% of the image, yaw within +/-15 and pitch within +/-10 degrees. It keeps
     sampling until each race group has N passing faces, copied to utk_images/{index}_{race}_{age}.jpg
     (label readable from the filename). Every checked image is logged to utk_images/prefilter_log.csv
     with its rejection reason, and a checked-vs-accepted summary is printed.
  2. For every image: POST /analyze with "prompt": "fairface" (the batch-testing vision prompt in
     backend/services/vision_service.py; --prompt app uses the app's prompt instead), merges a randomly
     picked fake persona into that profile, then POST /recommendations.
  3. Writes results.json and a self-contained results_dashboard.html.

Start the backend first (cd backend && uvicorn main:app), then:
    pip install -r scripts/requirements-harness.txt
    python scripts/fairface_harness.py                       # 15 per group, 75 images
    python scripts/fairface_harness.py --per-group 2         # quick smoke run
    python scripts/fairface_harness.py --utk-dir path/to/utk # images somewhere else
"""

import argparse
import asyncio
import base64
import csv
import html
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm

DEFAULT_UTK_DIR = str(Path(__file__).resolve().parent.parent / "test-images" / "part2")
RACE_CODES = {"0": "White", "1": "Black", "2": "Asian", "3": "Indian", "4": "Other"}
RACES = list(RACE_CODES.values())
MIN_ADULT_AGE = 10  # younger faces skipped by default: an adult skincare routine isn't meaningful for them

# age_gender_race_timestamp.jpg; the aligned/cropped UTKFace set adds ".chip" (….jpg.chip.jpg).
UTK_RE = re.compile(r"^(\d+)_([01])_([0-4])_(\d+)(?:\.jpg\.chip)?\.jpg$", re.IGNORECASE)

PERSONAS = [
    {"skin_type": "oily", "concerns": ["acne", "large pores"], "allergies": ["parabens"], "budget": 40, "fragrance_free": True},
    {"skin_type": "dry", "concerns": ["hyperpigmentation", "dullness"], "allergies": [], "budget": 75, "fragrance_free": False},
    {"skin_type": "combination", "concerns": ["PIH", "uneven texture"], "allergies": ["benzoyl peroxide"], "budget": 50, "fragrance_free": True},
]

# Monk Skin Tone scale (Google, 2022), tones 1-10.
MONK_HEX = {1: "#f6ede4", 2: "#f3e7db", 3: "#f7ead0", 4: "#eadaba", 5: "#d7bd96",
            6: "#a07e56", 7: "#825c43", 8: "#604134", 9: "#3a312a", 10: "#292420"}

FILENAME_RE = re.compile(r"^(\d+)_(.+)_([^_]+)\.jpg$")


# --- Dataset ---------------------------------------------------------------------------------
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def race_from_slug(race_slug: str) -> str:
    return {slug(r): r for r in RACES}.get(race_slug, race_slug)


def load_utk_images(folder, max_per_race: int | None = 15, seed: int | None = None) -> list[dict]:
    """
    UTKFace images in a flat folder, grouped by the race code in the filename and randomly sampled
    up to max_per_race per group (None: every image in the group, shuffled). Files whose name isn't
    age_gender_race_timestamp.jpg are skipped. Returns [{"path", "race", "age"}, ...].
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for path in sorted(Path(folder).glob("*.jpg")):
        match = UTK_RE.match(path.name)
        if not match:
            continue
        age, _gender, race_code, _timestamp = match.groups()
        race = RACE_CODES[race_code]
        groups[race].append({"path": str(path), "race": race, "age": age})
    images = []
    for race in RACES:
        group = groups.get(race, [])
        k = len(group) if max_per_race is None else min(max_per_race, len(group))
        images.extend(rng.sample(group, k))
    return images


def interleave_by_race(images: list[dict]) -> list[dict]:
    """Round-robin across races, so every group is screened at the same pace."""
    by_race = defaultdict(list)
    for img in images:
        by_race[img["race"]].append(img)
    queues = [by_race[r] for r in RACES if by_race[r]]
    out = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


LOG_FIELDS = ["source_file", "race", "age", "status", "reason", "yaw", "pitch", "face_area", "n_faces", "saved_as"]


def _round(value, digits=1):
    return "" if value is None else round(value, digits)


def collect_images(utk_dir: Path, image_dir: Path, per_group: int, seed: int, include_children: bool,
                   filter_settings: dict, save_rejected: bool) -> dict:
    """
    Screen UTKFace images in random order and keep sampling until every race group has per_group
    images that pass the face pre-filter. Every checked image is logged to prefilter_log.csv.
    Returns the pre-filter stats.
    """
    from PIL import Image
    from face_prefilter import FacePrefilter

    candidates = load_utk_images(utk_dir, max_per_race=None, seed=seed)
    if not candidates:
        raise SystemExit(f"No UTKFace images (age_gender_race_timestamp.jpg) found in {utk_dir}")
    print(f"Found {len(candidates)} UTKFace images in {utk_dir}")
    accepted, checked = Counter(), Counter()
    reasons = defaultdict(Counter)  # race -> reason -> count
    target = per_group * len(RACES)
    index = 0
    with (FacePrefilter(**filter_settings) as prefilter,
          open(image_dir / "prefilter_log.csv", "w", newline="", encoding="utf-8") as log_file,
          tqdm(total=target, desc="Filtering faces", unit="img") as bar):
        log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log.writeheader()
        for item in interleave_by_race(candidates):
            race, age = item["race"], item["age"]
            if accepted[race] >= per_group or (int(age) < MIN_ADULT_AGE and not include_children):
                continue
            source = Path(item["path"])
            try:
                with Image.open(source) as img:
                    image = img.convert("RGB")
            except OSError as e:
                print(f"Skipping unreadable {source.name}: {e}", file=sys.stderr)
                continue
            result = prefilter.check(image)
            checked[race] += 1
            entry = {"source_file": source.name, "race": race, "age": age, "reason": result.reason,
                     "yaw": _round(result.yaw), "pitch": _round(result.pitch),
                     "face_area": _round(result.face_area, 3), "n_faces": result.n_faces, "saved_as": ""}
            if result.ok:
                name = f"{index:03d}_{slug(race)}_{age}.jpg"
                shutil.copyfile(source, image_dir / name)
                entry.update(status="accepted", saved_as=name)
                accepted[race] += 1
                index += 1
                bar.update(1)
            else:
                entry["status"] = "rejected"
                reasons[race][result.reason] += 1
                if save_rejected:
                    rejected_dir = image_dir / "rejected" / result.reason
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, rejected_dir / source.name)
                    entry["saved_as"] = f"rejected/{result.reason}/{source.name}"
                bar.set_postfix(rejected=sum(checked.values()) - sum(accepted.values()), refresh=False)
            log.writerow(entry)
            if index >= target:
                break
    short = {r: per_group - accepted[r] for r in RACES if accepted[r] < per_group}
    if short:
        print(f"Warning: ran out of UTKFace images before filling every group (missing: {short})", file=sys.stderr)
    return {"checked": dict(checked), "accepted": dict(accepted),
            "reasons": {race: dict(c) for race, c in reasons.items()}}


def print_prefilter_summary(stats: dict) -> None:
    from face_prefilter import REASONS

    checked, accepted, reasons = stats["checked"], stats["accepted"], stats["reasons"]
    abbrev = {"no_face": "no_face", "multiple_faces": "multi", "face_too_small": "small",
              "yaw_out_of_range": "yaw", "pitch_out_of_range": "pitch"}
    print("\nFace pre-filter summary")
    print(f"{'Race':<18}{'checked':>8}{'accepted':>9}{'rate':>6}  " + "".join(f"{abbrev[r]:>8}" for r in REASONS))
    for race in RACES:
        c, a = checked.get(race, 0), accepted.get(race, 0)
        rate = f"{a / c:.0%}" if c else "-"
        cells = "".join(f"{reasons.get(race, {}).get(r, 0):>8}" for r in REASONS)
        print(f"{race.replace('_', '/'):<18}{c:>8}{a:>9}{rate:>6}  {cells}")
    total_c, total_a = sum(checked.values()), sum(accepted.values())
    totals = Counter()
    for per_race in reasons.values():
        totals.update(per_race)
    rate = f"{total_a / total_c:.0%}" if total_c else "-"
    print(f"{'Total':<18}{total_c:>8}{total_a:>9}{rate:>6}  " + "".join(f"{totals.get(r, 0):>8}" for r in REASONS))
    print("Rejections: " + (", ".join(f"{r} {n}" for r, n in totals.most_common()) or "none") + "\n")


def list_images(image_dir: Path) -> list[dict]:
    items = []
    for path in sorted(image_dir.glob("*.jpg")):
        match = FILENAME_RE.match(path.name)
        if match:
            index, race_slug, age = match.groups()
            items.append({"index": int(index), "file": path.name, "path": path,
                          "race": race_from_slug(race_slug), "age": age})
    return items


def select_per_group(items: list[dict], per_group: int) -> list[dict]:
    """The first per_group images of each race, so a smaller --per-group reuses an earlier, bigger download."""
    counts = Counter()
    picked = []
    for item in items:
        if counts[item["race"]] < per_group:
            counts[item["race"]] += 1
            picked.append(item)
    return picked


# --- Backend calls ---------------------------------------------------------------------------
def build_profile(vision: dict, persona: dict) -> dict:
    """Vision result + persona, mapped onto the backend's SkinProfile fields."""
    return {
        "monk_tone": vision.get("monk_tone", 0),
        "observed_concerns": vision.get("observed_concerns", []),
        "skin_type": persona["skin_type"],
        "confirmed_concerns": persona["concerns"],
        "allergies": persona["allergies"],
        "budget_monthly": persona["budget"],
        "fragrance_free": persona["fragrance_free"],
        # The backend treats product history as answered once either list is present.
        "products_liked": [],
        "products_disliked": [],
    }


def _detail(response: httpx.Response) -> str:
    try:
        return f"{response.status_code}: {response.json().get('detail', response.text)}"
    except ValueError:
        return f"{response.status_code}: {response.text[:300]}"


async def process(client: httpx.AsyncClient, sem: asyncio.Semaphore, item: dict, seed: int, prompt: str) -> dict:
    persona = random.Random(seed + item["index"]).choice(PERSONAS)
    result = {k: item[k] for k in ("index", "file", "race", "age")}
    result.update(prompt=prompt, persona=persona, monk_tone=None, observed_concerns=[], vision_skin_type="",
                  vision_confidence=None, recommendations=[], budget=None, error=None)
    async with sem:
        try:
            b64 = base64.b64encode(item["path"].read_bytes()).decode()
            res = await client.post("/analyze", json={"image_base64": f"data:image/jpeg;base64,{b64}", "prompt": prompt})
            if res.status_code != 200:
                result["error"] = f"/analyze {_detail(res)}"
                return result
            body = res.json()
            vision = body["skin_profile"]
            result["vision_confidence"] = body.get("vision_confidence")
            result["vision_skin_type"] = vision.get("skin_type", "")
            result["monk_tone"] = vision.get("monk_tone")
            result["observed_concerns"] = vision.get("observed_concerns", [])
            if not result["monk_tone"]:
                result["error"] = "Photo rejected by vision (monk_tone 0: too low quality to analyze)"
                return result

            res = await client.post("/recommendations", json={"skin_profile": build_profile(vision, persona)})
            if res.status_code != 200:
                result["error"] = f"/recommendations {_detail(res)}"
                return result
            body = res.json()
            result["recommendations"] = body["recommendations"]
            result["budget"] = body.get("budget")
        except httpx.HTTPError as e:
            result["error"] = f"{type(e).__name__}: {e}"
    return result


async def run_all(items: list[dict], base_url: str, concurrency: int, seed: int, prompt: str) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=180) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except httpx.HTTPError as e:
            raise SystemExit(f"Backend not reachable at {base_url} ({e}). Start it with: cd backend && uvicorn main:app")
        # All tasks go into one gather; the semaphore caps how many hit the backend (and OpenAI) at once.
        results = await tqdm.gather(*(process(client, sem, item, seed, prompt) for item in items),
                                    desc="Analyzing", unit="img")
    return sorted(results, key=lambda r: r["index"])


# --- Dashboard -------------------------------------------------------------------------------
CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f6f4f1;color:#2b2622}
header{padding:20px 28px;background:#fff;border-bottom:1px solid #e6e0d8;position:sticky;top:0;z-index:2}
h1{margin:0 0 6px;font-size:20px} .sub{color:#7a6f66;font-size:13px}
.controls{margin-top:10px;display:flex;gap:12px;align-items:center;font-size:13px}
select{padding:4px 8px;border-radius:6px;border:1px solid #d6cdc2;font:inherit}
.summary{padding:18px 28px;display:flex;gap:28px;flex-wrap:wrap}
table{border-collapse:collapse;font-size:12px;background:#fff;border-radius:8px;overflow:hidden}
th,td{padding:5px 8px;border-bottom:1px solid #eee;text-align:center} th{background:#faf7f3;font-weight:600}
td.race{text-align:left;font-weight:600} td.zero{color:#ccc}
.sw{display:inline-block;width:14px;height:14px;border-radius:4px;border:1px solid rgba(0,0,0,.15);vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px;padding:0 28px 40px}
.card{background:#fff;border-radius:12px;border:1px solid #e6e0d8;overflow:hidden;display:flex;flex-direction:column}
.card.err{border-color:#e8a29a}
.top{display:flex;gap:14px;padding:14px}
.top img{width:120px;height:120px;object-fit:cover;border-radius:8px;flex:none}
.meta{font-size:13px;line-height:1.5;min-width:0}
.label{font-weight:700;font-size:15px} .file{color:#9a8f86;font-size:11px;word-break:break-all}
.tone{display:flex;align-items:center;gap:8px;margin:6px 0}
.tone .big{width:34px;height:34px;border-radius:8px;border:1px solid rgba(0,0,0,.15)}
.chip{display:inline-block;background:#f1ebe4;border-radius:999px;padding:1px 8px;margin:2px 3px 0 0;font-size:11px}
.chip.p{background:#e7eef8}
.vision{font-size:12px;color:#6d625a;margin:-2px 0 4px}
.persona{padding:0 14px 10px;font-size:12px;color:#5d534b}
.recs{border-top:1px solid #f0ebe5;padding:10px 14px 14px;font-size:12px}
.step{margin-top:8px} .step:first-child{margin-top:0}
.step h4{margin:0 0 3px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#8a7d72}
.rec{display:flex;justify-content:space-between;gap:8px;padding:2px 0}
.rec.alt{color:#a79d94} .rec.out{color:#b9b0a8;text-decoration:line-through}
.tag{font-size:10px;border-radius:4px;padding:0 5px;background:#f4efe9;color:#7a6f66;margin-left:4px;white-space:nowrap}
.budget{margin-top:8px;color:#7a6f66}
.error{color:#b3261e;background:#fdecea;padding:8px 14px;font-size:12px}
"""

JS = """
const sel=document.getElementById('race'), toneSel=document.getElementById('tone'), status=document.getElementById('status');
function apply(){let n=0;document.querySelectorAll('.card').forEach(c=>{
 const show=(!sel.value||c.dataset.race===sel.value)&&(!toneSel.value||c.dataset.tone===toneSel.value)
   &&(status.value!=='err'||c.classList.contains('err'));
 c.style.display=show?'':'none';n+=show;});document.getElementById('count').textContent=n+' shown';}
[sel,toneSel,status].forEach(e=>e.addEventListener('change',apply));apply();
"""


def esc(value) -> str:
    return html.escape(str(value))


def swatch(tone, big=False) -> str:
    color = MONK_HEX.get(tone)
    return f'<span class="{"big" if big else "sw"}" style="background:{color}" title="Monk {tone}"></span>' if color else ""


def render_card(r: dict, image_dir: Path) -> str:
    img = base64.b64encode((image_dir / r["file"]).read_bytes()).decode()
    tone = r["monk_tone"]
    tone_html = (f'<div class="tone">{swatch(tone, big=True)}<span><b>Monk {tone}</b> / 10</span></div>'
                 if tone else '<div class="tone"><span>Monk tone: <b>n/a</b></span></div>')
    concerns = "".join(f'<span class="chip">{esc(c)}</span>' for c in r["observed_concerns"]) or "<i>none observed</i>"
    vision_bits = [f"confidence: <b>{esc(r['vision_confidence'])}</b>" if r.get("vision_confidence") else "",
                   f"looks {esc(r['vision_skin_type'])}" if r.get("vision_skin_type") else ""]
    vision_line = " · ".join(b for b in vision_bits if b)
    vision_html = f'<div class="vision">{vision_line}</div>' if vision_line else ""
    p = r["persona"]
    persona = (f'<b>Persona:</b> {esc(p["skin_type"])} skin · budget ${p["budget"]} · '
               f'{"fragrance-free" if p["fragrance_free"] else "fragrance OK"} · '
               f'allergies: {esc(", ".join(p["allergies"]) or "none")}<br>'
               + "".join(f'<span class="chip p">{esc(c)}</span>' for c in p["concerns"]))

    steps = defaultdict(list)
    for rec in r["recommendations"]:
        steps[rec["routine_step"]].append(rec)
    rec_html = ""
    for step, recs in steps.items():
        rows = ""
        for rec in recs:
            cls = "out" if rec.get("budget_left_out") else "alt" if rec.get("is_alternative") else ""
            tags = "".join(f'<span class="tag">{esc(t)}</span>' for t in
                           [rec.get("routine_time"), "alt" if rec.get("is_alternative") else "",
                            "over budget" if rec.get("budget_left_out") else ""] if t)
            rows += (f'<div class="rec {cls}"><span>{esc(rec["brand"])} — {esc(rec["product_name"])}</span>'
                     f'<span>{esc(rec.get("price") or "")}{tags}</span></div>')
        rec_html += f'<div class="step"><h4>{esc(step)}</h4>{rows}</div>'
    b = r.get("budget")
    if b:
        rec_html += (f'<div class="budget">Routine total ${b["total"]:.2f} vs budget '
                     f'${b["budget"] if b["budget"] is not None else "—"} ({esc(b["status"])})</div>')
    recs_block = f'<div class="recs">{rec_html}</div>' if rec_html else ""
    error = f'<div class="error">{esc(r["error"])}</div>' if r["error"] else ""

    return f"""<div class="card{' err' if r['error'] else ''}" data-race="{esc(r['race'])}" data-tone="{tone or ''}">
<div class="top"><img src="data:image/jpeg;base64,{img}" alt="{esc(r['race'])}">
<div class="meta"><div class="label">{esc(r['race'].replace('_', ' / '))} · {esc(r['age'])}</div>
<div class="file">{esc(r['file'])}</div>{tone_html}{vision_html}<div><b>Observed:</b> {concerns}</div></div></div>
<div class="persona">{persona}</div>{recs_block}{error}</div>"""


def render_summary(results: list[dict]) -> str:
    by_race = defaultdict(Counter)
    for r in results:
        by_race[r["race"]][r["monk_tone"] or 0] += 1
    header = "<th>Race</th><th>n</th><th>avg</th>" + "".join(
        f"<th>{swatch(t)}<br>{t}</th>" for t in range(1, 11)) + "<th>n/a</th>"
    rows = ""
    for race in RACES:
        c = by_race.get(race)
        if not c:
            continue
        toned = [(t, n) for t, n in c.items() if t]
        avg = sum(t * n for t, n in toned) / sum(n for _, n in toned) if toned else 0
        cells = "".join(f'<td class="{"zero" if not c[t] else ""}">{c[t]}</td>' for t in range(1, 11))
        rows += (f'<tr><td class="race">{esc(race.replace("_", " / "))}</td><td>{sum(c.values())}</td>'
                 f'<td>{avg:.1f}</td>{cells}<td class="{"zero" if not c[0] else ""}">{c[0]}</td></tr>')
    levels = ["high", "medium", "low", "none"]
    conf_rows = ""
    for race in RACES:
        rs = [r for r in results if r["race"] == race]
        if not rs:
            continue
        c = Counter(r.get("vision_confidence") or "n/a" for r in rs)
        conf_rows += (f'<tr><td class="race">{esc(race.replace("_", " / "))}</td>'
                      + "".join(f'<td class="{"zero" if not c[l] else ""}">{c[l]}</td>' for l in levels + ["n/a"])
                      + "</tr>")
    conf_table = ("<div><b>Vision confidence by race label</b><table><tr><th>Race</th>"
                  + "".join(f"<th>{l}</th>" for l in levels + ["n/a"]) + f"</tr>{conf_rows}</table></div>")
    errors = sum(1 for r in results if r["error"])
    no_recs = sum(1 for r in results if not r["error"] and not r["recommendations"])
    return (f'<div><b>Detected Monk tone by race label</b><table><tr>{header}</tr>{rows}</table></div>{conf_table}'
            f'<div style="font-size:13px;line-height:1.8"><b>{len(results)}</b> images<br>'
            f'<b>{errors}</b> errors / rejected photos<br><b>{no_recs}</b> with no recommendations</div>')


def render_dashboard(results: list[dict], image_dir: Path, out: Path) -> None:
    race_opts = "".join(f'<option value="{esc(r)}">{esc(r.replace("_", " / "))}</option>'
                        for r in RACES if any(x["race"] == r for x in results))
    tone_opts = "".join(f'<option value="{t}">Monk {t}</option>' for t in range(1, 11))
    cards = "\n".join(render_card(r, image_dir) for r in results)
    prompts = sorted({r.get("prompt") or "app" for r in results})
    prompt_label = {"fairface": "batch-testing prompt", "app": "app prompt (what real users get)"}
    prompt_text = " + ".join(prompt_label.get(x, x) for x in prompts)
    out.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>UTKFace harness results</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head>
<body><header><h1>UTKFace harness results</h1>
<div class="sub">Vision Monk tone and observed concerns for each UTKFace face, with an injected persona and the backend's recommended routine. Vision prompt: <b>{esc(prompt_text)}</b>.</div>
<div class="controls"><label>Race <select id="race"><option value="">All</option>{race_opts}</select></label>
<label>Tone <select id="tone"><option value="">All</option>{tone_opts}</select></label>
<label>Show <select id="status"><option value="">All</option><option value="err">Errors only</option></select></label>
<span id="count"></span></div></header>
<section class="summary">{render_summary(results)}</section>
<section class="grid">{cards}</section><script>{JS}</script></body></html>""", encoding="utf-8")


# --- Main ------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--utk-dir", default=DEFAULT_UTK_DIR, help="flat folder of UTKFace .jpg files")
    parser.add_argument("--per-group", type=int, default=15, help="images per race group (5 groups)")
    parser.add_argument("--concurrency", type=int, default=8, help="max requests in flight at once")
    parser.add_argument("--prompt", choices=["fairface", "app"], default="fairface",
                        help="vision prompt: 'fairface' (batch-testing prompt) or 'app' (what real users get)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="where utk_images/ and the outputs go")
    parser.add_argument("--refresh", action="store_true", help="delete and re-select the images")
    parser.add_argument("--include-children", action="store_true", help=f"keep faces younger than {MIN_ADULT_AGE}")
    parser.add_argument("--render-only", action="store_true", help="rebuild the HTML from results.json; no API calls")
    face = parser.add_argument_group("face pre-filter (runs locally before any API call)")
    face.add_argument("--max-yaw", type=float, default=15, help="max head turn left/right, degrees")
    face.add_argument("--max-pitch", type=float, default=10, help="max head tilt up/down, degrees")
    face.add_argument("--min-face-area", type=float, default=0.15, help="min face box as a fraction of the image")
    face.add_argument("--save-rejected", action="store_true", help="also copy rejected images to utk_images/rejected/{reason}/")
    args = parser.parse_args()
    filter_settings = {"max_yaw": args.max_yaw, "max_pitch": args.max_pitch, "min_face_area": args.min_face_area}
    utk_dir = Path(args.utk_dir).expanduser().resolve()

    image_dir = args.out_dir / "utk_images"
    results_json = args.out_dir / "results.json"
    dashboard = args.out_dir / "results_dashboard.html"

    if args.render_only:
        results = json.loads(results_json.read_text(encoding="utf-8"))
    else:
        image_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = image_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        items = select_per_group(list_images(image_dir), args.per_group)
        reusable = (not args.refresh and manifest.get("filter") == filter_settings
                    and manifest.get("utk_dir") == str(utk_dir)
                    and len(items) >= args.per_group * len(RACES))
        if reusable:
            print(f"Using existing pre-filtered images in {image_dir} (pass --refresh to re-select)")
            stats = manifest["prefilter"]
        else:
            if not utk_dir.is_dir():
                raise SystemExit(f"UTKFace folder not found: {utk_dir} (pass --utk-dir)")
            # Images from an older run were picked with other (or no) filter settings: start clean.
            for old in image_dir.glob("*.jpg"):
                old.unlink()
            shutil.rmtree(image_dir / "rejected", ignore_errors=True)
            manifest_path.unlink(missing_ok=True)
            stats = collect_images(utk_dir, image_dir, args.per_group, args.seed, args.include_children,
                                   filter_settings, args.save_rejected)
            manifest_path.write_text(json.dumps({"filter": filter_settings, "utk_dir": str(utk_dir),
                                                 "per_group": args.per_group,
                                                 "seed": args.seed, "prefilter": stats}, indent=2), encoding="utf-8")
            items = select_per_group(list_images(image_dir), args.per_group)
        print_prefilter_summary(stats)
        results = asyncio.run(run_all(items, args.base_url, args.concurrency, args.seed, args.prompt))
        results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    render_dashboard(results, image_dir, dashboard)
    errors = sum(1 for r in results if r["error"])
    print(f"Done: {len(results)} images, {errors} errors -> {dashboard.resolve()}")


if __name__ == "__main__":
    main()
