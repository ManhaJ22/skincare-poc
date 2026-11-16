"""Progress tracking storage: plain files on disk, for a single demo user (no auth).

backend/data/sessions/<session_id>.json         one ProgressSession per completed consultation
backend/data/photos/<session_id>/<photo_id>.jpg  progress photos (or .png / .webp)

Every id is validated before it becomes part of a path, so a request can't reach outside DATA_DIR.
"""
import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from models.schemas import PhotoMeta, ProgressSession

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_PHOTO_BYTES = 8 * 1024 * 1024

# Magic bytes -> file extension. The client's content type is never trusted.
IMAGE_SIGNATURES = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
]


class InvalidId(ValueError):
    pass


class InvalidPhoto(ValueError):
    pass


def _session_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise InvalidId("Not a valid session id.") from None


def _photo_id(value: str) -> str:
    try:
        parsed = uuid.UUID(hex=value)
    except (ValueError, TypeError, AttributeError):
        raise InvalidId("Not a valid photo id.") from None
    if parsed.hex != value:
        raise InvalidId("Not a valid photo id.")
    return value


def _sessions_dir() -> Path:
    return DATA_DIR / "sessions"


def _session_file(session_id: str) -> Path:
    return _sessions_dir() / f"{_session_id(session_id)}.json"


def _photos_dir(session_id: str) -> Path:
    return DATA_DIR / "photos" / _session_id(session_id)


def _write(session: ProgressSession) -> ProgressSession:
    path = _session_file(session.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    return session


def get_session(session_id: str) -> ProgressSession | None:
    path = _session_file(session_id)
    if not path.exists():
        return None
    return ProgressSession.model_validate_json(path.read_text(encoding="utf-8"))


def latest_session() -> ProgressSession | None:
    sessions = []
    for path in _sessions_dir().glob("*.json") if _sessions_dir().exists() else []:
        try:
            sessions.append(ProgressSession.model_validate_json(path.read_text(encoding="utf-8")))
        except ValueError:
            continue  # a half-written or hand-edited file; skip it
    return max(sessions, key=lambda s: s.created_at or "", default=None)


def save_session(session: ProgressSession) -> ProgressSession:
    """Upsert. Photos and a user-chosen start date survive re-saves of the same session."""
    session.session_id = _session_id(session.session_id)
    existing = get_session(session.session_id)
    if existing:
        session.photos = existing.photos
        session.start_date = existing.start_date or session.start_date
        session.created_at = existing.created_at or session.created_at
    session.created_at = session.created_at or datetime.now(timezone.utc).isoformat()
    session.start_date = session.start_date or session.created_at[:10]
    return _write(session)


def set_start_date(session_id: str, start_date: str) -> ProgressSession | None:
    session = get_session(session_id)
    if session is None:
        return None
    date.fromisoformat(start_date)  # raises ValueError on e.g. 2026-02-31
    session.start_date = start_date
    return _write(session)


def image_extension(data: bytes) -> str:
    for signature, ext in IMAGE_SIGNATURES:
        if data.startswith(signature):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise InvalidPhoto("Only JPEG, PNG or WEBP photos are supported.")


def add_photo(session_id: str, data: bytes, taken_at: str | None = None, note: str = "") -> PhotoMeta | None:
    session = get_session(session_id)
    if session is None:
        return None
    if len(data) > MAX_PHOTO_BYTES:
        raise InvalidPhoto("That photo is too large (8 MB max).")
    ext = image_extension(data)
    taken = taken_at or date.today().isoformat()
    try:
        date.fromisoformat(taken)
    except ValueError:
        raise InvalidPhoto("taken_at must be a date like 2026-11-16.") from None
    photo = PhotoMeta(photo_id=uuid.uuid4().hex, taken_at=taken, note=note[:200])
    folder = _photos_dir(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{photo.photo_id}.{ext}").write_bytes(data)
    session.photos.append(photo)
    session.photos.sort(key=lambda p: p.taken_at)
    _write(session)
    return photo


def photo_path(session_id: str, photo_id: str) -> Path | None:
    photo_id = _photo_id(photo_id)
    folder = _photos_dir(session_id)
    return next((p for p in folder.glob(f"{photo_id}.*")), None) if folder.exists() else None


def delete_photo(session_id: str, photo_id: str) -> bool:
    session = get_session(session_id)
    path = photo_path(session_id, photo_id)
    if session is None or path is None:
        return False
    path.unlink(missing_ok=True)
    session.photos = [p for p in session.photos if p.photo_id != photo_id]
    _write(session)
    return True
