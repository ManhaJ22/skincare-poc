"""
Climate lookup via Open-Meteo (no API key): city or postal code -> rounded coordinates -> recent weather.

Privacy: the browser only talks to our backend; this module only sends the search text and
coordinates rounded to 2 decimals to Open-Meteo. Nothing here logs places or coordinates.
"""

from datetime import date
from statistics import mean

import httpx

from models.schemas import EnvironmentLookupResponse, EnvironmentProfile
from services import environment_rules

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 6
RECENT_DAYS = 14  # the "recent average" window


class EnvironmentLookupError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason  # "not_found" or "unavailable"


def _get(url: str, params: dict) -> dict:
    try:
        response = httpx.get(url, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as e:
        raise EnvironmentLookupError("unavailable") from e


def geocode(query: str) -> tuple[str, float, float]:
    """Best match for a city or postal code: (place name, lat, lon), coordinates rounded to 2 decimals."""
    data = _get(GEOCODE_URL, {"name": query.strip(), "count": 1, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        raise EnvironmentLookupError("not_found")
    top = results[0]
    place = ", ".join(dict.fromkeys(p for p in (top.get("name"), top.get("admin1"), top.get("country")) if p))
    return place, round(top["latitude"], 2), round(top["longitude"], 2)


def climate(lat: float, lon: float) -> dict:
    """Recent averages (temperature, humidity, dew point) and today's max UV index."""
    data = _get(FORECAST_URL, {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,dew_point_2m_mean,uv_index_max",
        "past_days": RECENT_DAYS,
        "forecast_days": 1,
        "timezone": "auto",
    })
    daily = data.get("daily") or {}

    def recent(key: str) -> list[float]:
        return [v for v in daily.get(key) or [] if v is not None]

    temps, humidity, dew = recent("temperature_2m_mean"), recent("relative_humidity_2m_mean"), recent("dew_point_2m_mean")
    uv = recent("uv_index_max")
    if not (temps and humidity and dew and uv):
        raise EnvironmentLookupError("unavailable")
    return {
        "avg_temp_c": round(mean(temps), 1),
        "avg_humidity": round(mean(humidity)),
        "dew_point_c": round(mean(dew), 1),
        "uv_index_max": round(uv[-1], 1),  # the last day is today
    }


def lookup(
    query: str | None = None, lat: float | None = None, lon: float | None = None, today: date | None = None
) -> EnvironmentLookupResponse:
    """Climate for a city/postal code, or for saved coordinates (rounded again, just in case)."""
    if query:
        place, lat, lon = geocode(query)
    else:
        place, lat, lon = None, round(lat, 2), round(lon, 2)
    weather = climate(lat, lon)
    label = environment_rules.climate_label(weather["avg_temp_c"], weather["dew_point_c"], weather["avg_humidity"])
    current_season = environment_rules.season(lat, today or date.today())
    profile = EnvironmentProfile(
        climate_label=label, avg_humidity=weather["avg_humidity"], uv_index_max=weather["uv_index_max"]
    )
    return EnvironmentLookupResponse(
        place=place,
        lat=lat,
        lon=lon,
        season=current_season,
        climate_label=label,
        avg_temp_c=weather["avg_temp_c"],
        avg_humidity=weather["avg_humidity"],
        uv_index_max=weather["uv_index_max"],
        summary=environment_rules.summary(profile),
    )
