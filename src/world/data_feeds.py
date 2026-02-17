"""World data feeds — real-world Earth observation data.

Pulls live data from external sources to seed the world's systems.
Currently supports:
- Solar activity from NOAA Space Weather Prediction Center

All sources are free, no API keys required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# NOAA Space Weather Prediction Center — free, no key
SWPC_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
SWPC_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"


def _kp_category(kp: float) -> str:
    """Map Kp index to geomagnetic activity category."""
    if kp < 2:
        return "quiet"
    elif kp < 4:
        return "unsettled"
    elif kp < 5:
        return "active"
    elif kp < 7:
        return "storm"
    else:
        return "severe storm"


def _xray_class(flux: float | None) -> str | None:
    """Map GOES X-ray flux (W/m^2) to solar flare class."""
    if flux is None:
        return None
    if flux < 1e-7:
        return "A"
    elif flux < 1e-6:
        return "B"
    elif flux < 1e-5:
        return "C"
    elif flux < 1e-4:
        return "M"
    else:
        return "X"


@dataclass
class SolarActivity:
    """Current solar activity state from NOAA SWPC."""
    kp_index: float | None = None
    kp_category: str | None = None
    solar_wind_speed_kms: float | None = None
    solar_wind_density: float | None = None
    solar_wind_temperature_k: float | None = None
    bz_gsm_nt: float | None = None
    bt_nt: float | None = None
    xray_flux: float | None = None
    xray_class: str | None = None
    last_updated: str | None = None

    def to_dict(self) -> dict:
        return {
            "kp_index": self.kp_index,
            "kp_category": self.kp_category,
            "solar_wind_speed_kms": self.solar_wind_speed_kms,
            "solar_wind_density": self.solar_wind_density,
            "solar_wind_temperature_k": self.solar_wind_temperature_k,
            "bz_gsm_nt": self.bz_gsm_nt,
            "bt_nt": self.bt_nt,
            "xray_flux": self.xray_flux,
            "xray_class": self.xray_class,
            "last_updated": self.last_updated,
        }


@dataclass
class DataFeeds:
    """
    World data feeds — real-world Earth observation data.

    Fetches live space weather from NOAA SWPC on a periodic schedule.
    Data available to agents and the desktop dashboard.
    """
    _solar: SolarActivity | None = None
    _last_refresh: datetime | None = None
    _refresh_count: int = 0
    _refresh_failures: int = 0

    @property
    def solar_activity(self) -> SolarActivity | None:
        return self._solar

    @property
    def last_refresh(self) -> datetime | None:
        return self._last_refresh

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    async def refresh(self) -> bool:
        """Refresh all data feeds. Returns True if any data updated."""
        updated = False
        try:
            solar = await self._fetch_solar()
            if solar:
                self._solar = solar
                self._last_refresh = datetime.now(timezone.utc)
                self._refresh_count += 1
                updated = True
                logger.info(
                    f"Data feeds refreshed: Kp={solar.kp_index}, "
                    f"solar wind={solar.solar_wind_speed_kms} km/s"
                )
        except Exception as e:
            self._refresh_failures += 1
            logger.warning(f"Data feeds refresh failed: {e}")
        return updated

    async def _fetch_solar(self) -> SolarActivity | None:
        """Fetch solar activity from NOAA SWPC APIs."""
        state = SolarActivity()

        async with httpx.AsyncClient(timeout=15) as client:
            # Kp index — planetary geomagnetic activity
            try:
                resp = await client.get(SWPC_KP_URL)
                resp.raise_for_status()
                data = resp.json()
                # Format: array of arrays, first row is header
                # Rows: [time_tag, Kp, a_running, station_count]
                if len(data) > 1:
                    latest = data[-1]
                    kp = float(latest[1])
                    state.kp_index = round(kp, 2)
                    state.kp_category = _kp_category(kp)
            except Exception as e:
                logger.debug(f"SWPC Kp fetch failed: {e}")

            # Solar wind plasma — speed, density, temperature
            try:
                resp = await client.get(SWPC_PLASMA_URL)
                resp.raise_for_status()
                data = resp.json()
                # Format: array of arrays, first row is header
                # Rows: [time_tag, density, speed, temperature]
                if len(data) > 1:
                    latest = data[-1]
                    if latest[2] is not None:
                        state.solar_wind_speed_kms = round(float(latest[2]), 1)
                    if latest[1] is not None:
                        state.solar_wind_density = round(float(latest[1]), 2)
                    if latest[3] is not None:
                        state.solar_wind_temperature_k = round(float(latest[3]), 0)
            except Exception as e:
                logger.debug(f"SWPC plasma fetch failed: {e}")

            # Interplanetary magnetic field — Bz and total B
            try:
                resp = await client.get(SWPC_MAG_URL)
                resp.raise_for_status()
                data = resp.json()
                # Format: array of arrays, first row is header
                # Rows: [time_tag, bx_gsm, by_gsm, bz_gsm, lon_gsm, lat_gsm, bt]
                if len(data) > 1:
                    latest = data[-1]
                    if latest[3] is not None:
                        state.bz_gsm_nt = round(float(latest[3]), 2)
                    if latest[6] is not None:
                        state.bt_nt = round(float(latest[6]), 2)
            except Exception as e:
                logger.debug(f"SWPC mag fetch failed: {e}")

            # X-ray flux — GOES satellite solar flare monitoring
            try:
                resp = await client.get(SWPC_XRAY_URL)
                resp.raise_for_status()
                data = resp.json()
                # Format: array of objects with time_tag, satellite, flux, ...
                if data:
                    latest = data[-1]
                    flux = latest.get("flux")
                    if flux is not None:
                        state.xray_flux = flux
                        state.xray_class = _xray_class(flux)
            except Exception as e:
                logger.debug(f"SWPC X-ray fetch failed: {e}")

        state.last_updated = datetime.now(timezone.utc).isoformat()

        # Return None if we got nothing useful
        if state.kp_index is None and state.solar_wind_speed_kms is None:
            return None

        return state
