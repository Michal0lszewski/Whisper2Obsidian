"""
GroqRateLimiter – sliding-window guard to prevent 429 errors on Groq's free tier.

Tracks:
  - RPM  (requests per minute)  via sliding 60-second window
  - TPM  (tokens per minute)    via sliding 60-second window
  - RPD  (requests per day)     via daily counter reset at midnight

Usage:
    limiter = GroqRateLimiter()           # uses settings defaults
    await limiter.await_capacity(estimated_tokens=2000)
    response = client.chat(...)
    limiter.record_usage(actual_tokens=response.usage.total_tokens)
    print(limiter.usage_report())
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from whisper2obsidian.config import settings

logger = logging.getLogger(__name__)


@dataclass
class _TokenEvent:
    """A single API call recorded in the sliding window."""
    timestamp: float
    tokens: int


@dataclass
class GroqRateLimiter:
    """
    Thread-safe (asyncio) sliding-window rate limiter for Groq API.

    Defaults are read from settings but can be overridden per-instance.
    """

    rpm_limit: int = field(default_factory=lambda: settings.groq_rpm_limit)
    tpm_limit: int = field(default_factory=lambda: settings.groq_tpm_limit)
    rpd_limit: int = field(default_factory=lambda: settings.groq_rpd_limit)

    # Internal state
    _window: deque[_TokenEvent] = field(default_factory=deque, init=False, repr=False)
    _day_requests: int = field(default=0, init=False, repr=False)
    _day_date: str = field(default="", init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    # ── Public API ──────────────────────────────────────────────────────────

    async def await_capacity(self, estimated_tokens: int = 500) -> None:
        """
        Block (sleep) until there is capacity for `estimated_tokens` tokens
        within the current minute window and request budget for today.

        Call this BEFORE making a Groq API request.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                self._purge_old_events(now)
                self._check_daily_reset()

                rpm_ok = len(self._window) < self.rpm_limit
                tpm_ok = self._current_tpm() + estimated_tokens <= self.tpm_limit
                rpd_ok = self._day_requests < self.rpd_limit

                if rpm_ok and tpm_ok and rpd_ok:
                    # Reserve the slot optimistically
                    self._window.append(_TokenEvent(timestamp=now, tokens=estimated_tokens))
                    self._day_requests += 1
                    logger.debug(
                        "RateLimiter: capacity granted – rpm=%d/%d tpm=%d/%d rpd=%d/%d",
                        len(self._window),
                        self.rpm_limit,
                        self._current_tpm(),
                        self.tpm_limit,
                        self._day_requests,
                        self.rpd_limit,
                    )
                    return

                # Calculate how long to sleep
                sleep_s = self._seconds_until_slot(now, estimated_tokens, rpm_ok, tpm_ok)
                logger.info(
                    "RateLimiter: sleeping %.1fs – rpm_ok=%s tpm_ok=%s rpd_ok=%s",
                    sleep_s,
                    rpm_ok,
                    tpm_ok,
                    rpd_ok,
                )
                await asyncio.sleep(sleep_s)

    def record_usage(self, actual_tokens: int) -> None:
        """
        Correct the token count for the most recent reserved slot.
        Call this AFTER receiving the API response with real usage numbers.
        """
        if self._window:
            last = self._window[-1]
            # Adjust the reservation with the real count
            self._window[-1] = _TokenEvent(
                timestamp=last.timestamp,
                tokens=actual_tokens,
            )

    def usage_report(self) -> dict[str, int | str]:
        """Return current usage snapshot (for CLI display / logging)."""
        now = time.monotonic()
        self._purge_old_events(now)
        self._check_daily_reset()
        return {
            "rpm_used": len(self._window),
            "rpm_limit": self.rpm_limit,
            "tpm_used": self._current_tpm(),
            "tpm_limit": self.tpm_limit,
            "rpd_used": self._day_requests,
            "rpd_limit": self.rpd_limit,
            "window_date": self._day_date,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _purge_old_events(self, now: float) -> None:
        """Remove events older than 60 seconds from the sliding window."""
        cutoff = now - 60.0
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    def _current_tpm(self) -> int:
        return sum(e.tokens for e in self._window)

    def _check_daily_reset(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._day_date:
            self._day_date = today
            self._day_requests = 0
            logger.debug("RateLimiter: daily counter reset for %s", today)

    def _seconds_until_slot(
        self, now: float, tokens: int, rpm_ok: bool, tpm_ok: bool
    ) -> float:
        """Estimate seconds to sleep until at least RPM or TPM constraint is released."""
        if self._window:
            oldest = self._window[0].timestamp
            return max(0.5, 60.0 - (now - oldest) + 0.1)
        return 1.0
