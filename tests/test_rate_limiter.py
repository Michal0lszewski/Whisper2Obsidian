"""Tests for LLMRateLimiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from whisper2obsidian.services.llm_rate_limiter import LLMRateLimiter


@pytest.fixture()
def limiter() -> LLMRateLimiter:
    return LLMRateLimiter(provider="groq", rpm_limit=3, tpm_limit=100, rpd_limit=10)


@pytest.mark.asyncio
async def test_capacity_granted_immediately(limiter: LLMRateLimiter) -> None:
    """With no prior usage, capacity should be granted immediately."""
    start = time.monotonic()
    await limiter.await_capacity(estimated_tokens=20)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_rpm_tracking(limiter: LLMRateLimiter) -> None:
    """Three requests should succeed within limits."""
    for _ in range(3):
        await limiter.await_capacity(10)
    report = limiter.usage_report()
    assert report["rpm_used"] == 3
    assert report["rpd_used"] == 3


@pytest.mark.asyncio
async def test_tpm_blocks_when_exceeded(limiter: LLMRateLimiter) -> None:
    """TPM limit of 100: two calls of 60 tokens should fill, third blocks."""
    await limiter.await_capacity(60)
    await limiter.await_capacity(39)  # total = 99, still under limit

    report = limiter.usage_report()
    assert report["tpm_used"] == 99


def test_record_usage_updates_last_entry(limiter: LLMRateLimiter) -> None:
    """record_usage should correct the last token reservation."""
    asyncio.run(limiter.await_capacity(50))
    limiter.record_usage(30)
    report = limiter.usage_report()
    assert report["tpm_used"] == 30


def test_rpd_counter_reset_on_new_day(limiter: LLMRateLimiter) -> None:
    """Simulating a day change should reset the daily counter."""
    limiter._day_requests = 9
    limiter._day_date = "2000-01-01"  # force stale date
    limiter._check_daily_reset()
    assert limiter._day_requests == 0


def test_usage_report_structure(limiter: LLMRateLimiter) -> None:
    report = limiter.usage_report()
    required_keys = {"rpm_used", "rpm_limit", "tpm_used", "tpm_limit", "rpd_used", "rpd_limit"}
    assert required_keys.issubset(report.keys())
