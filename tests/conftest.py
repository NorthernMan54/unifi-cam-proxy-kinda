"""
Test harness for characterization/golden-master tests against the two
external interfaces (UniFi Protect WebSocket, Frigate MQTT).

This module deliberately contains no imports from `unifi.*` — it only
knows how to load and compare raw protocol message dicts. See
TEST_CASE_CREATION_PLAN.md for the full test plan this supports.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fields whose values legitimately differ between two otherwise-identical
# captures (timestamps, sequence counters, per-run IDs) and must be
# ignored by comparison. A test that specifically needs to assert on one
# of these (e.g. trackerID allocation in the missed-new-event recovery
# path) should pass `keep_fields={"trackerID"}` to override this default
# for just that field.
NONDETERMINISTIC_FIELDS = {
    "messageId",
    "clockMonotonic",
    "clockWall",
    "clockStream",
    "clockBestMonotonic",
    "clockBestWall",
    "clockStreamRate",
    "timeStamp",
    "firstShownTimeMs",
    "idleSinceTimeMs",
    "eventId",
    "trackerID",
}


def _strip(obj: Any, ignore_fields: set[str]) -> Any:
    """Recursively drop nondeterministic keys from a message dict/list."""
    if isinstance(obj, dict):
        return {
            k: _strip(v, ignore_fields)
            for k, v in obj.items()
            if k not in ignore_fields
        }
    if isinstance(obj, list):
        return [_strip(item, ignore_fields) for item in obj]
    return obj


def normalize(message: dict, keep_fields: set[str] | None = None) -> dict:
    """Return a copy of `message` with nondeterministic fields removed.

    `keep_fields` excludes field names from stripping — use this when a
    test needs to assert on a field that's normally treated as
    nondeterministic.
    """
    ignore = NONDETERMINISTIC_FIELDS - (keep_fields or set())
    return _strip(copy.deepcopy(message), ignore)


def assert_message_equal(
    actual: dict,
    expected: dict,
    *,
    keep_fields: set[str] | None = None,
    msg: str | None = None,
) -> None:
    """Assert two protocol messages match, ignoring nondeterministic fields.

    This is the primary assertion for external-interface tests — never
    compare raw message dicts with `==` directly, since messageId/
    timestamps/clocks will almost always differ between a live run and
    a captured fixture even when behavior is correct.
    """
    norm_actual = normalize(actual, keep_fields)
    norm_expected = normalize(expected, keep_fields)
    assert norm_actual == norm_expected, (
        f"{msg + ': ' if msg else ''}message mismatch\n"
        f"  actual:   {json.dumps(norm_actual, sort_keys=True)}\n"
        f"  expected: {json.dumps(norm_expected, sort_keys=True)}"
    )


def assert_no_reply(sent_messages: list, *, since_index: int = 0) -> None:
    """Assert no message was sent after `since_index`.

    Silence is a first-class, assertable outcome for `responseExpected:
    false` inputs — several pairing-cascade messages are correct
    precisely because they produce no reply.
    """
    remaining = sent_messages[since_index:]
    assert remaining == [], f"expected no reply, but got: {remaining}"


def assert_sequence_order(
    actual_messages: list[dict], expected_function_names: list[str]
) -> None:
    """Assert a list of messages matches an expected order of
    functionName, independent of payload content.

    Use for cascades like the pairing config sequence, where ordering
    itself is part of the observed external contract.
    """
    actual_names = [m.get("functionName") for m in actual_messages]
    assert actual_names == expected_function_names, (
        f"sequence order mismatch\n"
        f"  actual:   {actual_names}\n"
        f"  expected: {expected_function_names}"
    )


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file by name (without .json) from
    tests/fixtures/."""
    path = FIXTURES_DIR / f"{name}.json"
    with path.open() as f:
        return json.load(f)


@pytest.fixture
def fixture_loader():
    """Pytest fixture form of load_fixture, for tests that prefer
    dependency injection over a direct import."""
    return load_fixture
