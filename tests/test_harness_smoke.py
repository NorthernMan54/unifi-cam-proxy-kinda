"""
Smoke test for the fixture/comparator harness itself (Test Case Creation
Plan, build order step 1). This does NOT test proxy code — the proxy's
`unifi.*` internals aren't imported anywhere here. It validates that:

  1. the comparator correctly ignores nondeterministic fields
  2. the comparator still catches genuine payload differences
  3. silence (no reply) is assertable as a first-class outcome
  4. message-sequence order is assertable independent of payload content

Fixtures used are two independent real GetIlluminance request/reply
captures from the Backyard G5 Bullet (real UniFi hardware, 60 seconds
apart) — not synthesized data. If this file passes, the harness is
trustworthy enough to build the real test modules
(test_pairing.py, test_zone_mapping.py, etc.) against it.
"""
from tests.conftest import (
    assert_message_equal,
    assert_no_reply,
    assert_sequence_order,
    load_fixture,
)


def test_comparator_ignores_nondeterministic_fields():
    """Two GetIlluminance requests captured 60s apart differ only in
    messageId — real, correct, and expected to differ. The comparator
    must treat them as equivalent messages."""
    request_a = load_fixture("get_illuminance_request_a")
    request_b = load_fixture("get_illuminance_request_b")
    assert request_a["messageId"] != request_b["messageId"], (
        "fixture setup error: these should differ in messageId, "
        "otherwise this test proves nothing"
    )
    assert_message_equal(request_a, request_b, msg="GetIlluminance request")


def test_comparator_catches_real_differences():
    """Sanity check the comparator isn't a no-op: a genuinely different
    illuminance reading is a real field value, not a nondeterministic
    one, and must NOT be silently normalized away."""
    reply_a = load_fixture("get_illuminance_reply_a")
    reply_b = dict(reply_a)
    reply_b["payload"] = dict(reply_a["payload"], illuminance=999)

    raised = False
    try:
        assert_message_equal(reply_a, reply_b)
    except AssertionError:
        raised = True
    assert raised, "comparator failed to catch a real payload difference"


def test_silence_is_assertable():
    """responseExpected: false inputs must produce zero replies. This is
    a first-class assertion, not just an absence of a check — several
    pairing-cascade messages (e.g. ChangeBrightnessSettings) are only
    correct because nothing is sent back."""
    sent_messages: list = []  # a real test replaces this with the
    # proxy's actual outbound queue after feeding it a
    # responseExpected: false message
    assert_no_reply(sent_messages)


def test_sequence_order_assertable():
    """Cascade order (e.g. the pairing config sequence) is part of the
    observed external contract and must be checked as order, not just
    checked for membership."""
    messages = [
        {"functionName": "NetworkStatus"},
        {"functionName": "ChangeDeviceSettings"},
        {"functionName": "ChangeOsdSettings"},
    ]
    assert_sequence_order(
        messages,
        ["NetworkStatus", "ChangeDeviceSettings", "ChangeOsdSettings"],
    )
