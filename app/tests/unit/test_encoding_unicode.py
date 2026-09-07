from __future__ import annotations

import pandas as pd

from adsmod_core.common.utils.encoding import (
    decode_json_response_bytes,
    normalize_unicode_text,
    sanitize_dataframe_strings,
)

###############################################################################
def test_normalize_unicode_text_preserves_scientific_unicode() -> None:
    raw = "⁰¹²³⁴⁵⁶⁷⁸⁹ ₀₁₂₃₄₅₆₇₈₉ μµ αβγΔΩ ×÷±∓∞≈≠≤≥ √∑∏∫∂∇ ° ′ ″ µm Å Å Ω °C °F áéíóú ñçü åøæß “quotes” ‘single’ – — … • ‧"
    assert normalize_unicode_text(raw) == raw

###############################################################################
def test_normalize_unicode_text_removes_zero_width_and_normalizes_nbsp() -> None:
    raw = "A\u00a0B\u200bC\u200dD"
    assert normalize_unicode_text(raw) == "A BCD"

###############################################################################
def test_decode_json_response_bytes_utf8_round_trip() -> None:
    payload = decode_json_response_bytes(
        b'{"name":"CO\xe2\x82\x82","unit":"\xc2\xb5mol/g","sup":"\xe2\x81\xb0"}'
    )
    assert payload == {"name": "CO₂", "unit": "µmol/g", "sup": "⁰"}

###############################################################################
def test_sanitize_dataframe_strings_handles_unicode_without_loss() -> None:
    frame = pd.DataFrame(
        {
            "name": ["zeolite\u200b-a", "na\u00a0y"],
            "unit": ["µmol/g", "Å"],
        }
    )
    sanitized = sanitize_dataframe_strings(frame)
    assert sanitized.loc[0, "name"] == "zeolite-a"
    assert sanitized.loc[1, "name"] == "na y"
    assert sanitized.loc[0, "unit"] == "µmol/g"
    assert sanitized.loc[1, "unit"] == "Å"
