"""Extract phone numbers from raw OCR text.

Strategy (in order):
  1. phonenumbers.PhoneNumberMatcher – handles E.164, national, international formats
  2. Regex fallback for bare Indian 10-digit mobiles (starts with 6–9)
  3. Deduplicate preserving first-seen order
"""

import re

import phonenumbers
import structlog
from typing import List, Tuple
logger = structlog.get_logger(__name__)

# Cap OCR text fed to the phonenumbers library to prevent excessive scan time
# on adversarially crafted images.
_MAX_TEXT_LEN = 4_000  # characters

# Indian mobile: optional country code prefix then 10 digits starting 6-9.
# Deliberately simple to avoid ReDoS via catastrophic backtracking.
_INDIAN_MOBILE_RE = re.compile(
    r"(?:(?:\+|00)?91[\s\-.]?)?"  # optional +91 / 0091
    r"([6-9]\d{9})"               # 10-digit mobile
)


def _normalise(num_str: str) -> str:
    """Strip whitespace/hyphens/dots for deduplication key, then strip leading 91 country code."""
    key = re.sub(r"[\s\-.()+]", "", num_str)
    # Normalise Indian numbers: 918722359047 → 8722359047
    if len(key) == 12 and key.startswith("91") and key[2] in "6789":
        key = key[2:]
    return key


def extract_phone_numbers(
    text: str, default_region: str = "IN", max_results: int = 0
) -> Tuple[List[str], List[str]]:
    """Return (phone_number_strings, normalised_keys) deduplicated.

    Args:
        max_results: stop after finding this many unique numbers (0 = unlimited).
                     Pass 1 when only a single phone number is expected (e.g. a
                     labour card) to short-circuit both passes as soon as it is found.
    """
    # Guard against adversarially long OCR output
    text = text[:_MAX_TEXT_LEN]

    results: List[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> bool:
        """Add number if unseen. Returns True when max_results limit is reached."""
        key = _normalise(raw)
        if key and key not in seen and len(key) >= 7:
            seen.add(key)
            results.append(raw)
            if max_results > 0 and len(results) >= max_results:
                return True
        return False

    # Pass 1: phonenumbers library
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, default_region):
            e164 = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
            if _add(e164):
                logger.debug("phone_extraction.result", count=len(results), numbers=results)
                return results, list(seen)
    except Exception as exc:
        logger.warning("phonenumbers.matcher.failed", error=str(exc))

    # Pass 2: Indian mobile regex fallback
    for m in _INDIAN_MOBILE_RE.finditer(text):
        raw = m.group(1)  # bare 10 digits
        if _add(raw):
            logger.debug("phone_extraction.result", count=len(results), numbers=results)
            return results, list(seen)

    logger.debug("phone_extraction.result", count=len(results), numbers=results)
    return results, list(seen)
