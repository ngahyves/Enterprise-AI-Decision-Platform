# app/security/guardrails.py
"""Input sanitization and prompt-injection detection.

Two layers of defense:
1. Regex blocklist for well-known injection phrasings.
2. Unicode normalization + control-character stripping to defeat
   homoglyph and null-byte obfuscation.
"""

import re
import unicodedata

from fastapi import HTTPException, status


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts?)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts?)",
    r"forget\s+(all\s+)?(previous|above|prior)\s+",
    r"you\s+are\s+now\s+(DAN|a\s+different|an?\s+unrestricted)",
    r"system\s+override",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"###\s*instruction",
    r"role\s+switch",
    r"new\s+instructions?:",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_input(text: str) -> str:
    """Normalize unicode and strip control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Keep newlines and tabs, drop other control characters
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()


def contains_injection(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_PATTERNS)


def validate_user_input(text: str) -> str:
    """Sanitize + validate. Raise HTTP 400 on injection detection."""
    sanitized = sanitize_input(text)
    if contains_injection(sanitized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Potential prompt injection detected in user input.",
        )
    return sanitized
