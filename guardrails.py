import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    allowed: bool
    requires_human: bool
    reason: str = ""


# ---------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+your\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?your\s+system\s+prompt",
    r"print\s+(your\s+)?system\s+prompt",
    r"developer\s+mode",
    r"system\s+override",
    r"bypass\s+(your\s+)?safety",
]


# ---------------------------------------------------------
# Sensitive-information patterns
# ---------------------------------------------------------

PII_PATTERNS = {

    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "phone": r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",

    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",

}


# ---------------------------------------------------------
# High-risk medical requests
# ---------------------------------------------------------

HIGH_RISK_PATTERNS = [

    r"\bprescribe\b",

    r"\bprescription\b",

    r"\bwhat medicine should i take\b",

    r"\bwhich medicine should i take\b",

    r"\brecommend (a|some)? medicine\b",

    r"\bsuggest (a|some)? medicine\b",

    r"\bmedicine for my symptoms\b",

    r"\bwhat drug should i take\b",

    r"\bdiagnose me\b",

    r"\bwhat disease do i have\b",

]


# ---------------------------------------------------------
# Main guardrail
# ---------------------------------------------------------

def check_input_guardrails(question: str) -> GuardrailResult:

    if not question or not question.strip():

        return GuardrailResult(
            allowed=False,
            requires_human=False,
            reason="Question cannot be empty."
        )


    question = question.strip()


    # -----------------------------------------------------
    # Length protection
    # -----------------------------------------------------

    if len(question) > 4000:

        return GuardrailResult(
            allowed=False,
            requires_human=False,
            reason="Question is too long."
        )


    # -----------------------------------------------------
    # Prompt injection
    # -----------------------------------------------------

    for pattern in PROMPT_INJECTION_PATTERNS:

        if re.search(
            pattern,
            question,
            re.IGNORECASE
        ):

            return GuardrailResult(
                allowed=False,
                requires_human=False,
                reason=(
                    "The request appears to contain "
                    "a prompt-injection attempt."
                )
            )


    # -----------------------------------------------------
    # PII detection
    # -----------------------------------------------------

    detected_pii = []

    for pii_type, pattern in PII_PATTERNS.items():

        if re.search(
            pattern,
            question,
            re.IGNORECASE
        ):

            detected_pii.append(pii_type)


    if detected_pii:

        return GuardrailResult(
            allowed=False,
            requires_human=False,
            reason=(
                "Please do not provide personal "
                "identifying information."
            )
        )


    # -----------------------------------------------------
    # High-risk medical request
    # -----------------------------------------------------

    for pattern in HIGH_RISK_PATTERNS:

        if re.search(
            pattern,
            question,
            re.IGNORECASE
        ):

            return GuardrailResult(
                allowed=True,
                requires_human=True,
                reason=(
                    "This request involves diagnosis, "
                    "prescription, or personalized "
                    "medication guidance."
                )
            )


    # -----------------------------------------------------
    # Normal request
    # -----------------------------------------------------

    return GuardrailResult(
        allowed=True,
        requires_human=False
    )


# ---------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------

def check_output_guardrails(answer: str) -> GuardrailResult:

    if not answer:

        return GuardrailResult(
            allowed=False,
            requires_human=False,
            reason="The model returned an empty response."
        )


    # -----------------------------------------------------
    # System prompt leakage
    # -----------------------------------------------------

    suspicious_output_patterns = [

        r"system prompt\s*:",
        r"developer instructions\s*:",
        r"api[_\s-]?key\s*[:=]",
        r"secret[_\s-]?key\s*[:=]",
        r"password\s*[:=]",

    ]


    for pattern in suspicious_output_patterns:

        if re.search(
            pattern,
            answer,
            re.IGNORECASE
        ):

            return GuardrailResult(
                allowed=False,
                requires_human=False,
                reason=(
                    "The generated response failed "
                    "the security output check."
                )
            )


    return GuardrailResult(
        allowed=True,
        requires_human=False
    )