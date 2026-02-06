import re

INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"reveal (the )?system prompt",
    r"act as system",
    r"bypass safety",
    r"disable guardrails",
]

MEDICAL_ADVICE_PATTERNS = [
    r"diagnose",
    r"what is the diagnosis",
    r"prescribe",
    r"dosage",
    r"treatment plan",
    r"treatment for",
    r"treatment",
    r"therapy",
    r"medical advice",
    r"what is .*cancer",
    r"whats .*cancer",
    r"lung cancer",
]

DEFAULT_MEDICAL_DISCLAIMER = (
    "This is general information, not a medical diagnosis. "
    "Please consult a licensed clinician."
)


def detect_prompt_injection(text):
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def is_medical_advice(text):
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in MEDICAL_ADVICE_PATTERNS)


def should_refuse(text, allow_medical_advice=False):
    if allow_medical_advice:
        return False
    return is_medical_advice(text)


def refusal_response(message):
    return {
        "answer": message,
        "citations": [],
        "tool_calls": [],
        "refusal": True,
        "follow_up": "If this is urgent, contact emergency services or a licensed clinician."
    }
