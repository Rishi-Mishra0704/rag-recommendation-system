import time

import requests

from recommendation.constants.constants import OLLAMA_GENERATE_URL, LLM_MODEL


def _build_prompt(biz: dict) -> str:
    return (
        f"You are writing a B2B company profile for a trade platform.\n\n"
        f"Company: {biz['name']}\n"
        f"Industry: {biz['industry']} > {biz['sub_industry']}\n"
        f"Category: {biz['category']}\n"
        f"Location: {biz['location']}\n"
        f"Roles: {', '.join(biz['roles'])}\n"
        f"Tags: {', '.join(biz['tags'])}\n\n"
        f"Write exactly two sections:\n"
        f"DESCRIPTION: A 2-3 sentence factual description of the company's business and products.\n"
        f"PARTNER_GOALS: A 2-3 sentence statement of what kind of B2B partners this company is seeking.\n\n"
        f"Use plain text, no markdown, no bullet points."
    )


def _call_ollama(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def _parse_llm_response(text: str) -> tuple[str, str]:
    description, partner_goals = "", ""
    for line in text.splitlines():
        if line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PARTNER_GOALS:"):
            partner_goals = line.split(":", 1)[1].strip()
    # Fallback: split on the keyword if single-line response
    if not description and "PARTNER_GOALS:" in text.upper():
        idx = text.upper().index("PARTNER_GOALS:")
        description = text[:idx].replace("DESCRIPTION:", "").strip()
        partner_goals = text[idx:].split(":", 1)[1].strip()
    return description, partner_goals


def _fallback_texts(biz: dict) -> tuple[str, str]:
    description = (
        f"{biz['name']} is a {biz['category']} operating in the {biz['sub_industry']} "
        f"segment of the {biz['industry']} industry, based in {biz['location']}. "
        f"The company focuses on delivering quality products and services to B2B clients worldwide."
    )
    partner_goals = (
        f"{biz['name']} is looking for reliable {', '.join(biz['roles'])} partners "
        f"in the {biz['industry']} space. "
        f"We seek long-term trade relationships with companies that share our commitment to quality and efficiency."
    )
    return description, partner_goals


def generate_texts(biz: dict, retries: int = 3) -> tuple[str, str]:
    prompt = _build_prompt(biz)
    for attempt in range(retries):
        raw = _call_ollama(prompt)
        if raw:
            desc, goals = _parse_llm_response(raw)
            if desc and goals:
                return desc, goals
        if attempt < retries - 1:
            time.sleep(1)
    return _fallback_texts(biz)
