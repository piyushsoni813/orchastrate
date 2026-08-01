"""Calls an LLM with structured output to produce the notify/digest/mute
routing decision for messages that pass through safety_rules.py without a
hard override. Prompt text lives in prompts.py — this file only orchestrates
the provider calls.

Uses Gemini as the primary provider (reusing the same client, model, and
rate limiter already set up in media_resolver.py for vision) and falls back
to Groq on a transient Gemini failure, the same pattern as
resolve_all_media.py's image resolution fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import errors as genai_errors

from pipeline import prompts
from pipeline.context_loader import MessageContext, build_context
from pipeline.data_loader import load_all
from pipeline.evidence_retriever import retrieve_evidence
from pipeline.media_resolver import (
    GEMINI_MODELS,
    call_gemini_with_model_rotation,
    get_gemini_client,
    get_groq_client,
    groq_rate_limiter,
    load_media_cache,
)

GROQ_DECISION_MODEL = "llama-3.3-70b-versatile"


class RoutingDecision(BaseModel):
    action: Literal["notify", "digest", "mute"]
    message_type: Literal[
        "personal", "urgent", "event", "payment", "business_update",
        "promotion", "greeting", "forward", "spam", "scam", "unknown",
    ]
    reason: str
    confidence: float


def _parse_decision(raw_text: str) -> RoutingDecision:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return RoutingDecision.model_validate(json.loads(text))


def decide_via_gemini(
    ctx: MessageContext,
    media_cache: Optional[dict],
    evidence: List[Tuple[str, str]],
    client=None,
) -> RoutingDecision:
    from google.genai import types

    client = client or get_gemini_client()
    user_prompt = prompts.build_user_prompt(ctx, media_cache, evidence)

    def build_kwargs(model: str) -> dict:
        return dict(
            model=model,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=prompts.SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RoutingDecision,
            ),
        )

    response, _ = call_gemini_with_model_rotation(client, build_kwargs, models=GEMINI_MODELS)
    if response.parsed is not None:
        return response.parsed
    return _parse_decision(response.text)


def decide_via_groq(
    ctx: MessageContext,
    media_cache: Optional[dict],
    evidence: List[Tuple[str, str]],
    client=None,
) -> RoutingDecision:
    client = client or get_groq_client()
    user_prompt = prompts.build_user_prompt(ctx, media_cache, evidence)

    groq_rate_limiter.wait()
    response = client.chat.completions.create(
        model=GROQ_DECISION_MODEL,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _parse_decision(response.choices[0].message.content)


def make_decision(
    ctx: MessageContext,
    media_cache: Optional[dict],
    evidence: List[Tuple[str, str]],
    gemini_client=None,
) -> RoutingDecision:
    try:
        return decide_via_gemini(ctx, media_cache, evidence, client=gemini_client)
    except (genai_errors.ClientError, genai_errors.ServerError) as exc:
        is_transient = exc.code == 429 or exc.code >= 500
        if not is_transient:
            raise
        print(f"  [{exc.code}] {ctx.message.get('message_id')}: Gemini unavailable, "
              f"falling back to Groq ...")
        return decide_via_groq(ctx, media_cache, evidence)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bundle = load_all()
    media_cache = load_media_cache()

    for message_id in ["sample_msg_001", "sample_msg_004", "sample_msg_049"]:
        row = bundle.sample_messages[bundle.sample_messages["message_id"] == message_id].iloc[0]
        ctx = build_context(row, bundle)
        evidence = retrieve_evidence(ctx)
        decision = make_decision(ctx, media_cache, evidence)

        print(f"=== {message_id} ===")
        print(f"  labeled: action={row['action']} message_type={row['message_type']} "
              f"reason={row['reason']!r}")
        print(f"  predicted: action={decision.action} message_type={decision.message_type} "
              f"confidence={decision.confidence}")
        print(f"  reason: {decision.reason}")
        print()
