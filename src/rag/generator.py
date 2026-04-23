"""
LLM-based response generator using Anthropic Claude.
Falls back to a structured template response if no API key is set.
"""
from __future__ import annotations
import logging
import os

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are SentinelEDU's AI Academic Advisor Assistant. You help human academic advisors \
support at-risk students. You have access to retrieved intervention guidelines and the \
student's real-time academic profile data.

Rules:
- Always ground your advice in the retrieved context documents provided.
- Be specific and actionable — avoid generic platitudes.
- Keep your response concise (3–5 short paragraphs max).
- Never diagnose mental health conditions; instead, recommend professional referral when signals suggest it.
- When referencing data (attendance %, quiz scores, risk class), cite the actual numbers.
- End with a clear "Recommended Next Actions" bulleted list (2–4 items).
"""


def _template_response(query: str, context_docs: list[dict], student_context: str) -> str:
    """Rule-based fallback when no Anthropic API key is available."""
    titles = [d["title"] for d in context_docs]
    snippets = "\n".join(f"• {d['title']}: {d['content'][:200]}..." for d in context_docs)

    return (
        f"**Based on retrieved guidelines:** {', '.join(titles)}\n\n"
        f"{snippets}\n\n"
        f"**Student Context:**\n{student_context}\n\n"
        "**Recommended Next Actions:**\n"
        "• Schedule an advisor meeting within 48 hours\n"
        "• Review the relevant intervention guidelines above\n"
        "• Document all outreach attempts in the student record\n"
        "• Set a follow-up reminder for 1 week\n\n"
        "_Note: Set ANTHROPIC_API_KEY for AI-generated personalised recommendations._"
    )


def generate(
    query: str,
    context_docs: list[dict],
    student_context: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """
    Generate an advisor recommendation using Claude.
    Falls back to a template response if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set — using template fallback.")
        return _template_response(query, context_docs, student_context)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        context_text = "\n\n".join(
            f"[{d['category'].upper()}] {d['title']}\n{d['content']}"
            for d in context_docs
        )

        user_message = (
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            f"STUDENT DATA:\n{student_context}\n\n"
            f"ADVISOR QUESTION:\n{query}"
        )

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    except Exception as exc:
        logger.error("Claude generation failed: %s", exc)
        return _template_response(query, context_docs, student_context)
