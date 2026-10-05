"""
Prompt Builder

Constructs the system prompt and user prompt for the generation step.

Design principles:
- System prompt is treated as an engineering artifact (immutable guardrails)
- Context chunks are numbered and include source metadata (doc, page, type)
- Citation format is inline [N] footnotes with a SOURCES block at the end
- The LLM is instructed to refuse if context is insufficient — no hallucination
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# System Prompt (Engineering Artifact — do not soften these constraints)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise materials engineering and R&D assistant.
Your answers are used for safety-critical decisions — accuracy is paramount.

HARD RULES — these are engineering constraints, not suggestions:

1. CONTEXT-ONLY ANSWERS
   Answer exclusively from the provided context chunks below.
   Do NOT use prior knowledge, inference, or general scientific understanding.
   If the answer is not contained in the context, respond EXACTLY with:
   "I cannot answer this question based on the provided documents."
   — nothing else, no apologies, no partial guesses.

2. MANDATORY INLINE CITATIONS
   After every factual claim (numbers, formulas, material properties, test conditions,
   process parameters, conclusions), append an inline citation [N] where N is the
   source chunk number. Every sentence with a claim MUST have a citation.

3. VERBATIM PRECISION
   Copy numbers, units, chemical symbols, acronyms, and formula notation EXACTLY
   as they appear in the source. Do NOT rephrase or round.

4. NO EXTRAPOLATION
   Do not interpolate between data points, infer trends, or combine information
   from multiple sources to derive a conclusion that is not explicitly stated.

5. SOURCES BLOCK
   After your answer, output a fenced JSON block exactly like this:
   ```sources
   [
     {"ref": 1, "document_id": "...", "page_no": N, "chunk_type": "...", "excerpt": "..."},
     ...
   ]
   ```
   Include only sources you actually cited in your answer.
   The "excerpt" field must be the first 120 characters of the chunk text.
"""


# ---------------------------------------------------------------------------
# Context Assembly
# ---------------------------------------------------------------------------

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Format reranked chunks into a numbered context block for the LLM.

    Args:
        chunks: Reranked chunk dicts (must have `text`, `document_id`, `page_no`, `chunk_type`).

    Returns:
        A formatted string with numbered, source-labelled chunks.
    """
    lines = ["CONTEXT CHUNKS (use these and only these to answer):"]
    lines.append("=" * 70)

    for i, chunk in enumerate(chunks, start=1):
        doc_id = chunk.get("document_id", "unknown")
        page = chunk.get("page_no", "?")
        chunk_type = chunk.get("chunk_type", "text")
        headings = chunk.get("headings", [])
        text = chunk.get("text", "").strip()

        header_parts = [f"[{i}] Document: {doc_id} | Page: {page} | Type: {chunk_type}"]
        if headings:
            header_parts.append(f"     Section: {' > '.join(headings)}")

        lines.append("\n".join(header_parts))
        lines.append("-" * 40)
        lines.append(text)
        lines.append("")  # blank line between chunks

    lines.append("=" * 70)
    return "\n".join(lines)


def build_user_prompt(query: str, context_block: str) -> str:
    """
    Assemble the user message with the query and context.

    Args:
        query: The user's question.
        context_block: Formatted context from `build_context_block`.

    Returns:
        Full user prompt string.
    """
    return f"""{context_block}

QUESTION: {query}

ANSWER (cite every claim with [N], output SOURCES block at the end):"""
