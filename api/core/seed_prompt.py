"""
Seed prompt builder.

Constructs the system + user prompt for the cold-path LLM call.
The goal is a response that:
  1. Directly answers the user's first question (returned to the user immediately)
  2. Provides a comprehensive, structured knowledge base on the topic
     (chunked, embedded, and stored for all subsequent warm-path turns)

Response structure the LLM is asked to follow:

  ## Direct Answer
  [concise answer to the user's question]

  ## [Topic Section]
  ### [Subsection]
  ...

The "## Direct Answer" section is extracted by the pipeline and returned
to the user as the first message. All sections including Direct Answer
are chunked and stored.
"""

from dataclasses import dataclass


@dataclass
class SeedPrompt:
    system_prompt: str
    user_prompt: str


_SYSTEM_PROMPT = """\
You are a structured knowledge base generator. Your responses are stored and \
used to answer follow-up questions, so quality and structure matter more than \
conversational tone.

Rules:
- Always begin with a "## Direct Answer" section — a clear, concise answer \
to the user's specific question.
- Follow with multiple sections covering all relevant aspects of the topic. \
Use ## for main sections and ### for subsections.
- Every sentence must carry information. No filler, no repetition, no \
meta-commentary about what you are about to say.
- Each section should be self-contained — a reader should be able to \
understand it without reading the others.
- Be thorough. Cover definitions, mechanisms, history, key figures, \
common misconceptions, and related concepts where relevant.
- Use plain prose within sections. Bullet points are acceptable for \
lists of distinct items but should not dominate.\
"""


def build_seed_prompt(query: str) -> SeedPrompt:
    """
    Build the seed prompt for a given user query.

    Args:
        query: The user's first message, used as the basis for the
               knowledge base topic and the direct answer.

    Returns:
        SeedPrompt with system_prompt and user_prompt ready for the LLM call.
    """
    user_prompt = (
        f"User question: {query}\n\n"
        "Respond with:\n"
        "1. A ## Direct Answer section answering the question above concisely.\n"
        "2. A comprehensive structured knowledge base on this topic using "
        "## and ### markdown headers to organise content into clear, "
        "distinct sections. Cover the topic thoroughly enough to answer "
        "reasonable follow-up questions without needing another large LLM call."
    )

    return SeedPrompt(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


def extract_direct_answer(llm_response: str) -> str:
    """
    Pull the ## Direct Answer section out of the LLM response to return
    to the user as the first message.

    Falls back to the full response if the section isn't found — this
    shouldn't happen with a well-behaved model but guards against it.
    """
    lines = llm_response.strip().splitlines()
    in_section = False
    answer_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("## direct answer"):
            in_section = True
            continue

        # Next ## header ends the section
        if in_section and stripped.startswith("## "):
            break

        if in_section:
            answer_lines.append(line)

    answer = "\n".join(answer_lines).strip()
    return answer if answer else llm_response.strip()
