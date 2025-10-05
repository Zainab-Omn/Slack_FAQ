# qa_extractor.py
from __future__ import annotations

import json
import os
from typing import Dict, Any

from dotenv import load_dotenv
from dotenv import dotenv_values

from openai import OpenAI



config = dotenv_values(".env")
api_key = config.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)



SYSTEM_PROMPT = """
You are an assistant that extracts Q&A pairs from Slack threads to build an FAQ. 
Only include questions that have a confirmed working answer.

Detection rules (strict):

- Identify all distinct questions in the thread (initial post or replies).
- If the same user replies to themselves:
  * If it clarifies the same question → append to the question.
  * If it’s a different question → create a new Q&A candidate.
- Validation required to include a Q&A:
  * A question can be included only if there is explicit evidence of success, such as:
    - A follow-up from the asker or thread participants confirming success 
      (e.g., “that worked”, “fixed”, “resolved”, “✅”, “thanks, it works now”).
    - Or clear objective evidence in the thread that the fix worked (final logs/outcomes showing success).
  * If there is no explicit success confirmation, treat as unresolved and do not include it.
- Negative signals (treat as unresolved): 
  “didn’t work”, “still fails”, “same error”, “not solved”, 
  “ValueError remains”, “binary incompatibility”, “no luck”, 
  “any other ideas?”, “had to create env but didn’t work”.
- Skip any answer that clearly did not work. 
- If no working answer exists for a question, skip that Q&A. 
- If no questions have a working answer, output exactly:
  {"qas": []}
- If the thread has no question at all, output exactly:
  {"qas": []}
- If a Q&A is about a time-sensitive event (e.g. course deadlines, submission dates, schedules): 
  skip it entirely


Output format (strict JSON, no text outside JSON):

{
  "channel": "<channel_name>",
  "thread_ts": "<thread_timestamp>",
  "qas": [
    {
      "question": "Clean full question, including clarifications (remove mentions like <@U123>)",
      "answer": "The confirmed working answer. Preserve code/commands.",
      "asked_by": "Uxxxx",
      "answered_by": "Uxxxx"

    }

    },
    {
      "question": "...",
      "answer": "...",
      "asked_by": "...",
      "answered_by": "..."
    }
  ]
}

Never output placeholders like “no solution provided”. 
Do not include usernames or timestamps outside of the 'asked_by' and 'answered_by' fields. 
Be conservative: when in doubt about success, output {"qas": []}.
"""

def extract_qas(thread_data: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> Dict[str, Any]:
    """
    Run the LLM on a Slack thread payload and return parsed Q&A JSON.

    Parameters
    ----------
    thread_data : str
        The raw thread text or JSON you pass as the user message.
    model : str
        OpenAI chat model name. Default: "gpt-4o-mini".
    temperature : float
        Sampling temperature. Default: 0.2.

    Returns
    -------
    dict
        Parsed dict with shape: {"qas": [ {question, answer, asked_by, answered_by}, ... ]}

    Raises
    ------
    RuntimeError
        If the OpenAI response is empty.
    ValueError
        If the response is not valid JSON.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": thread_data},
        ],
        temperature=temperature,
    )

    if not resp.choices or not resp.choices[0].message or not resp.choices[0].message.content:
        raise RuntimeError("LLM returned no content.")

    content = resp.choices[0].message.content.strip()

    # Enforce strict JSON as required by your prompt
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        # Surface the raw content to help debug prompt drift
        raise ValueError(f"Model output was not valid JSON. Error: {e}\nOutput:\n{content}") from e

    # Optional: minimal schema sanity check
    if not isinstance(data, dict) or "qas" not in data or not isinstance(data["qas"], list):
        raise ValueError(f"Model output JSON missing expected 'qas' list.\nOutput:\n{json.dumps(data, indent=2)}")

    return data

__all__ = ["extract_qas", "SYSTEM_PROMPT"]
