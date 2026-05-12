"""Structured output pattern: extract, validate, and retry once."""

import json
import os
import sys
from pathlib import Path

from common import check_env
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[2]))


TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_name": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["calm", "frustrated", "angry"]},
        "category": {"type": "string", "enum": ["auth", "billing", "bug", "other"]},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "issue": {"type": "string"},
        "next_action": {"type": "string"},
        "confidence": {"type": "number","minimum": 0,"maximum": 1},
        "needs_escalation": {"type": "boolean"}
        
    },
    "required": ["customer_name", "sentiment", "category", "urgency", "issue", "next_action", "confidence", "needs_escalation"],
    "additionalProperties": False,
}


def ask_json(prompt, schema, *, name="result"):
    check_env()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = client.responses.create(
        model="gpt-5.4-nano",
        instructions="Return only JSON that matches the provided schema.",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": True,
            }
        },
    )
    return json.loads(response.output_text)


def extract_ticket(message: str):
    prompt = f"""
    You are an AI support ticket extraction system.
    Extract the customer support message into structured JSON
    Rules:
    Use only schema enum values,
    Keep issue short and clear,
    Infer urgency from message tone,
    Suggest useful next action,
    Return a confidence score between 0 and 1.
    Use:
    High confidence (0.8–1.0) for clear issues
    Medium confidence (0.5–0.79) for partially clear issues
    Low confidence (<0.5) for vague or ambiguous messages
    
    Message:
    {message}
    """

    # print(prompt)

    return ask_json(prompt, TICKET_SCHEMA, name="support_ticket")


def main():
    message = (
        "Ashna reports that the dashboard crashes every time she uploads a PDF. This issue has blocked her work for two days and she is extremely upset."
    )
    try:
        ticket = extract_ticket(message)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Extraction failed: {exc}")
        return

    print(json.dumps(ticket, indent=2))


if __name__ == "__main__":
    main()
