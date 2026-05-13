"""Structured output pattern: extract, validate, and retry once."""

import json
import os
import sys
from pathlib import Path

from common import check_env
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[2]))


TEST_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "feature": {"type": "string"},
        "test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},"title": {"type": "string"},"type": {"type": "string","enum": ["positive", "negative", "edge"]},
                    "preconditions": { "type": "array","items": {"type": "string"}},
                    "steps": {
                        "type": "array", "items": {"type": "string"}},
                    "expected_result": {"type": "string"},
                    "priority": {"type": "string","enum": ["low", "medium", "high"]}
                },
                "required": ["id","title","type","preconditions", "steps","expected_result","priority"],
                "additionalProperties": False } },
        "edge_cases": {
            "type": "array","items": {"type": "string"}},
            "missing_requirements": {
            "type": "array",
            "items": {"type": "string"}
        }
    },

    "required": [
        "feature","test_cases","edge_cases","missing_requirements"],
        "additionalProperties": False,
}

def ask_json(prompt, schema, *, name="result"):

    check_env()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model="gpt-5.4-nano",

        instructions="""
        Return ONLY valid JSON matching the schema.
        Do not include markdown.
        Keep the response concise.
        """,
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


def extract_test_cases(requirement: str):

    prompt = f"""
    You are an AI QA test case generation system. Convert the product requirement into structured test cases.
    Rules:
    - Generate ONLY:
      - 2 positive test cases
      - 2 negative test cases
      - 1 edge test case
    - Keep steps short
    - Keep expected results concise
    - Use only allowed enum values
    - Add unclear requirements into missing_requirements
    - Return valid JSON only

    Requirement:
    {requirement}
    """
    return ask_json(
        prompt,
        TEST_CASE_SCHEMA,
        name="test_case_workflow"
    )


def validate_test_cases(data):

    if not isinstance(data["test_cases"], list):
        raise ValueError("test_cases must be a list")

    valid_types = ["positive", "negative", "edge"]
    valid_priorities = ["low", "medium", "high"]

    for test_case in data["test_cases"]:

        if test_case["type"] not in valid_types:
            raise ValueError(
                f"Invalid type in {test_case['id']}"
            )

        if test_case["priority"] not in valid_priorities:
            raise ValueError(
                f"Invalid priority in {test_case['id']}"
            )

        if not isinstance(test_case["steps"], list):
            raise ValueError(
                f"Steps must be a list in {test_case['id']}"
            )


def generate_test_management_payload(data):

    payload = {
        "suite_name": data["feature"],
        "test_cases": []
    }

    for tc in data["test_cases"]:

        payload_case = {
            "title": tc["title"],
            "priority": tc["priority"],
            "steps": tc["steps"],
            "expected_result": tc["expected_result"]
        }

        payload["test_cases"].append(payload_case)

    return payload


def generate_coverage_summary(data):

    positive_cases = 0
    negative_cases = 0
    edge_cases = 0

    for tc in data["test_cases"]:

        if tc["type"] == "positive":
            positive_cases += 1

        elif tc["type"] == "negative":
            negative_cases += 1

        elif tc["type"] == "edge":
            edge_cases += 1

    return {
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "edge_cases": edge_cases,
        "missing_requirements_count": len(
            data["missing_requirements"]
        )
    }


def main():

    requirement = (
        "Users should be able to reset their password using an OTP sent to their registered email. "
        "The OTP should expire after 10 minutes.If the user enters the wrong OTP three times,the account should be temporarily locked."
    )
    retries = 1

    for attempt in range(retries + 1):

        try:
            structured_output = extract_test_cases(requirement)

            validate_test_cases(structured_output)

            break

        except (json.JSONDecodeError, ValueError) as exc:

            print(f"Attempt {attempt + 1} failed: {exc}")

            if attempt == retries:
                print("Workflow failed after retry.")
                return

            print("Retrying...\n")

    test_management_payload = (
        generate_test_management_payload(
            structured_output
        )
    )

    coverage_summary = (
        generate_coverage_summary(
            structured_output
        )
    )

    print("\nWorkflow completed successfully.")

    print("\nSTRUCTURED OUTPUT:\n")
    print(json.dumps(structured_output, indent=2))

    print("\nTEST MANAGEMENT PAYLOAD:\n")
    print(json.dumps(test_management_payload, indent=2))

    print("\nCOVERAGE SUMMARY:\n")
    print(json.dumps(coverage_summary, indent=2))

if __name__ == "__main__":
    main()