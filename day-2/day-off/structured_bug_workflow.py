
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from common import check_env
from openai import OpenAI

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parents[2]))

BUG_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "module": {"type": "string"},
        "environment": {"type": "string"},
        "browser": {"type": "string"},
        "severity": {"type": "string","enum": ["low", "medium", "high", "critical"]},
        "release_blocker": {"type": "boolean"},
        "steps_to_reproduce": {"type": "array","items": {"type": "string"}},
        "expected_result": {"type": "string"},
        "actual_result": {"type": "string"},
        "missing_information": {"type": "array", "items": {"type": "string"} },
        "suggested_next_action": {"type": "string"}
    },
    "required": [
        "title",
        "module",
        "environment",
        "browser",
        "severity",
        "release_blocker",
        "steps_to_reproduce",
        "expected_result",
        "actual_result",
        "missing_information",
        "suggested_next_action"
    ],
    "additionalProperties": False,
}


def ask_json(prompt, schema, *, name="result"):   #sends a prompt to OpenAI's GPT-4 model.
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions="Return only valid JSON matching the schema.",
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


def extract_bug_report(report: str):  #creates the specific instructions for the AI

    prompt = f"""
    You are an AI bug triage assistant.
    Extract the bug report into structured JSON.
    Rules:
    - Use only allowed enum values for severity.
    - Do not guess missing information.
    - Put unclear details inside missing_information.
    - release_blocker should be true only if the issue blocks release/testing.
    - steps_to_reproduce must be a list.
    - suggested_next_action should be practical and short.

    Bug Report:
    {report}
    """

    return ask_json(prompt, BUG_SCHEMA, name="bug_report")

def generate_jira_payload(bug):   #Converts the bug data into a format that Jira (a task tracker) understands

    jira_payload = {
        "summary": bug["title"],
        "priority": bug["severity"],
        "labels": [ bug["module"],bug["browser"], bug["environment"]],
        "description": (
            f"Steps: {bug['steps_to_reproduce']}\n\n"
            f"Expected Result: {bug['expected_result']}\n\n"
            f"Actual Result: {bug['actual_result']}\n\n"
            f"Suggested Next Action: {bug['suggested_next_action']}"
        )
    }

    return jira_payload

def generate_slack_alert(bug):  #If the bug is marked as a release blocker, it creates a formatted text alert to notify a team on Slack immediately.

    if bug["release_blocker"]:
        return (
    f"Release Blocker Detected\n"
    f"Bug: {bug['title']}\n"
    f"Module: {bug['module']}\n"
    f"Severity: {bug['severity']}\n"
    f"Environment: {bug['environment']}\n\n"
    f"Suggested Action:\n"
    f"{bug['suggested_next_action']}"
)
    return None


def main():
    bug_report = """
    During regression testing on staging, the login page is not working properly in Chrome.
    I entered a valid username and password, clicked the Login button, and the page just kept loading.
    Expected result: user should land on the dashboard.
    Actual result: loading spinner stays forever.
    This is blocking our release testing today. Firefox seems to work fine.
    """
    try:

        bug = extract_bug_report(bug_report)  #It calls extract_bug_report

        print("\nSTRUCTURED BUG JSON:\n")
        print(json.dumps(bug, indent=2))

        jira_payload = generate_jira_payload(bug)

        print("\nJIRA PAYLOAD:\n")
        print(
f"""
Summary: {jira_payload['summary']}
Priority: {jira_payload['priority']}
Labels: {jira_payload['labels']}

Description:
{jira_payload['description']}
"""
)

        slack_alert = generate_slack_alert(bug)

        if slack_alert:
            print("\nSLACK ALERT:\n")
            print(slack_alert)

    except (json.JSONDecodeError, ValueError) as exc:

        print(f"Workflow failed: {exc}")


if __name__ == "__main__":
    main()