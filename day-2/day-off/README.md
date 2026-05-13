# AI Bug Triage Workflow

## Workflow Overview

This workflow accepts a messy bug report written in natural language and converts it into structured JSON using the OpenAI API.

The workflow performs the following steps:

1. Accepts a bug report as input
2. Extracts structured JSON using an LLM
3. Validates required fields using a JSON schema
4. Generates a Jira-style payload
5. Generates a Slack-style alert if the issue is a release blocker

---

## Input

The workflow accepts a natural language bug report.

Example:

During regression testing on staging, the login page is not working properly in Chrome.

---

## Structured Output

The workflow generates structured JSON with fields such as:

- title
- module
- environment
- browser
- severity
- release_blocker
- steps_to_reproduce
- expected_result
- actual_result
- missing_information
- suggested_next_action

---

## Validation Added

The workflow validates:

- Output must be valid JSON
- severity must use allowed enum values
- release_blocker must be boolean
- steps_to_reproduce must be a list
- missing_information must be a list
- Required fields must be present

---

## System Payloads Generated

### Jira-style Payload
Used for creating issue tickets in Jira.

### Slack-style Alert
Generated only if the issue is marked as a release blocker.

---

## What Could Go Wrong

- The AI may return incomplete or unclear information
- The bug report may miss important details
- Invalid API key or missing environment variables can cause failures
- Network/API issues may interrupt the workflow

---

## Future Improvements

- Add retry logic for invalid responses
- Add logging and monitoring
- Connect directly with Jira and Slack APIs
- Improve severity detection using additional rules