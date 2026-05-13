# AI Test Case Generator Workflow

## Overview

This workflow converts messy product requirements into structured QA test cases using OpenAI structured outputs.

The workflow follows this pipeline:

Messy Requirement → AI Extraction → Structured JSON → Validation → Test Management Payload → Coverage Summary

---

# Input

The workflow accepts product requirements written in natural language.

Example input:

```text
Users should be able to reset their password using an OTP sent to their registered email.
The OTP should expire after 10 minutes.
If the user enters the wrong OTP three times, the account should be temporarily locked.
```

---

# Structured Output

The workflow generates structured JSON containing:

- Feature name
- Positive test cases
- Negative test cases
- Edge test cases
- Missing or unclear requirements

Example structure:

```json
{
  "feature": "Password reset using OTP",
  "test_cases": [],
  "edge_cases": [],
  "missing_requirements": []
}
```

---

# Validation Added

The workflow validates:

- Output must be valid JSON
- `test_cases` must be a list
- Each test case must include:
  - id
  - title
  - type
  - preconditions
  - steps
  - expected_result
  - priority
- `type` must be:
  - positive
  - negative
  - edge
- `priority` must be:
  - low
  - medium
  - high
- `steps` must be a list

The workflow also retries once if extraction or validation fails.

---

# Test Management Payload

The workflow generates a simplified test-management-style payload.

Example:

```json
{
  "suite_name": "Password reset using OTP",
  "test_cases": [
    {
      "title": "Reset password with valid OTP",
      "priority": "high",
      "steps": [],
      "expected_result": ""
    }
  ]
}
```

---

# Coverage Summary

The workflow generates a coverage summary including:

- Number of positive test cases
- Number of negative test cases
- Number of edge test cases
- Count of missing requirements

Example:

```json
{
  "positive_cases": 2,
  "negative_cases": 2,
  "edge_cases": 1,
  "missing_requirements_count": 3
}
```

---

# What Could Go Wrong

Possible issues include:

- AI may return incomplete or invalid JSON
- AI may hallucinate requirements
- Duplicate test cases may be generated
- Missing requirements may not always be detected correctly
- Ambiguous requirements can reduce output quality

---

# Improvements For Future Versions

Possible future improvements:

- Add database storage
- Add API integrations with test management tools
- Add automatic retry with exponential backoff
- Use Pydantic for stronger validation
- Add automatic duplicate test case detection
- Add severity/risk scoring for test cases

---
