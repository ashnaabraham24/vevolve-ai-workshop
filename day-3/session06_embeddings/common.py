"""Common utilities for session06_embeddings."""

import os

import dotenv
from openai import OpenAI


def load_env():
    """Load environment variables from .env file. Returns True if successful."""
    dotenv.load_dotenv()
    return os.path.exists(".env")


def check_env():
    """Check if environment variables are set. Raises ValueError if not."""
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    # if not os.getenv("OPENAI_BASE_URL"):
    #     raise ValueError("OPENAI_BASE_URL not set")


def get_openai_client():
    """Get an OpenAI client configured with environment variables."""
    check_env()
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_workshop_model(default="gpt-5.4-nano"):
    """Get the model name from environment or use default."""
    return os.getenv("OPENAI_MODEL", default)


def ask_model(prompt, max_output_tokens=200):
    """Simple helper to ask the model a question."""
    client = get_openai_client()
    model = get_workshop_model()
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text
