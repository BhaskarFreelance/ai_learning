import os

from dotenv import load_dotenv

load_dotenv()


def configure_hf_token() -> str | None:
    """Ensure Hugging Face Hub auth environment is populated.

    This reads a token from HF_TOKEN or HUGGINGFACE_HUB_TOKEN and mirrors it to
    the standard environment variable names used by the Hugging Face client.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return token
