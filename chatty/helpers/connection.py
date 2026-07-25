from dotenv import load_dotenv
import os
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from openai import APIStatusError


def load_environment() -> None:
	"""Load environment variables from the chatty project .env file when available."""
	project_root = Path(__file__).resolve().parent.parent
	for candidate in [project_root / ".env", project_root.parent / ".env"]:
		if candidate.exists():
			load_dotenv(candidate, override=False)

	# Fallback for direct execution from other working directories.
	if not os.getenv("OPENAI_API_KEY"):
		load_dotenv(override=False)


load_environment()


def resolve_model(model: str | None) -> str:
	"""Return a chat-compatible model name for the application's chat endpoint."""
	if not model:
		return "gpt-3.5-turbo"
	model_name = model.strip()
	if model_name in {"gpt-3.5-turbo-instruct", "gpt-3.5-turbo"}:
		return "gpt-3.5-turbo"
	if model_name in {"gpt-4o-mini", "gpt-4.1-mini"}:
		return "gpt-3.5-turbo"
	return model_name


def format_openai_error(error: Exception) -> str:
	"""Convert OpenAI SDK errors into a readable message for the API layer."""
	status_code = getattr(error, "status_code", None)
	body = getattr(error, "body", None)
	message = getattr(error, "message", str(error)) or str(error)
	if isinstance(body, dict):
		error_body = body.get("error") or {}
		code = error_body.get("code")
		message = error_body.get("message") or message
		if code == "insufficient_quota":
			return "OpenAI quota exhausted. Check your billing/credits and ensure the API key is for an account with available quota."
		if code == "rate_limit_exceeded":
			return "OpenAI rate limit exceeded. Please wait a moment and retry."
	if status_code == 429:
		return "OpenAI quota exhausted or rate limited. Check your billing/credits and retry later."
	return message


class OppenAI:
	"""Simple OpenAI API client that reads the API key from `.env`.

	The API key should be set in the `OPENAI_API_KEY` environment variable.
	"""

	def __init__(self, env_var: str = "OPENAI_API_KEY"):
		load_environment()
		self.api_key = os.getenv(env_var)
		if not self.api_key:
			raise RuntimeError(f"OpenAI API key not found in environment variable '{env_var}'")
		self.client = OpenAI(api_key=self.api_key)

	def chat(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 150) -> Dict[str, Any]:
		"""Send a chat-style request to the OpenAI API.

		messages should be a list like:
		[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
		"""
		try:
			response = self.client.chat.completions.create(
				model=resolve_model(model),
				messages=messages,
				temperature=temperature,
				max_tokens=max_tokens,
			)
			return response.model_dump()
		except APIStatusError as e:
			raise RuntimeError(format_openai_error(e)) from e
		except Exception as e:
			raise RuntimeError(format_openai_error(e)) from e

	def completion(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 150, temperature: float = 0.7) -> Dict[str, Any]:
		"""Create a text completion request using the modern chat/completions-compatible API."""
		try:
			response = self.client.completions.create(
				model=resolve_model(model),
				prompt=prompt,
				max_tokens=max_tokens,
				temperature=temperature,
			)
			return response.model_dump()
		except APIStatusError as e:
			raise RuntimeError(format_openai_error(e)) from e
		except Exception as e:
			raise RuntimeError(format_openai_error(e)) from e
