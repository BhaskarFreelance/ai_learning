from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import openai

load_dotenv()


class OppenAI:
	"""Simple OpenAI API client that reads the API key from `.env`.

	The API key should be set in the `OPENAI_API_KEY` environment variable.
	"""

	def __init__(self, env_var: str = "OPENAI_API_KEY"):
		self.api_key = os.getenv(env_var)
		if not self.api_key:
			raise RuntimeError(f"OpenAI API key not found in environment variable '{env_var}'")
		openai.api_key = self.api_key

	def chat(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 150) -> Dict[str, Any]:
		"""Send a chat-style request to the OpenAI API.

		messages should be a list like:
		[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
		"""
		response = openai.ChatCompletion.create(
			model=model,
			messages=messages,
			temperature=temperature,
			max_tokens=max_tokens,
		)
		return response

	def completion(self, prompt: str, model: str = "text-davinci-003", max_tokens: int = 150, temperature: float = 0.7) -> Dict[str, Any]:
		"""Create a legacy completion request (text completion)."""
		response = openai.Completion.create(
			model=model,
			prompt=prompt,
			max_tokens=max_tokens,
			temperature=temperature,
		)
		return response
