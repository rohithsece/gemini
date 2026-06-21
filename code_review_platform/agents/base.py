import os
import json
import httpx
from typing import Dict, Any, Optional

# Shared persistent HTTP client — reuses TCP connections across all agent calls
_shared_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    timeout=httpx.Timeout(20.0, connect=5.0),
)

class BaseGeminiAgent:
    def __init__(self, agent_name: str, system_instruction: str):
        self.agent_name = agent_name
        self.system_instruction = system_instruction
        # Try loading API key from environment variable
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # Default model
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    async def call_gemini(self, prompt: str, schema_description: str) -> Dict[str, Any]:
        """Calls the Gemini API to get a JSON response based on the prompt.
        If the API call fails or the API key is missing, raises an exception (handled by fallbacks).
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # We instruct Gemini to output JSON using systemInstruction and generationConfig
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"System Instruction:\n{self.system_instruction}\n\nTask prompt:\n{prompt}\n\nStrict Output Requirements:\nYou MUST respond with valid JSON following this description: {schema_description}"}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }

        response = await _shared_client.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(
                f"Gemini API returned status {response.status_code}: {response.text}"
            )

        resp_data = response.json()
        try:
            # Extract text output from Gemini response
            text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            # Parse JSON
            return json.loads(text.strip())
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to parse JSON response from Gemini: {e}. Raw response: {resp_data}")

    def run_fallback(self, code: str) -> Dict[str, Any]:
        """Override in subclasses to provide local heuristic-based analysis if Gemini fails."""
        return {
            "status": "warning",
            "summary": "Gemini API unavailable. Local fallback results shown.",
            "issues": []
        }
