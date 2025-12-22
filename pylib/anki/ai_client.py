# pylib/anki/ai_client.py
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AIEvalResult:
    mistake_summary: str
    wrong_points: List[str]
    missing_points: List[str]
    suggested_rating: str  # "Again", "Hard", "Good", "Easy"
    raw_response: str

class OllamaClient:
    def __init__(self, model: str = "phi4", api_url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.api_url = api_url

    def evaluate_card(self, front: str, back: str, user_answer: str) -> AIEvalResult:
        # Construct the prompt ensuring strict JSON output
        system_prompt = (
            "You are a flashcard grader. Compare the User Answer to the Correct Answer. "
            "The Front Context is provided for reference. "
            "Output valid JSON ONLY with these keys: "
            "'mistake_summary' (concise explanation), "
            "'wrong_points' (list of strings), "
            "'missing_points' (list of strings), "
            "'suggested_rating' (one of: 'Again', 'Hard', 'Good', 'Easy')."
        )
        
        user_prompt = f"""
        Front Context: {front}
        Correct Answer: {back}
        User Answer: {user_answer}
        """

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json"
        }

        try:
            req = urllib.request.Request(
                self.api_url, 
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return self._parse_response(result.get("response", ""), user_prompt)
                
        except Exception as e:
            # Handle Flow B: AI Failure
            return AIEvalResult(
                mistake_summary=f"AI Error: {str(e)}", 
                wrong_points=[], missing_points=[], 
                suggested_rating="None", raw_response=""
            )

    def _parse_response(self, raw_text: str, context: str) -> AIEvalResult:
        try:
            data = json.loads(raw_text)
            return AIEvalResult(
                mistake_summary=data.get("mistake_summary", "No summary provided."),
                wrong_points=data.get("wrong_points", []),
                missing_points=data.get("missing_points", []),
                suggested_rating=data.get("suggested_rating", "Good"),
                raw_response=raw_text
            )
        except json.JSONDecodeError:
            # Handle Flow C: Malformed Output
            return AIEvalResult(
                mistake_summary="Could not parse AI response.",
                wrong_points=[], missing_points=[],
                suggested_rating="None",
                raw_response=raw_text
            )