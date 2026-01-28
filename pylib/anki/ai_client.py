# pylib/anki/ai_client.py
import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass

GEMINI_SYSTEM_INSTRUCTION = textwrap.dedent(
    """
You are an AI tutor and grader for interview preparation.
You will be given three inputs:

FLASHCARD_QUESTION: the prompt on the front of the card
CORRECT_ANSWER: the ground-truth reference answer
USER_ANSWER: what the user typed

Your task:
Compare USER_ANSWER against CORRECT_ANSWER and evaluate whether the user demonstrates the essential understanding needed to answer this correctly in a real technical interview.
You must output ONLY a valid JSON object with EXACTLY these fields:
{
"verdict": "Correct" | "Partially Correct" | "Incorrect",
"suggested_rating": 1 | 2 | 3 | 4,
"feedback": "string"
}

Grading principles (follow strictly):
- Semantic correctness over wording: Ignore minor typos. Accept reasonable paraphrases.
- Be strict about factual accuracy: Mark errors or misconceptions as Incorrect.
- Interview realism: Ask "Would this answer pass in a real interview?"
- Do not hallucinate: Base evaluation ONLY on CORRECT_ANSWER.
- Handle edge cases: If USER_ANSWER is empty, "I don't know", or clearly evasive, verdict is Incorrect.

Rating rubric (Anki logic):
4 (Easy): Fully correct, clear, no meaningful gaps.
3 (Good): Correct core idea, only minor imprecision.
2 (Hard): Partially correct, notable missing piece.
1 (Again): Incorrect or shows misunderstanding.

Feedback rules:
- Exactly ONE sentence.
- State the single most important thing they missed or got wrong.
- Be concise, concrete, and actionable.
- No praise, no fluff.

Output format rules (critical):
- Output MUST be valid JSON.
- No markdown, no commentary, no extra text.
"""
).strip()

@dataclass
class AIEvalResult:
    verdict: str
    suggested_rating: str  # "Again", "Hard", "Good", "Easy"
    key_fix: str
    memory_tip: str
    raw_response: str

class AIClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_url_template: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.api_url_template = api_url_template

    def evaluate_card(
        self, front: str, back: str, user_answer: str, card_mode: str = "basic"
    ) -> AIEvalResult:
        return self.generate_response(front, back, user_answer)

    def generate_response(
        self,
        question: str,
        correct_answer: str,
        user_answer: str,
        is_cloze: bool = False,
    ) -> AIEvalResult:
        if not self.api_key:
            return AIEvalResult(
                verdict="Incorrect",
                suggested_rating="Again",
                key_fix="Missing GEMINI_API_KEY environment variable. Set it in your shell or .env file.",
                memory_tip="Add GEMINI_API_KEY to your environment and retry.",
                raw_response="",
            )

        user_message = (
            "FLASHCARD_QUESTION: "
            + question
            + "\nCORRECT_ANSWER: "
            + correct_answer
            + "\nUSER_ANSWER: "
            + user_answer
        )
        if is_cloze:
            user_message += (
                "\n[SYSTEM NOTE: This is a CLOZE (fill-in-the-blank) card. "
                "The 'USER_ANSWER' generally contains ONLY the missing text, "
                "while 'CORRECT_ANSWER' contains the full completed sentence. "
                "Grade 'Correct' if the user's input accurately fills the gap in the QUESTION.]"
            )

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": GEMINI_SYSTEM_INSTRUCTION,
                    }
                ]
            },
            "contents": [
                {
                    "parts": [
                        {
                            "text": user_message,
                        }
                    ]
                }
            ],
        }

        try:
            api_url = self.api_url_template.format(api_key=self.api_key)
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                try:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError) as e:
                    raise ValueError("Gemini response missing expected fields.") from e
                return self._parse_response(content, user_message)

        except Exception as e:
            # Handle Flow B: AI Failure
            return AIEvalResult(
                verdict="Incorrect",
                suggested_rating="Again",
                key_fix=f"AI Error: {str(e)}",
                memory_tip="Retry in a moment or check your connection.",
                raw_response="",
            )

    def _parse_response(self, raw_text: str, context: str) -> AIEvalResult:
        try:
            data = json.loads(raw_text)
            verdict = data.get("verdict", "Partially Correct")
            verdict_map = {
                "pass": "Correct",
                "borderline": "Partially Correct",
                "fail": "Incorrect",
            }
            verdict = verdict_map.get(verdict, verdict)
            if verdict not in {"Correct", "Partially Correct", "Incorrect"}:
                verdict = "Partially Correct"

            suggested_rating = data.get("suggested_rating", 3)
            rating_map = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
            if isinstance(suggested_rating, str):
                if suggested_rating in rating_map.values():
                    rating_label = suggested_rating
                else:
                    try:
                        rating_label = rating_map[int(suggested_rating)]
                    except (ValueError, TypeError, KeyError):
                        rating_label = "Good"
            elif isinstance(suggested_rating, int):
                rating_label = rating_map.get(suggested_rating, "Good")
            else:
                rating_label = "Good"

            feedback = data.get("feedback")
            if isinstance(feedback, str) and feedback.strip():
                key_fix = feedback.strip()
                memory_tip = ""
            else:
                key_fix = data.get("key_fix") or "No key fix provided."
                memory_tip = data.get("memory_tip") or "No memory tip provided."

            return AIEvalResult(
                verdict=verdict,
                suggested_rating=rating_label,
                key_fix=key_fix,
                memory_tip=memory_tip,
                raw_response=raw_text
            )
        except json.JSONDecodeError:
            # Handle Flow C: Malformed Output
            return AIEvalResult(
                verdict="fail",
                suggested_rating="Again",
                key_fix="Could not parse AI response.",
                memory_tip="Try again or simplify your answer for clarity.",
                raw_response=raw_text
            )
        except Exception as e:
            return AIEvalResult(
                verdict="fail",
                suggested_rating="Again",
                key_fix=f"AI response format error: {str(e)}",
                memory_tip="Try again or simplify your answer for clarity.",
                raw_response=raw_text
            )


class OllamaClient(AIClient):
    pass
