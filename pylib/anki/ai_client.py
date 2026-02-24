# pylib/anki/ai_client.py
import json
import logging
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GEMINI_SYSTEM_INSTRUCTION = textwrap.dedent(
    """
You are a senior technical interviewer evaluating interview prep answers.

PERSONA - STRICT INTERVIEWER + PERFORMANCE COACH

You are a senior technical interviewer for high-level infrastructure and systems roles.

Your evaluation style combines:

* FAANG / top infrastructure interview expectations
* Academic-style deep reasoning and conceptual rigor

Your primary goal is to evaluate whether an answer would realistically pass in a strong technical interview.

However, you are ALSO acting as a performance coach.

This means:

* You judge answers strictly and honestly.
* You do NOT behave like a tutor or teacher.
* You do NOT give long explanations.

But after evaluating, you briefly guide the candidate toward stronger interview performance.

Your coaching mindset:

* Assume the candidate is building long-term mastery.
* Focus on improving clarity, articulation, and mental models.
* Highlight what interview signal was missing (precision, structure, trade-offs, reasoning flow).
* Give concise direction that helps the candidate improve on the next attempt.

Important:

* You are strict but fair.
* You reward clear thinking, not memorized wording.
* You evaluate understanding, not surface similarity.
* You are helping the candidate become interview-ready through repeated practice.

Context:

The candidate is using this system as part of a structured interview-preparation framework designed around:

* deep understanding,
* active recall,
* teach-back loops,
* breadth-first skill progression,
* and long-term retention via Anki.

Your feedback should naturally align with that goal while staying concise.

You will be given three inputs:

FLASHCARD_QUESTION: the prompt on the front of the card
CORRECT_ANSWER: the ground-truth reference answer
USER_ANSWER: what the user typed

Your task:
Compare USER_ANSWER against CORRECT_ANSWER and determine whether the user shows the essential understanding needed to pass a strong real technical interview.
Focus on conceptual depth, clarity, completeness, and interview realism.
Be forgiving of wording and minor typos, but be strict about vagueness, missing core ideas, or incorrect claims.
Ask internally: "Would this answer pass a strong real interview without major follow-up?"

You must output ONLY a valid JSON object with EXACTLY these fields:
{
"verdict": "Correct" | "Partially Correct" | "Incorrect",
"suggested_rating": 1 | 2 | 3 | 4,
"key_fix": "string",
"memory_tip": "string"
}

Grading principles (follow strictly):
- Semantic correctness over wording: Accept reasonable paraphrases.
- Be strict about factual accuracy: Mark errors or misconceptions as Incorrect.
- Interview realism: If the answer is vague or missing a core idea, do not pass it.
- Do not hallucinate: Base evaluation ONLY on CORRECT_ANSWER.
- Handle edge cases: If USER_ANSWER is empty, "I don't know", or clearly evasive, verdict is Incorrect.

Rating rubric (Anki logic):
4 (Easy): Fully correct, clear, no meaningful gaps.
3 (Good): Correct core idea, only minor imprecision.
2 (Hard): Partially correct, notable missing piece.
1 (Again): Incorrect or shows misunderstanding.

key_fix rules:
- Exactly ONE concise sentence.
- State the single most important problem or missing idea.
- Be concrete and actionable.
- No praise, no fluff.

memory_tip rules:
- Exactly ONE concise sentence.
- Provide a mental hook or memory aid tied to the correct concept.
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
        api_url_template: str = "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={api_key}",
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

        prompt_text = GEMINI_SYSTEM_INSTRUCTION
        if is_cloze:
            prompt_text += (
                "\n\n[IMPORTANT NOTE: This is a CLOZE (fill-in-the-blank) card. "
                "The 'CORRECT_ANSWER' will be a full sentence. The 'USER_ANSWER' "
                "will be a text fragment. If the fragment correctly fills the blank "
                "in the context of the question, mark it as 'Correct'.]"
            )
        prompt_text += f"""

***
FLASHCARD_QUESTION: {question}
CORRECT_ANSWER: {correct_answer}
USER_ANSWER: {user_answer}
"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt_text,
                        }
                    ]
                }
            ],
        }

        try:
            api_url = self.api_url_template.format(api_key=self.api_key)
            logger.info("AI request: model_url=%s", self.api_url_template.split("?")[0])
            logger.info("AI request: is_cloze=%s", is_cloze)
            logger.info(
                "AI request: prompt_preview=%s",
                prompt_text[:1000].replace("\n", "\\n"),
            )
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as response:
                raw_body = response.read().decode("utf-8")
                print("\n[AI DEBUG] Raw GenAI HTTP response body:")
                print(raw_body)
                logger.info(
                    "AI response: status=%s length=%s",
                    getattr(response, "status", "unknown"),
                    len(raw_body),
                )
                logger.info(
                    "AI response: body_preview=%s",
                    raw_body[:1000].replace("\n", "\\n"),
                )
                result = json.loads(raw_body)
                try:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError) as e:
                    raise ValueError("Gemini response missing expected fields.") from e
                print("\n[AI DEBUG] Raw model text (before parsing):")
                print(content)
                return self._parse_response(content, prompt_text)

        except Exception as e:
            print(f"[AI DEBUG] Request exception: {e}")
            logger.exception("AI request failed")
            # Handle Flow B: AI Failure
            return AIEvalResult(
                verdict="Incorrect",
                suggested_rating="Again",
                key_fix=f"AI Error: {str(e)}",
                memory_tip="Retry in a moment or check your connection.",
                raw_response="",
            )

    def _parse_response(self, raw_text: str, context: str) -> AIEvalResult:
        print("\n[AI DEBUG] Entering _parse_response() with raw_text:")
        print(raw_text)
        logger.info("AI raw response:\n%s", raw_text)
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("`")
            cleaned_text = cleaned_text.removeprefix("json").strip()
        if not (cleaned_text.startswith("{") and cleaned_text.endswith("}")):
            start = cleaned_text.find("{")
            end = cleaned_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned_text = cleaned_text[start : end + 1]
        try:
            data = json.loads(cleaned_text)
            logger.info(
                "AI response (pretty):\n%s",
                json.dumps(data, indent=2, sort_keys=True),
            )
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
            print("[AI DEBUG] JSONDecodeError while parsing AI response.")
            logger.exception("AI response JSON parsing failed. raw_text=%r", raw_text)
            # Handle Flow C: Malformed Output
            return AIEvalResult(
                verdict="fail",
                suggested_rating="Again",
                key_fix="Could not parse AI response.",
                memory_tip="Try again or simplify your answer for clarity.",
                raw_response=raw_text
            )
        except Exception as e:
            print(f"[AI DEBUG] Unexpected parse exception: {e}")
            logger.exception("AI response format error. raw_text=%r", raw_text)
            return AIEvalResult(
                verdict="fail",
                suggested_rating="Again",
                key_fix=f"AI response format error: {str(e)}",
                memory_tip="Try again or simplify your answer for clarity.",
                raw_response=raw_text
            )


class OllamaClient(AIClient):
    pass
