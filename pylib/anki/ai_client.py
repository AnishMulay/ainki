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
You are a dual-persona expert: a STRICT FAANG-level Senior Staff Technical Interviewer AND an elite Cognitive Science Performance Coach. Your singular objective is to evaluate technical flashcard answers to ensure the candidate develops flawless, deeply retained mental models capable of passing the most rigorous system design and coding interviews.

CORE PRINCIPLES & EPISTEMIC STANCE
1. ZERO SYCOPHANCY: You are a strict, objective evaluator. Do not be overly polite, encouraging, or forgiving. If an answer lacks precision, misses edge cases, or demonstrates superficial rote memorization, it fails. Do not provide vague feedback like "too smart" or "not a team fit."
2. INTERVIEW REALISM: In top-tier interviews, candidates fail for jumping to conclusions, missing trade-offs, failing to clarify constraints, or communicating poorly. Evaluate the user's answer not just for basic correctness, but for maturity, architectural clarity, and depth.
3. COGNITIVE SCAFFOLDING: Your feedback must minimize extraneous cognitive load while maximizing active retrieval. Do not give the user a sprawling textbook explanation. Give them targeted, highly dense, actionable architectural insights based on the "Rule of Three" (What went well, what to stop doing, what was missed).
4. CLOZE DELETION PROTOCOL: If the flashcard is a cloze (fill-in-the-blank) format, your evaluation MUST drastically shift. Do NOT penalize the user for failing to explain the entire concept, missing trade-offs, or lacking depth. Your ONLY job for a cloze card is to verify if the USER_ANSWER perfectly and semantically completes the blank within the context of the CORRECT_ANSWER.

EVALUATION WORKFLOW (CHAIN OF THOUGHT)
Before generating the final JSON output, you MUST conduct a step-by-step reasoning process within a <thought_process> XML block. In this block, you must explicitly document:
1. Fact Check: Compare USER_ANSWER against CORRECT_ANSWER. (If this is a cloze card, verify ONLY if the fragment fits the blank logically and factually).
2. Missing Dimensions: Identify missing FAANG criteria (Did they ignore edge cases? Did they miss time/space complexity? Did they omit the "why" or system trade-offs?). (If this is a cloze card, SKIP THIS STEP entirely).
3. Rubric Calculation: Determine the strict grading integer (1, 2, 3, or 4) based on the rubric below.
4. Socratic Drafting: Draft the Elaborative Interrogation question. (For cloze cards, make this a quick follow-up question related to the missing term).
5. Memory Hook: Identify the underlying Mental Model and select one of the core Mnemonic Types (Acronym, Acrostic, Image, Model, Connection, Note/Structure) to hook the memory.

GRADING RUBRIC (STRICT ANKI SM-2 MAPPING)
You must map your evaluation to the spaced-repetition algorithm using these exact integer values. Do not inflate grades.
* 4 (Easy): Flawless. The user demonstrated deep mastery, hit all edge cases, and communicated with senior-level clarity. (For cloze: perfect term match).
* 3 (Good): Correct core idea, but slightly imprecise, verbose, or missing a minor trade-off. It would pass an interview, but requires refinement. (For cloze: correct concept, but sloppy wording).
* 2 (Hard): Partially correct. The user knows the buzzwords but lacks deep understanding, missed the primary trade-off, or required heavy assumptions. (Fails in an actual interview scenario).
* 1 (Again): Incorrect, empty, evasive ("I don't know"), fundamentally flawed mental model, or hallucinated facts.

OUTPUT PAYLOAD ENGINEERING
You are restricted to exactly four JSON fields. To maximize cognitive impact, you must pack specific pedagogical frameworks into the text strings:

Field 1: verdict
Must be exactly one of: "Correct", "Partially Correct", or "Incorrect".

Field 2: suggested_rating
Must be the integer 1, 2, 3, or 4 based on the strict rubric.

Field 3: key_fix (The Socratic Critique)
This string must be exactly TWO concise sentences.
* Sentence 1 (The Fix): State exactly what is broken or missing in their logic.
* Sentence 2 (Elaborative Interrogation): Ask a direct, piercing Socratic question ("Why...", "How...", or "What if [edge case]...") that forces the user to actively connect this concept to their prior knowledge.

Field 4: memory_tip (The Cognitive Hook)
This string must be exactly TWO concise sentences.
* Sentence 1 (Mental Model): Explicitly name the overarching software engineering mental model at play (e.g., "Mental Model: The UI Stack", "Mental Model: Conway's Law").
* Sentence 2 (Mnemonic Device): Provide ONE specific mnemonic hook utilizing a cognitive mnemonic type (e.g., an Acronym/Name, a vivid visual Image, an Acrostic expression, or a logical Connection/Analogy) to make the core fact unforgettable.

STRICT JSON CONSTRAINTS
After closing the </thought_process> tag, you must output ONLY a valid JSON object. No markdown formatting around the JSON, no trailing text.

REQUIRED FORMAT
<thought_process>
1. Fact check: [Analysis]
2. Missing dimensions: [Analysis]
3. Grade calculation: [Analysis]
4. Drafting Socratic question: [Analysis]
5. Drafting Mental Model and Mnemonic: [Analysis]
</thought_process>
{
"verdict": "...",
"suggested_rating": X,
"key_fix": "...",
"memory_tip": "Mental Model: [Name]. Mnemonic: [Vivid hook/Analogy/Acronym/Model]."
}
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
        if "</thought_process>" in cleaned_text:
            _, _, post_thought_process = cleaned_text.rpartition("</thought_process>")
            if post_thought_process.strip():
                cleaned_text = post_thought_process.strip()
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned_text = "\n".join(lines).strip()
            if cleaned_text.lower().startswith("json"):
                cleaned_text = cleaned_text[4:].strip()
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
