# pylib/anki/ai_client.py
import json
import urllib.request
import urllib.error
from dataclasses import dataclass

@dataclass
class AIEvalResult:
    verdict: str
    suggested_rating: str  # "Again", "Hard", "Good", "Easy"
    key_fix: str
    memory_tip: str
    raw_response: str

class OllamaClient:
    def __init__(self, model: str = "phi4", api_url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.api_url = api_url

    def evaluate_card(
        self, front: str, back: str, user_answer: str, card_mode: str = "basic"
    ) -> AIEvalResult:
        system_prompt = """
You are a study-coach grader for interview preparation.

You will be given:
- CARD_FRONT: the prompt/question
- CARD_BACK: the reference correct answer (ground truth)
- USER_ANSWER: what the user typed (may be incomplete or informal)

Your goal:
Evaluate whether USER_ANSWER demonstrates the essential understanding needed to answer this in a real interview.

Grading principles (follow strictly):
1) Prioritize conceptual correctness over exact wording.
2) Be fair: if USER_ANSWER captures the core idea with minor omissions, do NOT penalize harshly.
3) Be strict only for fatal misconceptions: if USER_ANSWER states something incorrect that would mislead an interviewer, mark it clearly.
4) Do NOT invent missing details beyond CARD_BACK. Base evaluation only on CARD_BACK and reasonable paraphrase equivalence.
5) Be concise: give at most one key correction and one memory tip.

Output format:
Return ONLY valid JSON with EXACTLY these keys:
- verdict: "pass" | "borderline" | "fail"
- suggested_rating: "Again" | "Hard" | "Good" | "Easy"
- key_fix: string (1-2 sentences max)
- memory_tip: string (1 sentence max)

Rating rubric:
- Easy: pass with strong confidence; clear and complete core idea.
- Good: pass; core idea correct; only minor omissions or minor imprecision.
- Hard: borderline; partially correct but needs follow-up or has a notable gap.
- Again: fail; core misunderstanding or fatal misconception.

Cloze handling:
If the card is cloze-style, accept reasonable synonyms/paraphrases if they preserve the meaning of the blank. Do not require exact string match unless CARD_BACK is clearly a proper noun/code/token where exactness is essential.

Now grade using CARD_FRONT, CARD_BACK, USER_ANSWER.
"""
        
        user_prompt = f"""
        CARD_TYPE: {card_mode}
        CARD_FRONT: {front}
        CARD_BACK: {back}
        USER_ANSWER: {user_answer}
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
                verdict="fail",
                suggested_rating="Again",
                key_fix=f"AI Error: {str(e)}",
                memory_tip="Retry in a moment or check your connection.",
                raw_response=""
            )

    def _parse_response(self, raw_text: str, context: str) -> AIEvalResult:
        try:
            data = json.loads(raw_text)
            verdict = data.get("verdict", "borderline")
            if verdict not in {"pass", "borderline", "fail"}:
                verdict = "borderline"
            suggested_rating = data.get("suggested_rating", "Good")
            if suggested_rating not in {"Again", "Hard", "Good", "Easy"}:
                suggested_rating = "Good"
            key_fix = data.get("key_fix") or "No key fix provided."
            memory_tip = data.get("memory_tip") or "No memory tip provided."
            return AIEvalResult(
                verdict=verdict,
                suggested_rating=suggested_rating,
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
