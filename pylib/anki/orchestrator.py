# pylib/anki/orchestrator.py
from .ai_client import OllamaClient, AIEvalResult

class AIReviewOrchestrator:
    def __init__(self):
        self.client = OllamaClient()

    def evaluate(self, card, user_answer: str) -> AIEvalResult:
        # Extract plain text from card HTML (simplified for MVP)
        # In a real implementation, use anki.utils.stripHTML
        q = card.q()
        a = card.a()
        return self.client.evaluate_card(q, a, user_answer)