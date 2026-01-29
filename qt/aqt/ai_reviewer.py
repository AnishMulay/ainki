# qt/aqt/ai_reviewer.py
from aqt.reviewer import Reviewer
from aqt.qt import *
from aqt.utils import tooltip
from anki.ai_client import AIClient
from anki.utils import strip_html

class AIReviewer(Reviewer):
    def __init__(self, mw):
        super().__init__(mw)
        self.client = AIClient()
        self.input_field = None
        self.submit_btn = None
        self.ai_feedback_label = None

    def _initWeb(self):
        super()._initWeb()
        # Inject our custom UI elements below the webview
        # Note: In production, you might want to modify the HTML/JS directly
        # but for an MVP, we can overlay Qt widgets or inject HTML.
        # Here we assume a simple Qt widget overlay for the input.
        pass

    def _showQuestion(self):
        super()._showQuestion()
        self.web.eval("document.getElementById('ai-feedback')?.remove();")
        # 1. Clear previous state
        if self.input_field:
            self.input_field.deleteLater()
        
        # 2. Add Input Field (Free-form answer)
        layout = self.mw.mainLayout
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type or speak your answer here...")
        self.input_field.setMaximumHeight(100)
        layout.addWidget(self.input_field)

        # 3. Add Evaluate Button
        self.submit_btn = QPushButton("Evaluate with AI")
        self.submit_btn.clicked.connect(self.on_evaluate)
        layout.addWidget(self.submit_btn)

    def on_evaluate(self):
        cid = self.card.id
        # Disable button to prevent double submit
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("Thinking...")
        
        user_ans = self.input_field.toPlainText()
        question_html = self.card.question()
        answer_html = self.card.answer()
        cleaned_question = strip_html(question_html)
        cleaned_answer = strip_html(answer_html)
        is_cloze = self.card.note_type()["type"] == 1

        # Run in background to avoid freezing UI
        self.mw.taskman.run_in_background(
            lambda: self.client.generate_response(
                cleaned_question,
                cleaned_answer,
                user_ans,
                is_cloze=is_cloze,
            ),
            lambda future, expected_cid=cid: self.on_evaluation_complete(
                future, expected_cid
            ),
        )

    def on_evaluation_complete(self, future, expected_cid):
        if not self.card or self.card.id != expected_cid:
            return
        try:
            result = future.result()
            self._showAnswer(result)
        except Exception as e:
            tooltip(f"Error: {e}")
            self.submit_btn.setEnabled(True)

    def _showAnswer(self, ai_result=None):
        # Call standard showAnswer to render the back of the card
        super()._showAnswer()
        
        # Remove input widgets
        if self.input_field: self.input_field.hide()
        if self.submit_btn: self.submit_btn.hide()

        # Display AI Feedback
        if ai_result:
            feedback = f"""
            <div id="ai-feedback">
            <h3>AI Feedback</h3>
            <b>Verdict:</b> {ai_result.verdict}<br>
            <b>Suggested Rating:</b> {ai_result.suggested_rating}<br>
            <b>Key Fix:</b> {ai_result.key_fix}<br>
            <b>Memory Tip:</b> {ai_result.memory_tip}
            </div>
            """
            # Append feedback to the webview
            self.web.eval(f"document.body.innerHTML += `{feedback}`;")
