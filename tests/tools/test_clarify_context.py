"""``context``: the material a clarify question refers to travels with the question.

The model's reasoning never reaches the user, so a question such as "proceed with this plan?"
must carry the plan. Surfaces that render ``context`` themselves (the messaging gateway) get it as
a field; every other surface gets it folded above the question text.
"""

import json

from tools.clarify_tool import MAX_CONTEXT_CHARS, clarify_tool


def _ask(questions, callback):
    return json.loads(clarify_tool(questions, callback=callback))


def test_plain_callback_sees_the_context_above_its_question():
    seen = []

    def callback(questions):
        seen.extend(questions)
        return {"answers": {"q0": "go"}, "outcome": "submitted"}

    result = _ask([{"question": "Proceed with this plan?", "context": "  1. research\n2. draft  ",
                    "choices": ["go", "stop"]}], callback)

    assert seen[0]["question"] == "1. research\n2. draft\n\nProceed with this plan?"
    # The model gets its own question back, not the folded display text.
    assert result["responses"][0]["question"] == "Proceed with this plan?"
    assert result["responses"][0]["user_response"] == "go"


def test_context_aware_callback_gets_the_field_and_an_untouched_question():
    seen = []

    def callback(questions):
        seen.extend(questions)
        return {"answers": {"q0": "go"}, "outcome": "submitted"}

    callback.supports_clarify_context = True
    _ask([{"question": "Proceed?", "context": "the plan"}, {"question": "Anything else?"}], callback)

    assert [(q["question"], q["context"]) for q in seen] == [("Proceed?", "the plan"), ("Anything else?", None)]


def test_blank_context_is_dropped():
    seen = []

    def callback(questions):
        seen.extend(questions)
        return {"answers": {}, "outcome": "submitted"}

    _ask([{"question": "Proceed?", "context": "   "}], callback)

    assert seen[0]["question"] == "Proceed?" and seen[0]["context"] is None


def test_invalid_context_is_rejected_before_anything_is_shown():
    called = []

    def callback(questions):
        called.append(questions)
        return {"answers": {}, "outcome": "submitted"}

    not_text = json.loads(clarify_tool([{"question": "Proceed?", "context": ["a"]}], callback=callback))
    too_long = json.loads(clarify_tool(
        [{"question": "Proceed?", "context": "x" * (MAX_CONTEXT_CHARS + 1)}], callback=callback))

    assert "context must be text" in not_text["error"]
    assert "limit" in too_long["error"]
    assert called == []
