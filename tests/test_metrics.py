from src.eval.metrics import groundedness, refusal_accuracy, tool_accuracy


def test_groundedness_true():
    response = {"citations": [{"doc_id": "doc1", "snippet": "abc"}]}
    corpus = {"doc1": {"text": "abc def"}}
    assert groundedness(response, corpus)


def test_tool_accuracy():
    response = {"tool_calls": [{"name": "draft_prior_auth", "arguments": {}}]}
    assert tool_accuracy(response, "draft_prior_auth")


def test_refusal_accuracy():
    response = {"refusal": True}
    assert refusal_accuracy(response, True)
