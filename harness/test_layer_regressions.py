"""Regression checks for line-scoped quotations and malformed citations."""

from types import SimpleNamespace

import pytest

from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic


def context(observed):
    docs = [SimpleNamespace(doc_id="source", body="First evidence line\nSecond evidence line")]
    corpus = SimpleNamespace(docs=docs, get=lambda doc_id: {d.doc_id: d for d in docs}.get(doc_id))
    return SimpleNamespace(corpus=corpus, observed_text=observed)


def test_observed_multiline_claim_is_removed_without_rewriting_it():
    ctx = context("First evidence line\nSecond evidence line")
    claim = {"text": ctx.observed_text, "doc_id": "source"}
    original = dict(claim)
    report = {"claims": [claim], "answer": claim["text"], "abstain": False}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert report["claims"] == [] and report["citations"] == []
    assert report["abstain"] is True
    assert claim == original


@pytest.mark.parametrize("doc_id", [[], ["source"], {}, 42, None])
def test_malformed_citation_is_repaired_from_observed_source(doc_id):
    ctx = context("First evidence line\nSecond evidence line")
    claim = {"text": "First evidence line", "doc_id": doc_id}
    report = {"claims": [claim]}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert claim == {"text": "First evidence line", "doc_id": "source"}
    assert report["claims"] == [claim] and report["citations"] == ["source"]


def test_malformed_citation_without_source_does_not_crash_or_invent_citation():
    ctx = context("Only an observed snippet")
    claim = {"text": "Only an observed snippet", "doc_id": ["invalid"]}
    report = {"claims": [claim]}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert claim["doc_id"] == ["invalid"]
    assert report["citations"] == []


def test_single_line_substring_keeps_exact_text_and_provenance():
    ctx = context("First evidence line\nSecond evidence line")
    claim = {"text": "evidence line", "doc_id": "source"}
    report = {"claims": [claim]}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert report["claims"] == [claim]
    assert claim["text"] == "evidence line"
