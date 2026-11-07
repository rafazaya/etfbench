"""Promotion gates protect editorial approval and pinned evidence."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from etfbench.knowledge.promotion import PromotionQueue, content_hash, evidence_hash


@pytest.fixture
def queue_data():
    text = "Portfolio trading costs reduce NAV, outside the expense ratio."
    evidence = {"excerpt": text, "path": "bank/Participants", "section": "The fund"}
    return {
        "version": "batch-01-v1",
        "claims": [
            {
                "id": "ETF-0001",
                "section": "The fund",
                "facet": "trading costs",
                "regime": "US '40 Act",
                "text": text,
                "source_type": "sourced",
                "review_status": "reviewed",
                "reviewed_on": "2026-11-07",
                "reviewed_by": "source reviewer",
                "evidence_status": "resolved",
                "evidence": evidence,
                "content_hash": content_hash(text),
                "evidence_hash": evidence_hash(evidence),
            }
        ],
        "candidates": [
            {
                "golden": {
                    "id": "P01",
                    "input": "How do costs affect NAV?",
                    "expected_output": text,
                    "category": "capital_markets",
                    "claim_ids": ["ETF-0001"],
                    "claim_version": "batch-01-v1",
                },
                "tier": "A",
                "origin": "heldout/HO-01",
                "approval_status": "approved",
                "approved_by": "Rafa",
                "approved_on": "2026-11-07",
            }
        ],
    }


def test_sourced_and_vouched_have_same_release_gates(queue_data):
    sourced = PromotionQueue.model_validate(queue_data)
    assert sourced.release()[0].claim_ids == ["ETF-0001"]
    claim = queue_data["claims"][0]
    claim["source_type"] = "vouched"
    claim["evidence"] = {"excerpt": claim["text"], "vouched_by": "Rafa", "date": "2026-11-07"}
    claim["evidence_hash"] = evidence_hash(claim["evidence"])
    assert PromotionQueue.model_validate(queue_data).release() == sourced.release()
    claim["review_status"] = "pending"
    with pytest.raises(ValueError, match="not reviewed"):
        PromotionQueue.model_validate(queue_data).release()


@pytest.mark.parametrize(
    "field,value",
    [
        ("approval_status", "pending"),
        ("approved_by", None),
        ("approved_on", None),
        ("open_issues", ["source conflict"]),
    ],
)
def test_pair_approval_required(queue_data, field, value):
    queue_data["candidates"][0][field] = value
    with pytest.raises(ValueError, match="not approved"):
        PromotionQueue.model_validate(queue_data).release()


@pytest.mark.parametrize(
    "field,value",
    [
        ("review_status", "pending"),
        ("review_status", "rejected"),
        ("evidence_status", "unresolved"),
    ],
)
def test_claim_review_required(queue_data, field, value):
    queue_data["claims"][0][field] = value
    with pytest.raises(ValueError, match="not reviewed"):
        PromotionQueue.model_validate(queue_data).release()


@pytest.mark.parametrize("field", ["input", "expected_output"])
def test_blank_pair_rejected(queue_data, field):
    queue_data["candidates"][0]["golden"][field] = "   "
    with pytest.raises(ValueError, match="not approved"):
        PromotionQueue.model_validate(queue_data).release()


@pytest.mark.parametrize("field", ["text", "evidence"])
def test_tampered_hash_rejected(queue_data, field):
    queue_data["claims"][0][field] = "edited" if field == "text" else {"excerpt": "edited"}
    with pytest.raises(ValidationError, match="hash mismatch"):
        PromotionQueue.model_validate(queue_data)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_version", "no_claims"])
def test_invalid_links_rejected(queue_data, change):
    if change == "duplicate":
        queue_data["claims"] *= 2
    elif change == "wrong_version":
        queue_data["candidates"][0]["golden"]["claim_version"] = "other"
    else:
        queue_data["candidates"][0]["golden"]["claim_ids"] = (
            [] if change == "no_claims" else ["absent"]
        )
    with pytest.raises(ValidationError):
        PromotionQueue.model_validate(queue_data)


def test_change_detection_includes_evidence_and_missing_claims(queue_data):
    old = PromotionQueue.model_validate(queue_data)
    assert old.affected_goldens(old) == []
    for field in ["text", "evidence"]:
        data = json.loads(json.dumps(queue_data))
        claim = data["claims"][0]
        if field == "text":
            claim[field] += " Updated."
            claim["content_hash"] = content_hash(claim[field])
        else:
            claim[field]["section"] = "Revised source"
            claim["evidence_hash"] = evidence_hash(claim[field])
        new = PromotionQueue.model_validate(data)
        assert old.affected_goldens(new) == ["P01"]
    assert old.affected_goldens(PromotionQueue(version="v2", claims=[], candidates=[])) == ["P01"]


def test_checked_in_batch_is_pending():
    path = Path(__file__).parents[2] / "runs/promotion-batch-01/queue.json"
    queue = PromotionQueue.model_validate_json(path.read_text())
    assert len(queue.candidates) == 10
    assert all(candidate.approval_status == "pending" for candidate in queue.candidates)
    with pytest.raises(ValueError, match="not approved"):
        queue.release()


def test_release_revalidates_mutated_evidence(queue_data):
    queue = PromotionQueue.model_validate(queue_data)
    queue.claims[0].evidence["excerpt"] = "Changed after validation"
    with pytest.raises(ValidationError, match="hash mismatch"):
        queue.release()


@pytest.mark.parametrize("change", ["missing_reviewer", "missing_date", "unattributed_vouch"])
def test_provenance_required(queue_data, change):
    claim = queue_data["claims"][0]
    if change == "missing_reviewer":
        claim["reviewed_by"] = None
    elif change == "missing_date":
        claim["reviewed_on"] = None
    else:
        claim["source_type"] = "vouched"
    with pytest.raises(ValidationError):
        PromotionQueue.model_validate(queue_data)


@pytest.mark.parametrize(
    "folder",
    [
        "heldout-2026-11",
        "pilot-2026-11-06-calibration",
        "pilot-2026-11-06-strong",
    ],
)
def test_audit_covers_every_unanimous_answer_and_pins_inputs(folder):
    path = Path(__file__).parents[2] / "runs" / folder
    audit = json.loads((path / "agreed-answer-audit.json").read_text())
    graders = [
        {row["blind_id"]: row["grade"] for row in json.loads(file.read_text())}
        for name in ["s", "o", "l"]
        if (file := path / f"grades-{name}.json").exists()
    ]
    expected = {bid for bid in graders[0] if len({grader[bid] for grader in graders}) == 1}
    assert set(audit["agreed_answer_ids"]) == expected
    assert len(audit["agreed_answer_ids"]) == len(expected)
    for name, digest in audit["input_sha256"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
