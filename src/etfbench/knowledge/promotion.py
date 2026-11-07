"""Validate reviewed promotion snapshots; build releases without writing files."""

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from etfbench.datasets.schema import ETFGolden


def content_hash(text: str) -> str:
    """SHA-256 of exact UTF-8 text, without whitespace normalization."""
    return hashlib.sha256(text.encode()).hexdigest()


def evidence_hash(evidence: dict[str, str]) -> str:
    """SHA-256 of canonical JSON evidence, including attribution and locators."""
    return content_hash(
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


class Claim(BaseModel):
    """A stable editorial identity and immutable version of claim and evidence."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    section: str = Field(min_length=1)
    facet: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_type: Literal["sourced", "vouched"]
    review_status: Literal["pending", "reviewed", "rejected"] = "pending"
    reviewed_on: date | None = None
    reviewed_by: str | None = None
    evidence_status: Literal["resolved", "unresolved"] = "unresolved"
    evidence: dict[str, str]
    content_hash: str
    evidence_hash: str

    @model_validator(mode="after")
    def validate_version(self) -> "Claim":
        if (
            not self.text.strip()
            or not self.evidence
            or any(not value.strip() for value in self.evidence.values())
        ):
            raise ValueError("Claim text and evidence must be nonblank")
        if self.content_hash != content_hash(self.text):
            raise ValueError("Claim content hash mismatch")
        if self.evidence_hash != evidence_hash(self.evidence):
            raise ValueError("Claim evidence hash mismatch")
        if self.source_type == "vouched" and not self.evidence.get("vouched_by"):
            raise ValueError("Vouched evidence must name its practitioner")
        if self.review_status == "reviewed" and not (self.reviewed_on and self.reviewed_by):
            raise ValueError("Reviewed claims need reviewer and date")
        return self


class Candidate(BaseModel):
    """A proposed golden; source approval never implies pair approval."""

    model_config = ConfigDict(extra="forbid")
    golden: ETFGolden
    tier: Literal["A", "B", "C"]
    origin: str
    approval_status: Literal["pending", "approved", "rejected"] = "pending"
    approved_by: Literal["Rafa"] | None = None
    approved_on: date | None = None
    open_issues: list[str] = []


class PromotionQueue(BaseModel):
    """Versioned claim snapshot and candidate pairs, outside data/goldens."""

    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1)
    claims: list[Claim]
    candidates: list[Candidate]

    @model_validator(mode="after")
    def validate_links(self) -> "PromotionQueue":
        ids = {claim.id for claim in self.claims}
        golden_ids = [candidate.golden.id for candidate in self.candidates]
        if len(ids) != len(self.claims) or len(set(golden_ids)) != len(golden_ids):
            raise ValueError("Duplicate claim or golden ID")
        for candidate in self.candidates:
            golden = candidate.golden
            if golden.claim_version != self.version or not golden.claim_ids:
                raise ValueError("Golden must reference this snapshot and at least one claim")
            if not set(golden.claim_ids) <= ids:
                raise ValueError("Unknown claim ID")
        return self

    def release(self) -> list[ETFGolden]:
        """Return eligible goldens only; reject the whole batch on any open gate."""
        snapshot = type(self).model_validate(self.model_dump())
        claims = {claim.id: claim for claim in snapshot.claims}
        for candidate in snapshot.candidates:
            golden = candidate.golden
            if (
                candidate.approval_status != "approved"
                or candidate.approved_by != "Rafa"
                or not candidate.approved_on
                or candidate.open_issues
                or not golden.input.strip()
                or not golden.expected_output.strip()
            ):
                raise ValueError(f"Pair not approved or unresolved: {golden.id}")
            for claim_id in golden.claim_ids:
                claim = claims[claim_id]
                if claim.review_status != "reviewed" or claim.evidence_status != "resolved":
                    raise ValueError(f"Claim not reviewed or unresolved: {claim_id}")
        return [candidate.golden.model_copy(deep=True) for candidate in snapshot.candidates]

    def affected_goldens(self, newer: "PromotionQueue") -> list[str]:
        """Flag dependencies whose exact text or evidence changed or disappeared."""
        old = {claim.id: (claim.content_hash, claim.evidence_hash) for claim in self.claims}
        new = {claim.id: (claim.content_hash, claim.evidence_hash) for claim in newer.claims}
        return [
            candidate.golden.id
            for candidate in self.candidates
            if any(old[cid] != new.get(cid) for cid in candidate.golden.claim_ids)
        ]
