"""Recorded source-based S adjudications after both independent grades were frozen."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RULE = "https://www.law.cornell.edu/cfr/text/17/270.6c-11"
IRISH = "https://www.centralbank.ie/docs/default-source/regulation/industry-market-sectors/funds/ucits/s-i-316-of-2026---central-bank-(supervision-and-enforcement)-act-2013-(section-48(1))-(undertakings-for-collective-investment-in-transferable-securities)-regul.pdf"
DECISIONS = {
    "ST008": (
        "o",
        (
            "The reference explicitly requires intermediary activity among observable "
            "indicators. Describing market makers in the mechanism does not recommend "
            "inspecting their activity. Retain half for point 2."
        ),
        "Reviewed QL-07 key point 2; IDC oversight paper",
    ),
    "ST035": (
        "s",
        (
            "A confidently asserted requirement to publish basket policies is a false "
            "regulatory side claim. Rule (c)(1) lists disclosures; (c)(3) requires "
            "policies, not website posting. R29 caps partial even though the PCF "
            "mechanism is right."
        ),
        RULE,
    ),
    "ST041": (
        "o",
        (
            "The iNAV-information phrase earns point 1 context credit, not point 3 "
            "credit: no positions/quantities, prospectus information statement or "
            "methodology/frequency is supplied. Only 1/3 total; below half."
        ),
        "Reviewed QL-34 key points; CBI Q&A 1012 and Regulation 88",
    ),
    "ST042": (
        "o",
        (
            "Haircuts are explicitly provided as general collateral-shortfall "
            "controls. Requiring repetition in the UCITS paragraph adds a location "
            "constraint absent from the rubric; the UCITS quality list supplies the "
            "other categories."
        ),
        "Reviewed QL-33 and ESMA paragraphs 30, 43–47",
    ),
    "ST051": (
        "o",
        (
            "Read the answer in context of AP-status entitlement. It denies the "
            "default pay entitlement and dealing duty, distinguishes LMM incentives "
            "and dealer earnings, and does not explicitly deny separate contracts. Do "
            "not manufacture a false universal rule."
        ),
        "Reviewed QL-12; IDC 2025 primer",
    ),
    "ST061": (
        "s",
        (
            "The frozen minimum expressly requires delivery as soon as practicable, "
            "not just by day 15. These are independent constraints; point 1 is an "
            "incomplete bundle."
        ),
        RULE + "#b",
    ),
    "ST063": (
        "o",
        (
            "Ordinarily responsible and contract-governed is equivalent qualified "
            "wording allowing exceptions; explicit waiver wording is not necessary. "
            "AP bears fees and recipients differ."
        ),
        "Reviewed transaction-fee reference point 2; Corgi SAI",
    ),
    "ST067": (
        "s",
        (
            "Regulation 90(4) requires disclosed and non-excessive costs; it does not "
            "impose an actual-cost/disclosed-ceiling rule. The added regulatory "
            "constraint is a material local falsehood under R29."
        ),
        IRISH + "#page=66",
    ),
    "ST072": (
        "s",
        (
            "Section 17(f) supplies the valid custody point, but citing the "
            "self-custody rule as an independent-custodian requirement is a false "
            "regulatory assertion, not an omission. R29 caps partial; no "
            "sponsor-separation core reversal. The tax blanket is an additional "
            "overstatement."
        ),
        "https://www.law.cornell.edu/cfr/text/17/270.17f-2",
    ),
    "ST075": (
        "o",
        (
            "NAV strike versus closing auction, stale and fair-valued inputs "
            "addresses contemporaneous pricing in substance. Together with trading "
            "and basket impediments it meets the bundle; no need for exact "
            "market-hours phrase. The truncated 2% fragment makes no rule."
        ),
        "Reviewed QL-31; SEC Release 33-10695",
    ),
    "ST076": (
        "o",
        (
            "The answer conveys ordinary secondary trading but does not say the "
            "warning must be in prospectus/marketing. The source-backed required "
            "warning minimum earns half; exception/cost protections otherwise "
            "complete."
        ),
        IRISH + "#page=65",
    ),
    "ST086": (
        "s",
        (
            "Net assets are after liabilities as well as costs. The claim that only "
            "worthless underlying assets can cause total loss is false when "
            "liabilities exhaust positive assets. Core net-proceeds mechanism "
            "survives; cap partial."
        ),
        "Reviewed QL-02; filed One + One liquidation supplement and SEC liquidation bulletin",
    ),
    "ST087": (
        "s",
        (
            "Non-pro-rata is not equivalent to non-representative. The first "
            "representative basket need not be pro-rata; rule definition "
            "distinguishes it from later differing representative baskets. R17 makes "
            "this local and partial, rather than wrong."
        ),
        RULE + "#a",
    ),
    "ST088": (
        "o",
        (
            "The minimum requires three risks and conditional US controls, not an "
            "additional indemnity carve-out. Conditional indemnification plus US "
            "collateral levels cover point 1; adding an unlisted requirement would "
            "change the instrument."
        ),
        "Reviewed QL-33 key point 1; iShares SAI",
    ),
}


def main():
    def read(name):
        return json.loads((ROOT / name).read_text())

    raw = read("grades-o-raw.json")
    bundle = {b["blind_id"]: b for b in read("blind-bundle.json")}
    normalized, log = [], []
    for original in raw:
        g = copy.deepcopy(original)
        required = {
            p["point"] for p in bundle[g["blind_id"]]["rubric"]["points"] if p["mark"] == "required"
        }
        removed = [p for p in g["coverage"] if p["point"] not in required]
        g["coverage"] = [p for p in g["coverage"] if p["point"] in required]
        if removed:
            log.append(dict(blind_id=g["blind_id"], removed_optional_coverage=removed))
        if g["blind_id"] == "ST035":
            log.append(
                dict(
                    blind_id="ST035",
                    nonmaterial_comment=g["local_error"],
                    reason=(
                        "O explicitly classifies this as nonmaterial. Clear material-error field "
                        "without changing O label; adjudication separately rejects this severity."
                    ),
                )
            )
            g["local_error"] = None
        normalized.append(g)
    f = ROOT / "grades-o.json"
    f.write_text(json.dumps(normalized, indent=2) + "\n")
    (ROOT / "grades-o.json.sha256").write_text(hashlib.sha256(f.read_bytes()).hexdigest() + "\n")
    (ROOT / "grades-o-raw.json.sha256").write_text(
        hashlib.sha256((ROOT / "grades-o-raw.json").read_bytes()).hexdigest() + "\n"
    )
    (ROOT / "normalization.json").write_text(json.dumps(log, indent=2) + "\n")
    s = {g["blind_id"]: g for g in read("grades-s.json")}
    o = {g["blind_id"]: g for g in normalized}
    assert set(DECISIONS) == {b for b in s if s[b]["grade"] != o[b]["grade"]}
    out = []
    lines = [
        "# Source-based disagreement resolutions",
        "",
        (
            "Both independent label files were frozen first. S adjudication is not a "
            "third independent grader. Sources are the reviewed reference text and "
            "frozen minimum; selected live source checks are recorded below."
        ),
        "",
    ]
    for bid, (winner, reason, source) in DECISIONS.items():
        g = copy.deepcopy((s if winner == "s" else o)[bid])
        g.update(adjudication=reason, source=source)
        out.append(g)
        lines.extend(
            [
                f"## {bid}: S {s[bid]['grade']}; O {o[bid]['grade']}; resolved {g['grade']}",
                "",
                reason,
                "",
                f"Source: {source}",
                "",
            ]
        )
    (ROOT / "resolutions.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "disagreements.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
