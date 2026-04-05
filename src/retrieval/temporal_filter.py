"""Temporal filtering — estimate publication year and flag outdated evidence.

Uses PubMed ID (PMID) ranges to estimate publication year, since PubMedQA
doesn't include dates directly. PMIDs are assigned sequentially, so higher
IDs correspond to more recent publications.

Two filtering modes:
  - flag:   adds 'is_outdated' and 'estimated_year' metadata to each result
  - filter: removes outdated results entirely
"""

import re
from typing import List, Dict, Any, Optional


# ── PMID-to-Year Estimation ───────────────────────────────────

# Known PMID milestones (approximate ranges based on PubMed history)
_PMID_YEAR_TABLE = [
    (1_000_000,  1978),
    (3_000_000,  1988),
    (5_000_000,  1993),
    (8_000_000,  1997),
    (10_000_000, 2000),
    (12_000_000, 2002),
    (14_000_000, 2004),
    (16_000_000, 2006),
    (18_000_000, 2008),
    (20_000_000, 2010),
    (22_000_000, 2012),
    (24_000_000, 2014),
    (26_000_000, 2016),
    (28_000_000, 2018),
    (30_000_000, 2020),
    (33_000_000, 2022),
    (36_000_000, 2024),
]


def estimate_year_from_pmid(pmid: str) -> Optional[int]:
    """Estimate publication year from a PubMed ID using known PMID ranges.

    Args:
        pmid: PubMed ID string (e.g. '21645374').

    Returns:
        Estimated year (int), or None if PMID is not numeric.
    """
    try:
        pmid_int = int(pmid)
    except (ValueError, TypeError):
        return None

    # Walk the table to find the bracket
    year = 1975  # default for very old PMIDs
    for threshold, y in _PMID_YEAR_TABLE:
        if pmid_int >= threshold:
            year = y
        else:
            break

    return year


def extract_year_from_text(text: str) -> Optional[int]:
    """Try to extract a 4-digit publication year from evidence text.

    Looks for patterns like (2015), published in 2018, etc.
    Returns the most recent plausible year found, or None.
    """
    matches = re.findall(r'\b(19[89]\d|20[0-2]\d)\b', text)
    if not matches:
        return None
    years = [int(y) for y in matches]
    return max(years)


def estimate_year(doc: Dict[str, Any]) -> Optional[int]:
    """Estimate publication year from a document using best available signal.

    Priority: existing year field > text extraction > PMID estimation.
    """
    # 1. Check if year is already set
    existing = doc.get("year")
    if existing is not None:
        try:
            return int(existing)
        except (ValueError, TypeError):
            pass

    # 2. Try text extraction
    text = doc.get("retrieval_text", "") or doc.get("text", "")
    text_year = extract_year_from_text(text)
    if text_year:
        return text_year

    # 3. Fall back to PMID estimation
    pmid = doc.get("id", "")
    return estimate_year_from_pmid(pmid)


# ── Temporal Filtering ─────────────────────────────────────────


def add_temporal_metadata(
    documents: List[Dict[str, Any]],
    recency_threshold: int = 2010,
) -> List[Dict[str, Any]]:
    """Add year estimation and recency flag to each document.

    Args:
        documents: List of retrieval documents from chunk_documents().
        recency_threshold: Year cutoff; documents before this are 'outdated'.

    Returns:
        Same documents list with 'estimated_year' and 'is_outdated' added.
    """
    for doc in documents:
        year = estimate_year(doc)
        doc["estimated_year"] = year
        doc["year"] = year
        doc["is_outdated"] = (year is not None and year < recency_threshold)

    return documents


def filter_evidence_by_recency(
    results: List[Dict[str, Any]],
    recency_threshold: int = 2010,
    mode: str = "flag",
) -> List[Dict[str, Any]]:
    """Apply temporal filtering to retrieved evidence.

    Args:
        results: Retrieved evidence dicts from the retrieval pipeline.
        recency_threshold: Year cutoff for outdated evidence.
        mode: 'flag' = add metadata only, 'filter' = remove outdated,
              'downweight' = reduce score of outdated evidence.

    Returns:
        Filtered/flagged evidence list.
    """
    for item in results:
        # Estimate year from the evidence item
        doc_id = item.get("id", "")
        text = item.get("text", "")

        year = extract_year_from_text(text)
        if year is None:
            year = estimate_year_from_pmid(doc_id)

        item["estimated_year"] = year
        item["is_outdated"] = (year is not None and year < recency_threshold)

    if mode == "filter":
        return [item for item in results if not item.get("is_outdated", False)]

    if mode == "downweight":
        for item in results:
            if item.get("is_outdated", False):
                # Reduce rerank score by 50% for outdated evidence
                if "rerank_score" in item:
                    item["rerank_score"] *= 0.5
                if "hybrid_score" in item:
                    item["hybrid_score"] *= 0.5

    return results


def compute_temporal_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute temporal statistics for a set of retrieved evidence.

    Returns:
        Dict with counts and fraction of outdated evidence.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "outdated": 0, "recent": 0, "outdated_fraction": 0.0}

    outdated = sum(1 for r in results if r.get("is_outdated", False))
    return {
        "total": total,
        "outdated": outdated,
        "recent": total - outdated,
        "outdated_fraction": outdated / total,
        "years": [r.get("estimated_year") for r in results],
    }
