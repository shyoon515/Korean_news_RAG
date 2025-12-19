import math
from collections import Counter
from typing import List, Dict, Union, Tuple

Example = Dict[str, object]  # {'qid': int, 'gt': int, 'retrieved': List[int]}

def dcg_at_k(rels: List[int], k: int) -> float:
    """rels: relevance list aligned with ranking positions (0/1/2/...)"""
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved: List[int], gt: int, k: int) -> float:
    """
    Binary relevance per position, but duplicates of gt are all counted as relevant
    """
    rels = [1 if docid == gt else 0 for docid in retrieved[:k]]
    dcg = dcg_at_k(rels, k)

    # number of relevant occurrences within top-k (since only those can contribute to DCG)
    r = sum(rels)
    if r == 0:
        return 0.0

    # ideal relevance list: r ones first
    ideal_rels = [1] * r + [0] * (k - r)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(
    retrieved: List[int],
    gt: int,
    k: int) -> float:
    """
    Recall@k with duplicate-aware hits.
    """
    hits = sum(1 for docid in retrieved[:k] if docid == gt)

    denom = sum(1 for docid in retrieved if docid == gt)
    # edge: if gt never appears anywhere, define recall as 0
    if denom == 0:
        return 0.0
    
    return min(1.0, hits / denom)

def evaluate_at_k(
    data,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Dict[int, float]]:
    """
    Returns mean metrics across queries:
    {
      "recall": {k: mean},
      "ndcg": {k: mean}
    }
    """
    out = {
        "recall": {k: 0.0 for k in ks},
        "ndcg": {k: 0.0 for k in ks},
    }

    n = len(data)
    for ex in data:
        gt = ex["gt"]
        retrieved = ex["retrieved"]

        for k in ks:
            out["recall"][k] += recall_at_k(retrieved, gt, k)
            out["ndcg"][k] += ndcg_at_k(retrieved, gt, k)

    for k in ks:
        out["recall"][k] /= n
        out["ndcg"][k] /= n

    return out


def dcg_at_k_from_rels(rels: List[float], k: int) -> float:
    """rels: relevance list aligned with ranking positions (0/1/2/...)"""
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k_from_rels(rels: List[float], k: int) -> float:
    """nDCG@k computed directly from per-position relevance list (0/1)."""
    dcg = dcg_at_k_from_rels(rels, k)

    # ideal: sort by relevance descending
    ideal_rels = sorted(rels[:k], reverse=True)
    idcg = dcg_at_k_from_rels(ideal_rels, k)

    if idcg <= 0:
        return 0.0
    return dcg / idcg

def recall_at_k_as_mean_relevance(rels: List[float], k: int) -> float:
    topk = rels[:k]
    if k <= 0:
        return 0.0
    return sum(1.0 for r in topk if r > 0) / k

def evaluate_rels_at_k(
    data,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Dict[int, float]]:
    """
    Input format:
      [{"qid": 0, "rel_scores": [0/1, ...]}, ...]
    Output format:
      {
        "recall": {k: mean},
        "ndcg": {k: mean}
      }
    """
    out = {
        "recall": {k: 0.0 for k in ks},
        "ndcg": {k: 0.0 for k in ks},
    }

    n = len(data)
    if n == 0:
        return out

    for ex in data:
        rels = ex["rel_scores"]
        for k in ks:
            out["recall"][k] += recall_at_k_as_mean_relevance(rels, k)
            out["ndcg"][k] += ndcg_at_k_from_rels(rels, k)

    for k in ks:
        out["recall"][k] /= n
        out["ndcg"][k] /= n

    return out