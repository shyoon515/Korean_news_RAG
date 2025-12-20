from copy import deepcopy

def minmax_map(results, score_key="score"):
    """results(list[dict]) -> {id: scaled_score}"""
    if not results:
        return {}

    scores = [r[score_key] for r in results]
    mn, mx = min(scores), max(scores)

    if mx == mn:
        return {r["id"]: 1.0 for r in results}

    return {r["id"]: (r[score_key] - mn) / (mx - mn) for r in results}


def hybrid_union_merge(dense_results, sparse_results, alpha: float):
    """
    dense_results, sparse_results: topK 결과(list[dict]) - id 집합이 달라도 됨
    alpha: dense 가중치 (0~1)
    return: union된 문서들을 final_score로 내림차순 정렬한 list[dict]
    """
    assert 0.0 <= alpha <= 1.0

    dense_scaled = minmax_map(dense_results)
    sparse_scaled = minmax_map(sparse_results)

    dense_map = {r["id"]: r for r in dense_results}
    sparse_map = {r["id"]: r for r in sparse_results}

    all_ids = set(dense_map) | set(sparse_map)

    merged = []
    for doc_id in all_ids:
        d_item = dense_map.get(doc_id)
        s_item = sparse_map.get(doc_id)

        d_score = dense_scaled.get(doc_id, 0.0)   # 없으면 0
        s_score = sparse_scaled.get(doc_id, 0.0)  # 없으면 0

        final = alpha * d_score + (1 - alpha) * s_score

        base = deepcopy(d_item if d_item is not None else s_item)

        base["dense_score_scaled"] = d_score
        base["sparse_score_scaled"] = s_score
        base["final_score"] = final
        base["in_dense_topk"] = d_item is not None
        base["in_sparse_topk"] = s_item is not None

        merged.append(base)

    merged.sort(key=lambda x: x["final_score"], reverse=True)
    return merged