import ast
import torch

import pytrec_eval
import re

def calculate_ndcg(docid_list, score_arr, qrels, qid, top_k):
    run = {qid: {k:float(v) for k, v in zip(docid_list, score_arr)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f'ndcg_cut_{top_k}'})
    eval_result = evaluator.evaluate(run)
    ndcg_score = eval_result[qid][f'ndcg_cut_{top_k}']
    return ndcg_score

def parse_output(text: str):
    try:
        think_retrieval = re.search(r"<think_retrieval>(.*?)</think_retrieval>", text, re.DOTALL).group(1).strip()
    except:
        think_retrieval = None
    try:
        retriever_policy = re.search(r"<(DENSE|SPARSE)>", text).group(1)
    except:
        retriever_policy = None
    try:
        think_qr = re.search(r"<think_qr>(.*?)</think_qr>", text, re.DOTALL).group(1).strip()
    except:
        think_qr = None
    try:
        qr = re.search(r"<QR>(.*?)</QR>", text, re.DOTALL).group(1).strip()
    except:
        qr = None

    return think_retrieval, retriever_policy, think_qr, qr

def build_reward_func(queries, qrels, dense_indexer, sparse_indexer, tokenizer, top_k=10, logger=None):
    
    def reward_func(qid, completion_ids, **kwargs):

        decoded_completions = [tokenizer.decode(ids, skip_special_tokens=False) for ids in completion_ids]
        
        parsed_output_list = [parse_output(completion) for completion in decoded_completions]
        _, retriever_policies, _, qrs = zip(*parsed_output_list)

        rewards = []
        qr_output_ndcgs = []
        original_output_ndcgs = []
        other_ret_output_ndcgs = []

        for i, (retriever_policy, qr) in enumerate(zip(retriever_policies, qrs)):

            query_id = qid[i]
            original_query = queries[query_id]

            if qr is None or retriever_policy is None:
                rewards.append(-2.0)  # Format error, give penalty
                qr_output_ndcgs.append(0.0)
                original_output_ndcgs.append(0.0)
                other_ret_output_ndcgs.append(0.0)
                continue

            if retriever_policy == 'DENSE':
                qr_output, original_output = dense_indexer.search([qr, original_query], top_k)
                other_ret_output = sparse_indexer.search(qr, top_k)

                qr_output_ndcg = calculate_ndcg(qr_output[0], qr_output[1], qrels, query_id, top_k)
                original_output_ndcg = calculate_ndcg(original_output[0], original_output[1], qrels, query_id, top_k)
                other_ret_output_ndcg = calculate_ndcg(other_ret_output[0], other_ret_output[1], qrels, query_id, top_k)
            elif retriever_policy == 'SPARSE':
                qr_output, original_output = sparse_indexer.search([qr, original_query], top_k)
                other_ret_output = dense_indexer.search(qr, top_k)[0]
                qr_output_ndcg = calculate_ndcg(qr_output[0], qr_output[1], qrels, query_id, top_k)
                original_output_ndcg = calculate_ndcg(original_output[0], original_output[1], qrels, query_id, top_k)
                other_ret_output_ndcg = calculate_ndcg(other_ret_output[0], other_ret_output[1], qrels, query_id, top_k)

            rewards.append(2 * qr_output_ndcg - max(original_output_ndcg, other_ret_output_ndcg))
            qr_output_ndcgs.append(qr_output_ndcg)
            original_output_ndcgs.append(original_output_ndcg)
            other_ret_output_ndcgs.append(other_ret_output_ndcg)
        
        if logger:
            logger.info("==== Reward Calculation ====")
            for i, reward in enumerate(rewards):
                text = f"Query ID: {qid[i]} | Original query: {queries[qid[i]]} | Reward: {reward:.3f} ({qr_output_ndcgs[i]:.3f}, {original_output_ndcgs[i]:.3f}, {other_ret_output_ndcgs[i]:.3f}) | Retriever Policy: {retriever_policies[i]} | QR: {qrs[i]}"
                logger.info(text)

        return rewards

    return reward_func