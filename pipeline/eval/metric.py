import collections
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_acc(prediction : str, answers : list):
    for gt in answers:
        if gt in prediction:
            return 1
    return 0

# def calculate_em(prediction : str, answers : list):
#     for gt in answers:
#         if prediction == gt:
#             return 1
#     return 0

# def calculate_bleu(reference, hypothesis):
#     smoothie = SmoothingFunction().method4
#     return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

# def compute_f1_one(prediction: str, ground_truth: str) -> float:
#     """Compute F1 score between two strings."""
#     pred_tokens = normalize_answer(prediction).split()
#     gt_tokens = normalize_answer(ground_truth).split()

#     common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
#     num_same = sum(common.values())

#     if num_same == 0:
#         return 0.0

#     precision = num_same / len(pred_tokens)
#     recall = num_same / len(gt_tokens)
#     f1 = 2 * precision * recall / (precision + recall)
#     return f1

# def compute_f1(prediction: str, answers: list) -> float:
#     """Compute F1 score between prediction and a list of answers."""
#     f1_scores = [compute_f1_one(prediction, answer) for answer in answers]
#     return max(f1_scores)