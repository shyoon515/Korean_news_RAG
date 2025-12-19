import re
import string
import json

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))) 

def answer_extractor(text: str) -> str:
    # if not string, convert to string
    if not isinstance(text, str):
        text = str(text)

    # JSON-like {"Answer": "..."} patterns
    answer_matches = re.findall(r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}', text)
    if answer_matches:
        return answer_matches[-1]

    # answer is: ~ (allowing linebreak after colon)
    cot_regex = re.compile(r"answer is[:\s]*([\s\S]*?)(?:\.|\n|$)", re.IGNORECASE | re.DOTALL)
    match = cot_regex.search(text)
    if match:
        return match.group(1).strip()

    # the answer to the question is: ~
    cot_regex = re.compile(r"the answer to the question is[:\s]*([\s\S]*?)(?:\.|\n|$)", re.IGNORECASE | re.DOTALL)
    match = cot_regex.search(text)
    if match:
        return match.group(1).strip()

    return text.strip() # if no match, return the original text

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records