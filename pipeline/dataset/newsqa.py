from datasets import Dataset
from pathlib import Path
import json

def load_news_qa_dataset(json_path = None) -> Dataset:
    """
    Loads news qa dataset from local file (/qa/newsqa.json).
    """
    if json_path is None:
        json_path = Path(__file__).resolve().parent.parent.parent / "qa" / "newsqa.json"
        print(f"Loading NewsQA dataset from {json_path}")
    
    try:
        newsqa = json.load(open(json_path, 'r'))
    except Exception as e:
        raise RuntimeError(f"Failed to load NewsQA dataset from {json_path}: {e}")
    
    parsed_newsqa = []

    for item in newsqa:
        qa = json.loads(item["qa_pair"])   # 🔥 핵심
        parsed_newsqa.append({
            "qid": item["qid"],
            "docid": item["docid"],
            "question": qa["question"],
            "answer": qa["answer"],
        })
    
    return parsed_newsqa