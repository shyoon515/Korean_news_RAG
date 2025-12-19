import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.eval.evaluator import LLMRelevanceEvaluator, AtkEvaluator
import json
from pipeline.common import setup_logger

logger = setup_logger(name="eval_retrieval")

atk_evaluator = AtkEvaluator()
llm_relevance_evaluator = LLMRelevanceEvaluator(model_name="qwen", logger=logger)

folder_path = "/workspace/final_project/outputs/releval"
all_file_names = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        all_file_names.append(os.path.join(folder_path, file_name))


for file_path in all_file_names:
    final_scores = {
        "atk" : None,
        "llm_relevance": None
    }
    final_scores["atk"] = atk_evaluator.evaluate(file_path)
    final_scores["llm_relevance"] = llm_relevance_evaluator.evaluate(file_path)

    # 평가 점수 결과를 json파일로 저장
    output_path = file_path.replace(".json", "_eval.json")
    
    with open(output_path, "w") as f:
        json.dump(final_scores, f, indent=4)