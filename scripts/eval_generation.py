import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.eval.evaluator import LLMGenerationEvaluator
import json
from pipeline.common import setup_logger

logger = setup_logger(name="eval_generation")

llm_generation_evaluator = LLMGenerationEvaluator(model_name="qwen", logger=logger)

folder_path = "/workspace/final_project/outputs/geneval"
all_file_names = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        all_file_names.append(os.path.join(folder_path, file_name))

for file_path in all_file_names:
    final_score = llm_generation_evaluator.evaluate(file_path)

    # 평가 점수 결과를 json파일로 저장
    output_path = file_path.replace(".json", "_eval.json")
    
    with open(output_path, "w") as f:
        json.dump(final_score, f, indent=4)