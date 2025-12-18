# argparse를 통해 openai generation 모델 지정 후 qa set을 generation하여 폴더에 저장
import argparse
import sys
path = "C:/Users/james/OneDrive/바탕 화면/대학교/수업/2025-2/산업텍스트애널리틱스/기말텀프/final_project"
sys.path.append(path)

from pipeline.common import setup_logger
from pipeline.generator.llm import OpenAIGenerator
from pipeline.generator.prompter import PromptGenerator
from pipeline.dataset.news import get_news_dataset

import random
import json

def main():
    parser = argparse.ArgumentParser(description="QA dataset generation using OpenAI model.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the OpenAI model.")
    parser.add_argument("--logger", type=bool, default=True, help="Enable or disable logger.")
    args = parser.parse_args()

    if args.logger:
        generator_logger = setup_logger(name="generator_logger", log_dir=path + "/logs")
    else:
        generator_logger = None
    
    # Load news dataset and sample 1000 documents
    print("Loading news dataset...")
    dataset = get_news_dataset()
    random.seed(42)
    random_indices = random.sample(range(len(dataset['text'])), 1000)
    docs = [dataset['text'][i] for i in random_indices]
    print("Sampling done. Number of documents sampled:", len(docs))

    # Generate prompts for the sampled documents
    prompts = PromptGenerator.generate_qa_pair(docs=docs)
    print("Prompt generation done. Number of prompts generated:", len(prompts))

    # Initialize OpenAIGenerator with specified model and logger
    generator = OpenAIGenerator(model_name=args.model_name, logger=generator_logger)
    print("Generator initialized with model:", args.model_name)

    # Generate QA pairs and organize results
    print("Starting QA pair generation...")
    results = generator.generate(prompts=prompts)
    print("QA pair generation done. Number of QA pairs generated:", len(results))
    result_dict = [{ "qid" : i, "docid": random_indices[i], "qa_pair": results[i]} for i in range(len(results))]

    # Save results to a json file
    print("Saving results to JSON file...")
    output_file = path + f"/qa/newsqa.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print("Results saved to", output_file)

if __name__ == "__main__":
    main()