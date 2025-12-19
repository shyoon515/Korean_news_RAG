from typing import List, Tuple, Dict

from .utils import answer_extractor, load_jsonl
from .metric import evaluate_at_k, evaluate_rels_at_k

from pipeline.generator.llm import OpenAIGenerator, VLLMGenerator
from pipeline.generator.prompter import PromptGenerator
from pipeline.dataset.newsqa import load_news_qa_dataset

class AtkEvaluator:
    def __init__(self, ks: set = {5}):
        self.ks = ks
        self.qrels = None
        self.results = None
        self.score = None

    def evaluate(self, output_file_path) -> Dict[str, float]:

        results = load_jsonl(output_file_path)
        self.results = results
        qrels = self._extract_qrel_prediction(results)
        self.qrels = qrels

        metrics = evaluate_at_k(qrels, ks=self.ks)
        self.score = metrics
        return metrics

    def _extract_qrel_prediction(self,
        results: list[dict],
    ) -> list[dict]:
        """
        Extract qrel and prediction from the retrieval results.
        """
        retrieval_result = []

        for record_dict in results:
            relevance_pair = {
                'qid' : record_dict['qid'],
                'gt' : record_dict['docid'],
                'retrieved' : [doc_dict['original_docid'] for doc_dict in record_dict['retrieval_results']]
            }
            retrieval_result.append(relevance_pair)
        return retrieval_result

class LLMRelevanceEvaluator:
    def __init__(self, ks:set = {5}, model_name: str = "qwen", logger = None):
        self.ks = ks
        self.logger = logger
        if model_name.startswith("gpt"):
            self.generator = OpenAIGenerator(model_name=model_name, logger=logger)
        else:
            self.generator = VLLMGenerator(model_name=model_name, logger=logger)

        self.rel_scores = None
        self.results = None
        self.score = None

    def evaluate(self, output_file_path) -> Dict[str, float]:
        results = load_jsonl(output_file_path)
        self.results = results
        prompts_by_qids = {}
        for record_dict in results:
            prompts = []

            for ret_result in record_dict['retrieval_results']:
                prompt = PromptGenerator.generate_relevance_judge(doc=record_dict['question'], question=ret_result['chunked_text'])
                prompts.append(prompt)

            prompts_by_qids[record_dict['qid']] = prompts
        
        rel_scores = []

        for qid, prompts in prompts_by_qids.items():
            scores = self.generator.generate(prompts, temperature=0)
            scores_parsed = []

            for score in scores:
                try:
                    score = float(score.strip())
                except:
                    print(score)
                    score = 0.0  # default to 0.0 if parsing fails
                scores_parsed.append(score)
            
            rel_score = {
                'qid': qid,
                'rel_scores': scores_parsed
            }
            rel_scores.append(rel_score)
        
        self.rel_scores = rel_scores
        metrics = evaluate_rels_at_k(rel_scores, ks=self.ks)
        self.score = metrics
        return metrics

class LLMGenerationEvaluator:
    def __init__(self, model_name: str = "qwen", logger = None):
        self.logger = logger
        if model_name.startswith("gpt"):
            self.generator = OpenAIGenerator(model_name=model_name, logger=logger)
        else:
            self.generator = VLLMGenerator(model_name=model_name, logger=logger)
        self.newsqa = load_news_qa_dataset()
        self.generation_scores = None
        self.results = None
        self.score = None
    
    def evaluate(self, output_file_path) -> List[dict]:
        results = load_jsonl(output_file_path)
        self.results = results

        prompts = []
        for qid, newsqa_dict in enumerate(self.newsqa):
            assert newsqa_dict['qid'] == results[qid]['qid']

            parsed_answer = answer_extractor(results[qid]['generated_answer'])

            prompt = PromptGenerator.generate_generation_judge(
                question = results[qid]['question'],
                answer = newsqa_dict['answer'],
                prediction = parsed_answer
                )
            prompts.append(prompt)
        
        generation_scores = []

        for qid, prompt in enumerate(prompts):
            score = self.generator.generate([prompt], temperature=0)[0]
            try:
                score = float(score.strip())
            except:
                score = 0.0  # default to 0.0 if parsing fails
            
            generation_score = {
                'qid': qid,
                'generation_score': score
            }
            generation_scores.append(generation_score)

            # if qid == 2: # for testing, remove this line for full evaluation
            #     break

        self.generation_scores = generation_scores

        hit = 0
        for scores in generation_scores:
            hit+=scores['generation_score']
        self.score = hit/len(generation_scores)
        return self.score

