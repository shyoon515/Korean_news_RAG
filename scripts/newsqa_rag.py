# argparse를 통해 NewsQA 데이터셋 기반 QA 쌍 생성 수행
import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.chain.generation import RAGChain
from pipeline.common import setup_logger
from pipeline.dataset.newsqa import load_news_qa_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from NewsQA dataset.")
    parser.add_argument("--retrieval-type", type=str, default="hybrid", help="Retrieval type. Supported inputs: 'dense', 'sparse', 'hybrid'.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.6, help="Alpha value for hybrid retrieval (0~1). Only used if retrieval-type is 'hybrid'.")
    parser.add_argument("--encoder", type=str, default="bge", help="Encoder model name. Should be a sentence-transformers model. Supported inputs: 'bge', 'sbert', 'e5'.")
    parser.add_argument("--generator-type", type=str, default="vllm", help="Generator type. Supported inputs: 'vllm', 'openai'.")
    parser.add_argument("--generator", type=str, default="midm", help="Generator model name. Supported inputs for vllm: 'exaone', 'midm', 'hyperclovax'. For openai, use model names like 'gpt-4o-mini'.")
    parser.add_argument("--seq-type", type=str, default='dq', help="Determines the allocation of documents and query in the generation prompt: 'dq', 'qd'.")
    parser.add_argument("--doc-rearrange", type=int, default=1, help="Whether to rearrange documents in a 1 3...n 2 order before generation. Use 1 for True, 0 for False.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top documents to retrieve.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting context.")
    parser.add_argument("--overlap-size", type=int, default=200, help="Overlap size between chunks.")
    parser.add_argument("--output-path", type=str, default=str(Path(__file__).resolve().parent.parent / "outputs"), help="Folder path to save the generated QA pairs.")
    parser.add_argument("--logger", type=bool, default=True, help="Logger usage flag.")
    args = parser.parse_args()

    if args.logger:
        logger = setup_logger("NewsQAGeneration")

    rag = RAGChain(
        retrieval_type=args.retrieval_type,
        hybrid_alpha=args.hybrid_alpha,
        encoder_name=args.encoder,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        top_k=args.top_k,
        generator_type=args.generator_type,
        generator_name=args.generator,
        seq_type = args.seq_type,
        doc_rearrange = args.doc_rearrange,
        with_retrieval_results=True,
        logger=logger if args.logger else None,
    )

    # Load NewsQA dataset and generate QA pairs
    newsqa_data = load_news_qa_dataset()
    questions = [item["question"] for item in newsqa_data]

    # Generate answers
    answers = rag.ask(questions)

    # Save generated QA pairs
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"REARR{int(args.doc_rearrange)}_SEQ{args.seq_type}_TYPE{args.retrieval_type}_ALPHA{int(args.hybrid_alpha*100)}_GEN{args.generator}_ENC{args.encoder}_{args.chunk_size}_{args.overlap_size}_top{args.top_k}_generated_newsqa_qa_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        for item, answer in zip(newsqa_data, answers):
            output_item = {
                "qid": item["qid"],
                "docid": item["docid"],
                "question": item["question"],
                "generated_answer": answer[0],
                "retrieval_results": answer[1]
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    if args.logger:
        logger.info(f"(newsqa_generation) Generated QA pairs saved to {output_file}")

if __name__ == "__main__":
    main()
    