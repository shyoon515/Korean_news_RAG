# argparse를 통해 컬렉션 생성 및 데이터 업서트 수행
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.retriever.embed import create_new_collection
from pipeline.qdrant.client import QdrantService
from pipeline.retriever.embed import load_encoding_model

def main():
    parser = argparse.ArgumentParser(description="Create Qdrant collection and upsert data.")
    parser.add_argument("--collection-name", type=str, required=True, help="Name of the Qdrant collection.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the embedding model.")
    parser.add_argument("--chunk-size", type=int, required=True, help="Chunk size for text splitting.")
    parser.add_argument("--chunk-overlap", type=int, required=True, help="Chunk overlap for text splitting.")
    args = parser.parse_args()

    if args.model_name == "bge-m3-ko":
        model_name = "dragonkue/bge-m3-ko"
    elif args.model_name == "kr-sbert":
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    
    collection_name = args.collection_name + "_" + str(args.chunk_size) + "_" + str(args.chunk_overlap)
    qdrant_service = QdrantService()

    # Load embedding model
    embed_model = load_encoding_model(model_name)

    create_new_collection(qdrant_service, collection_name, embed_model, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)


if __name__ == "__main__":
    main()