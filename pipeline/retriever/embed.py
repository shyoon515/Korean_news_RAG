from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pipeline.qdrant.client import QdrantService
from pipeline.dataset.news import get_news_dataset
import torch
from typing import List
from tqdm import tqdm

def load_encoding_model(model_name : str):
    """
    Load the encoding model.
    Args:
        model_name (str): The name of the model to load.
    Returns:
        model (SentenceTransformer): The loaded encoding model.
    """
    model = SentenceTransformer(model_name)
    print(f"Model is on device: {model.device}")
    return model

def chunk_korean_sentence(sentence, chunk_size=500, chunk_overlap=0) -> List[str]:
    """
    Splits a sentence into smaller chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "다. ",
            "요. ",
            ". ",
            "!",
            "?",
        ],
        length_function=len,
        keep_separator="end"
    )
    chunks = text_splitter.split_text(sentence)
    return chunks

def simple_encode(texts : List[str], model : SentenceTransformer, batch_size : int = 32):
    """
    Encode a list of texts using the provided embedding model in batches.
    Args:
        texts (List[str]): List of texts to encode.
        model (SentenceTransformer): The embedding model.
        batch_size (int): The batch size for encoding.
    Returns:
        embeddings (List[List[float]]): List of embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings.tolist())
    return embeddings

def organize_chunks_for_upsert(dataset, encoding_model : SentenceTransformer, chunk_size : int, chunk_overlap : int = 0, batch_size : int = 32, index_from : int = 0, index_to : int = -1) -> List[dict]:
    """
    Organize chunks for upsert into Qdrant.
    Args:
        chunks (List[dict]): List of chunk dictionaries with 'id' and 'text'.
        encoding_model (SentenceTransformer): The encoding model.
        batch_size (int): The batch size for encoding.
    Returns:
        points (List[dict]): List of points ready for upsert.
    """
    points = []
    
    for doc_idx, _ in enumerate(dataset['text']):
        if index_to != -1 and doc_idx >= index_to:
            break
        if doc_idx < index_from:
            continue
        
        chunk_texts = chunk_korean_sentence(dataset[doc_idx]['text'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = simple_encode(chunk_texts, encoding_model, batch_size=batch_size)
        
        for chunk_idx, chunk_text in enumerate(chunk_texts):
            point = {
                "id": f"{doc_idx}-{chunk_idx}",
                "vector": embeddings[chunk_idx],
                "payload": {
                    'category': dataset[doc_idx]['category'],
                    'press' : dataset[doc_idx]['press'],
                    'title' :dataset[doc_idx]['title'],
                    'document' : dataset[doc_idx]['document'],
                    'link' : dataset[doc_idx]['link'],
                    'summary' : dataset[doc_idx]['summary'],
                    'bucket' : dataset[doc_idx]['bucket'],
                    'text' : dataset[doc_idx]['text'],
                    'chunked_text' : chunk_text,
                    'original_docid' : doc_idx
                }
            }
            points.append(point)
    return points

def create_new_collection(qdrant_service : QdrantService, collection_name : str, embed_model : SentenceTransformer, chunk_size=500, chunk_overlap=0):
    """
    Create new collection in qdrant for korean news dataset
    Args:
        qdrant_service (QdrantService): The Qdrant service instance.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    Returns:
        None
    """
    # check if collection exists, if not, create it with appropriate vector size
    if qdrant_service.collection_exists(collection_name) == False:
        print(f"Creating new collection: {collection_name}")
        created = qdrant_service.ensure_collection(
            collection_name=collection_name,
            vector_size=embed_model.get_sentence_embedding_dimension())
        if created:
            print(f"Collection {collection_name} created.")
        else:
            raise ValueError(f"Failed to create collection: {collection_name}")
    else:
        print(f"Collection {collection_name} already exists.")
        return 0

    # organize data for upsert, in batches
    dataset = get_news_dataset()

    for i in range(0, len(dataset), 100):
        points = organize_chunks_for_upsert(dataset, embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap, batch_size=16, index_from=i, index_to=min(i+100, len(dataset)))
        qdrant_service.upsert_points(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted points for documents {i} to {min(i+100, len(dataset))}")
    
    print("All points upserted successfully, collection name:", collection_name)
