import json
from tqdm import tqdm
from pathlib import Path
from typing import List
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_passage(entry : dict) -> str:
    """
    From wikipedia entry, build a passage: concats title, section and text if available
    """
    title = entry.get("title", "")
    section = entry.get("section", "")
    text = entry.get("text", "")
    return f"{title} {section}".strip() + " " + text.strip()

def chunk_sentence(sentence, chunk_size=500, chunk_overlap=0) -> List[str]:
    """
    Splits a sentence into smaller chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(sentence)
    return chunks

def load_wiki_data(file_path):
    """
    Loads the wiki data and returns the text.
    Args:
        file_path (str): Path to the JSONL file of atals wiki corpus
    """
    try:
        with open(file_path, 'r') as file:
            content = [json.loads(line) for line in file]
        return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

def load_embed_model(model_name: str, cache_dir: str = None):
    """
    Load the embedding model and tokenizer.
    Args:
        model_name (str): The name of the model to load.
        cache_dir (str): Directory to cache the model.
    Returns:
        model (AutoModel): The loaded embedding model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer

def create_chunked_wiki_data(file_path, save_path, chunk_size=500, chunk_overlap=0):
    """
    Loads the wiki data, chunks it and saves it to a file.
    Args:
        file_path (str): Path to the JSONL file of atals wiki corpus
        save_path (str): Path to save the chunked data
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
    
    usage:
        file_path = "/data/shyoon/rag_data/wiki/enwiki-dec2018/text-list-100-sec.jsonl"
        create_chunked_wiki_data(file_path, save_path="/data/shyoon/rag_data/wiki/enwiki-dec2018", chunk_size=500, chunk_overlap=0)
    """
    path = Path(save_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Creating directory: {path}")
    output_file = path / f"wiki_chunked_{chunk_size}_{chunk_overlap}.jsonl"

    if output_file.exists(): # if the file already exists, skip processing
        print(f"File already exists: {output_file}")
        return

    content = load_wiki_data(file_path)
    if not content:
        print("No content to process.")
        return
    
    all_chunks = []
    # Process the content
    for entry in tqdm(content):
        passage = build_passage(entry)
        chunks = chunk_sentence(passage)

        entry_chunks = []
        for chunk_id, chunk in enumerate(chunks):
            chunk_data = {}
            chunk_data['id'] = str(entry['id'])+'-'+str(chunk_id)
            chunk_data['text'] = chunk
            entry_chunks.append(chunk_data)
        all_chunks.extend(entry_chunks)
    
    data_dict = {item['id']: {'text':item['text']} for item in all_chunks}
    print(f"Saving {len(data_dict)} chunks to {output_file}")
    with open(output_file, "w") as f:
        for key, value in data_dict.items():
            json.dump({key: value}, f)
            f.write("\n")

    print(f"Saved all chunks to {output_file}")

def mean_pooling(token_embeddings, mask):
    """
    mean pooling for contriever embeddings
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1e-9)
    return sentence_embeddings

def simple_embed(texts : List[str], model : AutoModel, tokenizer : AutoTokenizer, device, batch_size : int = 32):
    all_embeddings = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_sentences = texts[i:i+batch_size]
            
            inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            cpu_embeddings = embeddings.cpu().numpy()

            assert len(batch_sentences) == cpu_embeddings.shape[0]

            all_embeddings[i:i+len(batch_sentences)] = cpu_embeddings
    return all_embeddings
