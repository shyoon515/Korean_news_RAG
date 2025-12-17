import os
import pickle
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

import faiss
from llama_index.core.data_structs import Node
from llama_index.retrievers.bm25 import BM25Retriever
from transformers import AutoTokenizer, AutoModel
import Stemmer

import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.nn import DataParallel
import torch.nn.functional as F

from .embed import simple_embed, load_embed_model, mean_pooling

class Indexer(ABC):

    @abstractmethod
    def search(self, query_vectors: np.array, top_k: int) -> List[Tuple[List[object], List[float]]]:
        pass

def create_faiss_index(corpus, index_path, cuda_device : str, embed_model_name : str, hf_cache_dir : str, batch_size : int = 128):
    """
    Create a FAISS index from the given corpus.
    Args:
        corpus (list): List of documents to index. [{dict with 'id' and 'text' keys}, {}, {}, ...]
        index_path (str): Path to save the FAISS index.
        cuda_device (str): CUDA device to use. '0,1,2,3' for multiple GPUs.
    """
    folder_path = Path(index_path)
    index_path = folder_path / 'indexer'
    faiss_path = index_path / 'index.faiss'

    if not index_path.exists():
        index_path.mkdir(parents=True, exist_ok=True)
        print(f"Creating directory: {index_path}")

    # if faiss_path.exists(): # if the index.faiss file already exists, skip processing
    #     print(f"Index already exists: {faiss_path}")
    #     return

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    indexer = FaissDenseIndexer(embed_model_name=embed_model_name, cache_dir=hf_cache_dir, gpu_device_index=cuda_device)
    print(f"New indexer created, dimension: {indexer.embed_vec_dim}")

    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), batch_size)):
            batch_corpus = corpus[i:i+batch_size]
            batch_ids = [list(d.keys())[0] for d in batch_corpus]
            batch_sentences = [list(d.values())[0]['text'] for d in batch_corpus]
            assert len(batch_corpus) == len(batch_ids) == len(batch_sentences) # check if the lengths match
            cpu_embeddings = indexer._embed(batch_sentences)
            assert len(batch_ids) == cpu_embeddings.shape[0]
            indexer.index_data(ids=batch_ids, embeddings=cpu_embeddings)

    indexer.serialize(index_path)

class FaissDenseIndexer(Indexer):
    def __init__(self, embed_model_name, cache_dir, gpu_device_index=None, n_subquantizers=0, n_bits=8, logger=None):
        self.embed_model, self.embed_tokenizer = load_embed_model(model_name=embed_model_name, cache_dir=cache_dir)
        self.embed_vec_dim = self.embed_model.config.hidden_size

        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(self.embed_vec_dim, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(self.embed_vec_dim) # IndexFlatL2 -> euclidean distance, IndexFlatIP -> need normalization
            #self.index = faiss.IndexPreTransform(self.index)
        
        # if gpu_device_index:
        #     res = faiss.StandardGpuResources()
        #     # self.index = faiss.index_cpu_to_gpu(res, gpu_device_index, self.index)
        #     if isinstance(gpu_device_index, str):
        #         gpu_device_index = [int(i) for i in gpu_device_index.split(',')][0]
        #     self.index = faiss.index_cpu_to_gpu(res, gpu_device_index, self.index)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_model.to(self.device)
        self.embed_model.eval()
        self.index_id_to_db_id = []
        self.logger = logger

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float16')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        if self.logger:
            self.logger.info(f'Indexed {len(embeddings)} vectors with ids {ids}')
        else:
            # print(f'Indexed {len(embeddings)} vectors with ids {ids}')
            pass

    def search(self, query_text : str, top_k: int) -> List[Tuple[List[object], List[float]]]:
        query_vectors = self._embed(query_text).astype('float16')
        scores, indexes = self.index.search(query_vectors, top_k)
        db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def _embed(self, query_text : Union[str, List[str]]) -> np.array:
        if isinstance(query_text, str):
            query_text = [query_text]
        else:
            assert isinstance(query_text, list), "query_text must be a string or a list of strings"

        question_embeddings = simple_embed(query_text, self.embed_model, self.embed_tokenizer, self.device)
        return question_embeddings

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        
        if self.logger:
            self.logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')
        else:
            print(f'Serializing index to {index_file}, meta data to {meta_file}')
        index_to_save = faiss.index_gpu_to_cpu(self.index) if hasattr(faiss, "index_gpu_to_cpu") else self.index
        faiss.write_index(index_to_save, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)
        
        if self.logger:
            self.logger.info(f'Serialized index size {self.index.ntotal}')
        else:
            print(f'Serialized index size {self.index.ntotal}')

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        if self.logger:
            self.logger.info(f'Loading index from {index_file}, meta data from {meta_file}')
        else:
            print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        if self.logger:
            self.logger.info(f'Loaded index size {self.index.ntotal}')
        else:
            print(f'Loaded index size {self.index.ntotal}')

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        #new_ids = np.array(db_ids, dtype=np.int64)
        #self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)

class BM25Indexer(Indexer):
    def __init__(self, corpus, top_k):
        self.nodes = [Node(id_=docid, metadata={}, text=corpus[docid]['text']) for docid in corpus]
        self.retriever = BM25Retriever.from_defaults(
                            nodes=self.nodes,
                            similarity_top_k=top_k,
                            stemmer=Stemmer.Stemmer("english"),
                            language="english"
                        )
        self.top_k = top_k

    def search(self, query_text : str, top_k : int):
        if type(query_text) == str:
            result = self.retriever.retrieve(query_text)[:top_k]
            return ([i.id_ for i in result], np.array([i.score for i in result]))
        elif type(query_text) == list:
            return [self.search(text, top_k) for text in query_text]
        else:
            NotImplementedError("Input type not str nor list.")