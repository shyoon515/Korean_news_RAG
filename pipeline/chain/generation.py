from pathlib import Path

from pyexpat import model
from typing import List, Union

from pipeline.retriever.embed import load_encoding_model
from pipeline.qdrant.client import QdrantService
from pipeline.generator.llm import OpenAIGenerator, VLLMGenerator
from pipeline.generator.prompter import PromptGenerator
from pipeline.retriever.hybrid import hybrid_union_merge

class RAGChain:
    """
    Configuration for the RAG chain.
    """
    def __init__(
        self,
        retrieval_type : str, # 'sparse', 'dense', 'hybrid'
        hybrid_alpha : float = 0.5,
        encoder_name : str = 'bge', # 'bge', 'sbert', 'e5'
        chunk_size : int = 1000,
        overlap_size : int = 200,
        generator_name : str = 'midm', # exaone, midm, hyperclovax
        top_k : int = 5,
        generator_type : str = "vllm",
        vllm_api_base : str = "http://localhost:8000/v1",
        with_retrieval_results : bool = True,
        logger = None,
    ):
        self.retrieval_type = retrieval_type
        self.hybrid_alpha = hybrid_alpha
        self.encoder_name = encoder_name
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.top_k = top_k
        self.collection_name = encoder_name + f"_{chunk_size}_{overlap_size}"
        self.generator_name = generator_name
        self.generator_type = generator_type
        self.vllm_api_base = vllm_api_base
        self.with_retrieval_results = with_retrieval_results
        self.logger = logger

        if self.retrieval_type in ['sparse', 'hybrid']:
            self.sparse_retriever = self._load_bm25_retriever()
        if self.retrieval_type in ['dense', 'hybrid']:
            self.encoder = self._load_encoding_model(encoder_name)
            self.qdrant_service = self._load_qdrant_service(self.logger)
        if self.retrieval_type not in ['sparse', 'dense', 'hybrid']:
            raise ValueError(f"Unsupported retrieval type: {self.retrieval_type}. It should be one of ['sparse', 'dense', 'hybrid']")
        
        self.generator = self._load_generator(self.generator_name, self.generator_type, self.logger)
    
    def ask(
        self,
        question : Union[str, List[str]],
    ) -> List[str]:

        # retrieve relevant documents
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Retrieving top {self.top_k} documents for the question, encoding {len(question) if isinstance(question, list) else 1} questions.")
        if isinstance(question, str):
            question = [question]

        all_retrieved_docs = self.retrieve(question)
        
        # generate answers using retrieved documents
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Generating answers using retrieved documents with generator: {self.generator_name}")
        final_answers = []
        prompts = []
        for q_idx, docs in enumerate(all_retrieved_docs):
            doc_texts = [doc['chunked_text'] for doc in docs]
            prompt = PromptGenerator.generate_answer_with_docs(docs=doc_texts, question=question[q_idx])
            prompts.append(prompt[0])
        final_answers = self.generator.generate(prompts)
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Answer generation completed.")
        
        if self.with_retrieval_results:
            return list(zip(final_answers, all_retrieved_docs))
        else:
            return final_answers
    
    def retrieve(self, question):
        """
        output should be:
        [{'id': ..., 'score': ..., 'chunked_text': ..., 'original_chunk_id': ..., 'original_docid': ...}, ...]
        """
        if self.retrieval_type == 'sparse':
            return self._sparse_retriever_search(question)
        elif self.retrieval_type == 'dense':
            return self._dense_retriever_search(question)
        elif self.retrieval_type == 'hybrid':
            return self._hybrid_retriever_search(question)
    
    def _dense_retriever_search(self, question):

        question_embeddings = self.encoder.encode(question)

        # for each question, retrieve top_k documents
        all_retrieved_docs = []
        for i, q_emb in enumerate(question_embeddings):
            retrieved_docs = self.qdrant_service.search(
                collection_name=self.collection_name,
                query_vector=q_emb,
                limit=self.top_k
            )
            all_retrieved_docs.append(retrieved_docs)
            if self.logger:
                self.logger.info(f"[RAGChain] (ask) Question ({i+1}/{len(question)}): Retrieved {len(retrieved_docs)} documents.")
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Document retrieval completed.")
        
        return all_retrieved_docs
    
    def _sparse_retriever_search(self, question):
        """
        output should be:
        [{'id': ..., 'score': ..., 'chunked_text': ..., 'original_chunk_id': ..., 'original_docid': ...}, ...]
        """
        results = []
        for i, q in enumerate(question):
            if self.logger:
                self.logger.info(f"[RAGChain] BM25 searching ... ({i+1}/{len(question)})")
            results.append(self.sparse_retriever.search(q))

        all_retrieved_docs = []
        for docids, scores in results:
            topk_docs = docids[:self.top_k]
            topk_scores = scores[:self.top_k].tolist()
            retrieved_docs = []
            for docid, score in zip(topk_docs, topk_scores):
                retrieved_doc = {
                    'id': docid,
                    'score': score,
                    'chunked_text': self.sparse_retriever.texts[self.sparse_retriever.doc_ids.index(docid)],
                    'original_chunk_id': docid % 1000,
                    'original_docid': docid // 1000
                }
                retrieved_docs.append(retrieved_doc)
            all_retrieved_docs.append(retrieved_docs)
        
        return all_retrieved_docs
    
    def _hybrid_retriever_search(self, question):
        if self.logger:
            self.logger.info(f"[RAGChain] Hybrid retriever - envoking both retrievers...")
        all_retrieved_docs_dense = self._dense_retriever_search(question)
        all_retrieved_docs_sparse = self._sparse_retriever_search(question)

        all_retrieved_docs = []
        for ds, ss in zip(all_retrieved_docs_dense, all_retrieved_docs_sparse):
            all_retrieved_docs.append(hybrid_union_merge(
                dense_results=ds,
                sparse_results=ss,
                alpha=self.hybrid_alpha
            )[:self.top_k])
        return all_retrieved_docs
        
    
    def _load_encoding_model(self, model_name : str):
        """
        Load the encoding model.
        Args:
            model_name (str): The name of the model to load.
        Returns:
            model (SentenceTransformer): The loaded encoding model.
        """
        if self.logger:
            self.logger.info(f"[RAGChain init] Loading embedding model: {model_name}")
        if model_name == "bge":
            _encoder = "dragonkue/bge-m3-ko"
        elif model_name == "sbert":
            _encoder = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        elif model_name == "e5":
            _encoder = "intfloat/multilingual-e5-large-instruct"
        else:
            raise ValueError(f"Unsupported encoder name: {model_name}. It should be one of ['bge', 'sbert', 'e5']")
        embed_model = load_encoding_model(_encoder)
        if self.logger:
            self.logger.info(f"[RAGChain init] Embedding model loaded.")
        return embed_model
    
    def _load_qdrant_service(self, logger):
        """
        Initialize Qdrant service.
        """
        if logger:
            logger.info(f"[RAGChain init] Initializing Qdrant service.")
        qdrant_service = QdrantService(logger=logger)
        
        if qdrant_service.collection_exists(self.collection_name):
            if logger:
                logger.info(f"[RAGChain init] Collection {self.collection_name} exists.")
        else:
            raise ValueError(f"Collection {self.collection_name} does not exist in Qdrant.")
        return qdrant_service
    
    def _load_bm25_retriever(self):

        from ..retriever.bm25 import KoreanBM25Indexer
        from ..dataset.news import get_news_dataset
        from ..retriever.embed import chunk_korean_sentence

        if self.logger:
            self.logger.info(f"[RAGChain init] loading datasets for BM25 corpus sets...")
        dataset = get_news_dataset()

        if self.logger:
            self.logger.info(f"[RAGChain init] loading BM25 retriever...")

        points = []

        for doc_idx, _ in enumerate(dataset['text']):
            
            chunk_texts = chunk_korean_sentence(dataset[doc_idx]['text'], chunk_size=self.chunk_size, chunk_overlap=self.overlap_size)
            
            for chunk_idx, chunk_text in enumerate(chunk_texts):
                point = {
                    "id": doc_idx*1000+chunk_idx,
                    'chunked_text' : chunk_text,
                    'original_docid' : doc_idx
                }
                points.append(point)
        bm25_indexer = KoreanBM25Indexer(corpus=points, top_k=100) # for normalization, set top_k to a large number
        if self.logger:
            self.logger.info(f"[RAGChain init] BM25 retriever loaded with {len(points)} documents.")
        return bm25_indexer
    
    def _load_generator(self, model_name : str, generator_type: str, logger):
        """
        Load the text generation model.
        Args:
            model_name (str): The name of the model to load.
            generator_type (str): Type of generator ('openai' or 'vllm')
            logger: Optional logger instance
        Returns:
            generator: The loaded text generation model (OpenAIGenerator or VLLMGenerator)
        """
        if logger:
            logger.info(f"[RAGChain init] Loading generator model: {model_name} (type: {generator_type})")
        
        if generator_type == "openai":
            generator = OpenAIGenerator(model_name, logger=logger)
        elif generator_type == "vllm":
            generator = VLLMGenerator(model_name, api_base=self.vllm_api_base, logger=logger)
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}. It should be one of ['openai', 'vllm']")
        
        if logger:
            logger.info(f"[RAGChain init] Generator model loaded.")
        return generator
  


    
    