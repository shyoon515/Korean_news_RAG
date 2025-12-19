from pathlib import Path

from pyexpat import model
from typing import List, Union

from pipeline.retriever.embed import load_encoding_model
from pipeline.qdrant.client import QdrantService
from pipeline.generator.llm import OpenAIGenerator, VLLMGenerator
from pipeline.generator.prompter import PromptGenerator

class RAGChain:
    """
    Configuration for the RAG chain.
    """
    def __init__(
        self,
        encoder_name : str = 'bge', # 'bge', 'sbert', 'e5'
        collection_name : str = 'bge_1000_200',
        generator_name : str = 'midm', # exaone, midm, hyperclovax
        top_k : int = 5,
        generator_type : str = "vllm",
        vllm_api_base : str = "http://localhost:8000/v1",
        with_retrieval_results : bool = True,
        logger = None,
    ):
        self.encoder_name = encoder_name
        self.top_k = top_k
        self.collection_name = collection_name
        self.generator_name = generator_name
        self.generator_type = generator_type
        self.vllm_api_base = vllm_api_base
        self.with_retrieval_results = with_retrieval_results
        self.logger = logger

        self.encoder = self._load_encoding_model(encoder_name)
        self.qdrant_service = self._load_qdrant_service(self.logger)
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
  


    
    