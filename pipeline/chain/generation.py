from pathlib import Path

from pyexpat import model
from typing import List, Union

from pipeline.retriever.embed import load_encoding_model
from pipeline.qdrant.client import QdrantService
from pipeline.generator.llm import OpenAIGenerator
from pipeline.generator.prompter import PromptGenerator

class RAGChain:
    """
    Configuration for the RAG chain.
    """
    def __init__(
        self,
        encoder_name : str = "bge-m3-ko",
        top_k : int = 5,
        collection_name : str = "bge_500_150",
        generator_name : str = "gpt-4o-mini",
        with_retrieval_results : bool = True,
        logger = None,
    ):
        self.encoder_name = encoder_name
        self.top_k = top_k
        self.collection_name = collection_name
        self.generator_name = generator_name
        self.with_retrieval_results = with_retrieval_results
        self.logger = logger

        self.encoder = self._load_encoding_model(encoder_name)
        self.qdrant_service = self._load_qdrant_service(self.logger)
        self.generator = self._load_generator(self.generator_name, self.logger)
    
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
        for q_emb in question_embeddings:
            retrieved_docs = self.qdrant_service.search(
                collection_name=self.collection_name,
                query_vector=q_emb,
                limit=self.top_k
            )
            all_retrieved_docs.append(retrieved_docs)
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Document retrieval completed.")

        # generate answers using retrieved documents
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Generating answers using retrieved documents with generator: {self.generator_name}")
        final_answers = []
        for q_idx, docs in enumerate(all_retrieved_docs):
            doc_texts = [doc['chunked_text'] for doc in docs]
            prompts = PromptGenerator.generate_answer_with_docs(docs=doc_texts, question=question[q_idx])
            generation_response = self.generator.generate(prompts)
            final_answers.append(generation_response[0])
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
        if model_name == "bge-m3-ko":
            _encoder = "dragonkue/bge-m3-ko"
        elif model_name == "kr-sbert":
            _encoder = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        else:
            raise ValueError(f"Unsupported encoder name: {model_name}. It should be one of ['bge-m3-ko', 'kr-sbert']")
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
    
    def _load_generator(self, model_name : str, logger):
        """
        Load the text generation model.
        Args:
            model_name (str): The name of the model to load.
        Returns:
            generator (OpenAIGenerator): The loaded text generation model.
        """
        if logger:
            logger.info(f"[RAGChain init] Loading generator model: {model_name}")
        generator = OpenAIGenerator(model_name, logger=logger)
        if logger:
            logger.info(f"[RAGChain init] Generator model loaded.")
        return generator
  


    
    