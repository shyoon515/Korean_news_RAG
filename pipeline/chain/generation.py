from pathlib import Path

from pyexpat import model
from typing import List, Union

from pipeline.retriever.embed import load_encoding_model
from pipeline.qdrant.client import QdrantService
from pipeline.generator.llm import OpenAIGenerator, VLLMGenerator
from pipeline.generator.prompter import PromptGenerator
from pipeline.common import VLLMServerManager

class RAGChain:
    """
    Configuration for the RAG chain.
    """
    def __init__(
        self,
        encoder_name : str = "bge",
        top_k : int = 5,
        collection_name : str = "bge_500_150",
        generator_name : str = "exaone",
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
        for q_emb in question_embeddings:
            retrieved_docs = self.qdrant_service.search(
                collection_name=self.collection_name,
                query_vector=q_emb,
                limit=self.top_k
            )
            all_retrieved_docs.append(retrieved_docs)
        if self.logger:
            self.logger.info(f"[RAGChain] (ask) Document retrieval completed.")
        
        # # start vllm server if using vllm generator
        # if self.generator_type == "vllm":
        #     if self.generator_name == "exaone":
        #         self.gen_model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        #     elif self.generator_name == "midm":
        #         self.gen_model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
        #     elif self.generator_name == "hyperclovax":
        #         self.gen_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        #     else:
        #         raise ValueError(f"Unsupported model_name: {self.generator_name}. Supported models are: exaone, midm, hyperclovax.")
        #     vllm_manager = VLLMServerManager(
        #         model=self.gen_model_name,
        #         host="localhost",
        #         port=8000,
        #         logger=self.logger
        #     )
        #     vllm_manager.start()

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
        
        # # stop vllm server if using vllm generator
        # if self.generator_type == "vllm":
        #     vllm_manager.stop()
        
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
        else:
            raise ValueError(f"Unsupported encoder name: {model_name}. It should be one of ['bge', 'sbert']")
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
  


    
    