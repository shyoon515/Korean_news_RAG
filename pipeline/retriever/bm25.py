from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
import numpy as np

from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi


class Indexer(ABC):
    @abstractmethod
    def search(self, query_text: Any, top_k: int):
        pass

class KoreanBM25Indexer(Indexer):
    def __init__(self, corpus: List[Dict[str, Any]], top_k: int = 5):
        """
        corpus format:
        [
          {
            "id": chunk_id,
            "chunked_text": "...",
            "original_docid": ...
          },
          ...
        ]
        """
        self.top_k = top_k

        self.doc_ids = [item["id"] for item in corpus]
        self.texts = [item["chunked_text"] for item in corpus]

        # 한국어 형태소 토크나이저
        self.kiwi = Kiwi()

        # 문서 토크나이즈
        tokenized_corpus = [self._tokenize(text) for text in self.texts]

        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # 형태소 단위 토큰
        return [tok.form for tok in self.kiwi.tokenize(text)]

    def search(self, query_text: Any, top_k: int = None):
        if top_k is None:
            top_k = self.top_k

        if isinstance(query_text, str):
            q_tokens = self._tokenize(query_text)
            scores = np.asarray(self.bm25.get_scores(q_tokens))

            top_idx = np.argsort(scores)[::-1][:top_k]
            doc_ids = [self.doc_ids[i] for i in top_idx]
            top_scores = scores[top_idx]

            return doc_ids, top_scores

        elif isinstance(query_text, list):
            return [self.search(q, top_k) for q in query_text]

        else:
            raise NotImplementedError("Input type not str nor list.")