from typing import List, Union

"""
Formatting functions for prompts.
"""
class PromptGenerator:

    @staticmethod
    def generate_qa_pair(
            docs: Union[str, List[str]]
        ) -> List[str]:

        prompts = []
        
        if isinstance(docs, str):
            PROMPT = f"""
            다음 기사의 내용에만 근거해서, RAG 평가 데이터셋으로 쓸 질문-답 쌍을 하나 만들어줘. 질문을 구체적이어야 하고, 답변의 길이는 20자 이내로 만들어줘.
            반드시 기사의 내용에만 근거해서 질문과 답변을 만들어야 해. 기사의 맥락을 이해해야만 답할 수 있는 질문이어야 하고, 가능하면 언제, 어디서, 누구 등의 구체적인 정보를 포함하는 질문이면 좋아.
            '기사에 따르면', '기사 내용 중' 등의 표현은 사용하지 말아줘.
            출력 형식은 json으로 다음과 같아야 해.
            {{"question": "질문 내용", "answer": "답변 내용"}}

            아래가 기사의 내용이야.
            기사 내용: {docs}
            출력: 
            """
            prompts.append(PROMPT)

        elif isinstance(docs, list):
            for i, d in enumerate(docs):
                if isinstance(docs[i], list):
                    raise Exception("docs must be a list of strings, not a list of lists")
                PROMPT = f"""
                다음 기사의 내용에만 근거해서, RAG 평가 데이터셋으로 쓸 질문-답 쌍을 하나 만들어줘. 질문을 구체적이어야 하고, 답변의 길이는 20자 이내로 만들어줘.
                반드시 기사의 내용에만 근거해서 질문과 답변을 만들어야 해. 기사의 맥락을 이해해야만 답할 수 있는 질문이어야 하고, 가능하면 언제, 어디서, 누구 등의 구체적인 정보를 포함하는 질문이면 좋아.
                '기사에 따르면', '기사 내용 중' 등의 표현은 사용하지 말아줘.
                출력 형식은 json으로 다음과 같아야 해.
                {{"question": "질문 내용", "answer": "답변 내용"}}

                아래가 기사의 내용이야.
                기사 내용: {docs[i]}
                출력: 
                """
                prompts.append(PROMPT)
        return prompts

    def generate_answer_with_docs(
            docs: List[str], 
            question: str
        ) -> List[str]:
        """
        Generate prompts for answering a question based on provided documents.
        Args:
            docs (List[str]): Document(s) to base the answer on.
            question (str): The question to be answered.
        Returns:
            List[str]: List of generated prompts.
        """

        prompts = []
        
        PROMPT = f"""
        다음 기사들의 내용에만 근거해서, 아래 질문에 대한 답변을 해줘.
        답변은 간결하게 해주고, 출력 형식은 json으로 다음과 같아야 해.
        {{"answer": "답변 내용"}}
        반드시 기사 내용에만 근거해서 답변을 만들어야 해.
        """
        for i, doc in enumerate(docs):
            PROMPT += f"\n기사 {i+1} 내용: {doc}\n"
        PROMPT += f"\n질문: {question}\n답변: "

        prompts.append(PROMPT)
        return prompts
    
    def generate_relevance_judge(
            doc: str, 
            question: str
        ) -> str:
        """
        Generate a prompt for judging the relevance of a document to a question-answer pair.
        Args:
            doc (str): The document text.
            question (str): The question text.
            answer (str): The answer text.
        Returns:
            str: The generated prompt.
        """
        PROMPT = f"""
        다음 기사 내용 안에 아래 질문에 대한 답이 들어있는지 답해줘.
        답이 들어있으면 1, 들어있지 않으면 0으로 답해줘.
        반드시 " 없이 1이나 0으로만 답변해줘.

        기사 내용: {doc}
        질문: {question}
        출력: 
        """
        return PROMPT
    
    def generate_generation_judge(
            question: str, 
            answer: str,
            prediction: str
        ) -> str:
        """
        Generate a prompt for judging whether a document contains the answer to a question.
        Args:
            question (str): The question text.
            answer (str): The answer text.
            prediction (str): The generated answer text.
        Returns:
            str: The generated prompt.
        """
        PROMPT = f"""
        다음 질문에 대한 답변이 올바른지 판단해줘.
        생성된 답변이 올바르면 1, 올바르지 않으면 0으로 답해줘.
        반드시 " 없이 1이나 0으로만 답변해줘.
        
        질문: {question}
        올바른 답변: {answer}
        생성된 답변: {prediction}
        
        출력: 
        """
        return PROMPT