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
