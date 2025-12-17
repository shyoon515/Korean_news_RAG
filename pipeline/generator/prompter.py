from typing import List, Union

"""
Formatting functions for prompts.
"""
class PromptGenerator:

    @staticmethod
    def format_docs(docs : Union[List[dict], List[str], dict]) -> str:
        """
        Format documents for display.
        Args:
            docs (list of string): List of documents to format. automatically gives ids
            docs (list of dict): List of dictionaries containing documents with 'id' and 'text' keys.
            docs (dict): Dictionary of documents with document IDs as keys and document texts as values.
        Returns:
            str: Formatted string of documents.
        """
        formatted_docs = []
        if isinstance(docs, dict):
            for docid in docs:
                formatted_doc = f"Document ID: {docid}\nContent: {docs[docid]}\n"
                formatted_docs.append(formatted_doc)
        else:
            if isinstance(docs[0], str):
                for i in range(len(docs)):
                    formatted_doc = f"Document {i+1}:\n{docs[i]}\n"
                    formatted_docs.append(formatted_doc)
            elif isinstance(docs[0], dict):
                for doc in docs:
                    formatted_doc = f"Document ID: {doc['id']}\nContent: {doc['text']}\n"
                    formatted_docs.append(formatted_doc)
            else:
                raise Exception("docs must be a list of strings or a list of dictionaries")
        
        return "\n".join(formatted_docs)

    @staticmethod
    def with_formatted_docs(
            query: Union[str, List[str]],
            formatted_docs: Union[str, List[str]]
        ) -> List[str]:

        prompts = []
        assert type(query) == type(formatted_docs), "query and formatted_docs must be of the same type(str or list)"
        if isinstance(query, list):
            assert len(query) == len(formatted_docs), "query and formatted_docs must be of the same length"
        
        if isinstance(query, str):
            PROMPT = f"""
            Based on the given reference documents, answer the following question.
            When answering, do not repeat the question, and only provide the correct answer.
            Provide the answer only in JSON format as {{"Answer":"Your answer"}}.
            Reference Documents:
            ---------------------
            {formatted_docs}
            ——————————
            Question: {query}
            Answer: 
            """
            prompts.append(PROMPT)

        elif isinstance(query, list):
            for i, q in enumerate(query):
                if isinstance(formatted_docs[i], list):
                    raise Exception("formatted_docs must be a list of strings, not a list of lists")
                    
                PROMPT = f"""
                Based on the given reference documents, answer the following question.
                When answering, do not repeat the question, and only provide the correct answer.
                Provide the answer only in JSON format as {{"Answer":"Your answer"}}.
                Reference Documents:
                ---------------------
                {formatted_docs[i]}
                ——————————
                Question: {q}
                Answer: 
                """
                prompts.append(PROMPT)
        return prompts
    
    @staticmethod
    def without_formatted_docs(
            query: Union[str, List[str]]
        ) -> List[str]:

        prompts = []
        if isinstance(query, str):
            PROMPT = f"""
            Answer the following question.
            When answering, do not repeat the question, and only provide the correct answer.
            Provide the answer only in JSON format as {{"Answer":"Your answer"}}.
            Question: {query}
            Answer: 
            """
            prompts.append(PROMPT)

        elif isinstance(query, list):
            for q in query:
                PROMPT = f"""
                Answer the following question.
                When answering, do not repeat the question, and only provide the correct answer.
                Provide the answer only in JSON format as {{"Answer":"Your answer"}}.
                Question: {q}
                Answer: 
                """
                prompts.append(PROMPT)
        return prompts
    
    @staticmethod
    def path_planning(
            query: Union[str, List[str]],
        ) -> List[str]:
        """
        Generate a path planning prompt based on the query and options.
        Args:
            query (str or list of str): The question to be answered.
            options (list of str): The options to choose from.
        Returns:
            list of str: Formatted prompts for path planning.
        """
        prompts = []
        if isinstance(query, str):
            PROMPT = f"""
            ### Instruction:
            You are an expert in query planning for a retrieval-augmented system.

            Given a user's question, choose:
            **Step 1 - Retrieval policy (choose one)**  
            - <DENSE>: dense vector retriever (e.g., embedding/Faiss)  
            - <SPARSE>: sparse keyword retriever (e.g., BM25)

            **Step 2 - Query reformulation strategy (choose one)**  
            - <QR_ORIGINAL>: Use the original query without changes.
            - <QR_KEYWORD>: Focus on the most essential keywords.
            - <QR_SYNONYM>: Rewrite using simpler or more common synonyms.
            - <QR_SEMANTIC>: Add useful context to enrich the meaning.
            - <QR_EXPLAIN>: Rephrase with a short explanation of the user's intent.

            When selecting a reform strategy, **use your own internal knowledge** to infer what clarification or context the question might need.

            Also, **choose the two tokens jointly** — the selected retrieval method and the reform strategy should complement each other for the best final generation.

            Your response should be **exactly two tokens** without any reasoning or explanation, formatted as:
            <RETRIEVAL_POLICY><QUERY_REFORM_STRATEGY>

            Now, analyze the following user query and respond with your choice:

            [QUESTION]: "{query}"

            ### Response:
            """
            prompts.append(PROMPT)

        elif isinstance(query, list):
            for q in query:
                PROMPT = f"""
                ### Instruction:
                You are an expert in query planning for a retrieval-augmented system.

                Given a user's question, choose:
                **Step 1 - Retrieval policy (choose one)**  
                - <DENSE>: dense vector retriever (e.g., embedding/Faiss)  
                - <SPARSE>: sparse keyword retriever (e.g., BM25)

                **Step 2 - Query reformulation strategy (choose one)**  
                - <QR_ORIGINAL>: Use the original query without changes.
                - <QR_KEYWORD>: Focus on the most essential keywords.
                - <QR_SYNONYM>: Rewrite using simpler or more common synonyms.
                - <QR_SEMANTIC>: Add useful context to enrich the meaning.
                - <QR_EXPLAIN>: Rephrase with a short explanation of the user's intent.

                When selecting a reform strategy, **use your own internal knowledge** to infer what clarification or context the question might need.

                Also, **choose the two tokens jointly** — the selected retrieval method and the reform strategy should complement each other for the best final generation.

                Your response should be **exactly two tokens** without any reasoning or explanation, formatted as:
                <RETRIEVAL_POLICY><QUERY_REFORM_STRATEGY>

                Now, analyze the following user query and respond with your choice:

                [QUESTION]: "{q}"

                ### Response:
                """
                prompts.append(PROMPT)
        
        return prompts

    @staticmethod
    def path_planning_free_qr(
            query: Union[str, List[str]],
        ) -> List[str]:
        """
        Generate a path planning prompt based on the query and options.
        Args:
            query (str or list of str): The question to be answered.
            options (list of str): The options to choose from.
        Returns:
            list of str: Formatted prompts for path planning.
        """
        prompts = []
        if isinstance(query, str):
            PROMPT = f"""
            ### Instruction:
            You are an expert in query planning for a retrieval-augmented system.

            Given a user's question, perform **both** of the following steps:

            **Step 1 - Retrieval policy (choose one)**
            - <DENSE>   : dense vector retriever (e.g., embeddings / Faiss)
            - <SPARSE>  : sparse keyword retriever (e.g., BM25)

            **Step 2 - Query reformulation**
            Rewrite the question in the style you believe will retrieve the most relevant documents.
            Place the rewritten text **between** the tags **<QR> … </QR>**.

            When rewriting, you may:
            - keep the wording identical (original),
            - extract essential keywords,
            - replace terms with simpler synonyms,
            - add clarifying context,
            - briefly explain user intent,
            whichever you judge most effective.

            **Choose the retrieval token and write the reformulated query *jointly*.**  
            Your answer must contain **exactly one retrieval token followed by one <QR> block**, and nothing else.

            **Allowed output format** (examples):
            <DENSE> <QR>capital city of France is Paris</QR>
            <SPARSE> <QR>Zamalek Centennial opponent league</QR>
            
            Now analyze the user query and respond:

            [QUESTION]: "{query}"

            ### Response:
            """
            prompts.append(PROMPT)

        elif isinstance(query, list):
            for q in query:
                PROMPT = f"""
                ### Instruction:
                You are an expert in query planning for a retrieval-augmented system.

                Given a user's question, perform **both** of the following steps:

                **Step 1 - Retrieval policy (choose one)**
                - <DENSE>   : dense vector retriever (e.g., embeddings / Faiss)
                - <SPARSE>  : sparse keyword retriever (e.g., BM25)

                **Step 2 - Query reformulation**
                Rewrite the question in the style you believe will retrieve the most relevant documents.
                Place the rewritten text **between** the tags **<QR> … </QR>**.

                When rewriting, you may:
                - keep the wording identical (original),
                - extract essential keywords,
                - replace terms with simpler synonyms,
                - add clarifying context,
                - briefly explain user intent,
                whichever you judge most effective.

                **Choose the retrieval token and write the reformulated query *jointly*.**  
                Your answer must contain **exactly one retrieval token followed by one <QR> block**, and nothing else.

                **Allowed output format** (examples):
                <DENSE> <QR>capital city of France is Paris</QR>
                <SPARSE> <QR>Zamalek Centennial opponent league</QR>
                
                Now analyze the user query and respond:

                [QUESTION]: "{q}"

                ### Response:
                """
                prompts.append(PROMPT)
        
        return prompts
    
    @staticmethod
    def retriever_reasoning(
            query: Union[str, List[str]],
            ret: str
        ) -> List[str]:
        if ret not in ["dense", "sparse"]:
            raise ValueError("ret must be either 'dense' or 'sparse'")
        prompts = []
        if isinstance(query, str):
            PROMPT = f"""
            ### Instruction:
            Given the following query, write a short reasoning passage (1-2 sentences) as if you are **considering both dense and sparse retrieval**, but **eventually conclude that {ret} retrieval is more suitable**.

            You should reflect briefly on why {ret} option may work better, and naturally arrive at the conclusion that {ret} is the better choice.

            Focus your reasoning on the **linguistic and structural characteristics** of the query itself, rather than generic properties of dense or sparse retrieval.

            [QUERY]: {query}

            ### Response:
            """
            prompts.append(PROMPT)

        elif isinstance(query, list):
            for q in query:
                PROMPT = PromptGenerator.retriever_reasoning(q, ret)
                prompts.append(PROMPT)
        return prompts
    
    @staticmethod
    def query_reformulation(
        query: Union[str, List[str]],
        strategy: str
    ):
        if strategy not in ["keyword", "synonym", "semantic", "explain"]:
            raise ValueError("strategy must be one of: 'keyword', 'synonym', 'semantic', 'explain'")
        prompts = []

        if isinstance(query, str):
            PROMPT_KEYWORD = f"""
                You are an AI assistant enhancing generation performance by focusing on the most essential elements of the user's question.
                Extract and restate the query using only the most informative keywords that best describe what the user wants to know.

                Provide the reformed query only without any additional context or explanation.

                Question: {query}
                Answer:
            """
            PROMPT_SYNONYM = f"""
                You are an AI assistant helping improve answer generation by restating user questions using clearer or more commonly used synonyms.
                Rephrase the query using alternative expressions that might help the language model better understand and generate a more accurate answer.

                Provide the reformed query only without any additional context or explanation.

                Question: {query}
                Answer:
            """
            PROMPT_SEMANTIC = f"""
                You are an AI assistant improving language model generation quality by enriching the query with additional context.
                Reformulate the question by inserting relevant background or clarifying terms, so that the language model can generate a more informed and precise response.

                Provide the reformed query only without any additional context or explanation.

                Question: {query}
                Answer:
            """
            PROMPT_EXPLAIN = f"""
                You are an AI assistant enhancing answer generation by rephrasing the user's question with a brief explanation of its purpose or intent.
                Reformulate the question so that it includes a short explanatory phrase to help the language model generate a more relevant and accurate answer.

                Provide the reformed query only without any additional context or explanation.

                Question: {query}
                Answer:
            """
            if strategy == "keyword":
                PROMPT = PROMPT_KEYWORD
            elif strategy == "synonym":
                PROMPT = PROMPT_SYNONYM
            elif strategy == "semantic":
                PROMPT = PROMPT_SEMANTIC
            elif strategy == "explain":
                PROMPT = PROMPT_EXPLAIN
            prompts.append(PROMPT)
        elif isinstance(query, list):
            for q in query:
                PROMPT = PromptGenerator.query_reformulation(q, strategy)
                prompts.append(PROMPT)
        return prompts
    
    @staticmethod
    def reform_reasoning(
            query: Union[str, List[str]],
            reformed_query: Union[str, List[str]],
            ret: str
        ) -> List[str]:
        assert type(query) == type(reformed_query), "query and reformed_query must be of the same type(str or list)"
        if isinstance(query, str):
            PROMPT = f"""
            ### Instruction:
            ### Instruction:
            You are simulating the internal reasoning process of a query planner that has been instructed to use a specific retriever and reformulate a given query accordingly.

            Below, you are given:
            - a retrieval policy,
            - an original user query,
            - and the final reformulated query that was produced.

            Your task is to **reconstruct the internal reasoning process** that would naturally lead to this reformulation, **before** the reformulation was written.  
            This should reflect an active planning mindset: given the retriever and query, **what did you observe, and what did you decide to change or preserve?**

            Write your answer in the form of a brief strategic thought process (1-2 sentences).  
            Avoid describing what was done; instead, focus on **why it *should* be done**, leading to the final result.

            [RETRIEVER]: {ret}
            [ORIGINAL QUERY]: {query}
            [REFORMULATED QUERY]: {reformed_query}

            ### Response:
            """
            return [PROMPT]
        elif isinstance(query, list):
            assert len(query) == len(reformed_query), "query and reformed_query must be of the same length"
            prompts = []
            for i in range(len(query)):
                PROMPT = PromptGenerator.reform_reasoning(query[i], reformed_query[i], ret)
                prompts.append(PROMPT)
            return prompts
    
    @staticmethod
    def path_planning_full(
            query: Union[str, List[str]],
        ) -> List[str]:
        """
        Generate a path planning prompt based on the query and options.
        Args:
            query (str or list of str): The question to be answered.
            options (list of str): The options to choose from.
        Returns:
            list of str: Formatted prompts for path planning.
        """
        prompts = []
        if isinstance(query, str):
            PROMPT = f"""
            ### Instruction:
            You are an expert query planner in a retrieval-augmented generation (RAG) system.

            Given a user query, perform the following steps **in a structured format**:

            1. Analyze the query and decide whether **dense** or **sparse** retrieval is more suitable.  
            Write your reasoning inside a `<think_retrieval>...</think_retrieval>` tag.

            2. Based on the selected retriever, describe your **query reformulation strategy** —  
            i.e., what you should do to make the query better aligned with the retriever's strengths.  
            Write this reasoning inside a `<think_qr>...</think_qr>` tag.

            3. Finally, output the reformulated query using the `<QR>...</QR>` tag.  
            You may either simplify the query to emphasize keywords, or rephrase it to add semantic clarity — whichever suits the chosen retriever.

            4. Include exactly one retrieval policy token — either `<DENSE>` or `<SPARSE>` — **between** the two reasoning blocks.

            Your final output must follow this **exact format**:
            <think_retrieval>...</think_retrieval>
            <DENSE> or <SPARSE>
            <think_qr>...</think_qr>
            <QR>...</QR>
            
            Use natural, well-structured English for both reasoning blocks.

            ---

            ### Now process the following user query:

            [QUERY]: {query}

            ### Response:
            """
            prompts.append(PROMPT)
        elif isinstance(query, list):
            prompts = []
            for q in query:
                PROMPT = PromptGenerator.path_planning_full(q)
                prompts.append(PROMPT)
        
        return prompts



