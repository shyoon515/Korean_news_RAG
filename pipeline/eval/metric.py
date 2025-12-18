

def llm_relevance_score(prediction : str, answers : list):
    for gt in answers:
        if gt in prediction:
            return 1
    return 0

def llm_generation_score():
    pass