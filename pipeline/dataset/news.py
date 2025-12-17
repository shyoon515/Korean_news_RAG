from datasets import load_dataset, Dataset

def get_news_dataset() -> Dataset:
    """
    Loads news dataset from datasets library given a file path.
    """
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

    data_dict = dataset['train'].to_dict()
    for bucket in ['validation', 'test']:
        for key in data_dict:
            data_dict[key].extend(dataset[bucket][key])
    data_dict['bucket'] = ['train'] * len(dataset['train']) + ['validation'] * len(dataset['validation']) + ['test'] * len(dataset['test'])
    
    data_dict['text'] = [data_dict['title'][i] + "\n" + data_dict['document'][i] + "\n" + data_dict['summary'][i] for i in range(len(data_dict['title']))]
    return Dataset.from_dict(data_dict)