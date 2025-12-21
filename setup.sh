mkdir final_project
cd final_project
git init
git remote add origin https://github.com/shyoon515/Korean_news_RAG.git
git fetch origin
git checkout -b main origin/main

conda create -n final_project python=3.10.18 -y
conda activate final_project

pip install datasets
pip install sentence-transformers
pip install qdrant-client
pip install langchain-text-splitters
pip install nvitop
pip install openai
pip install vllm
pip install rank-bm25
pip install kiwipiepy
pip install gradio