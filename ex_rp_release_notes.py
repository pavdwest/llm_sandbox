import glob
import json

from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PDFMinerLoader
from langchain.llms import OpenAI

import config


# Load data
print('Loading data into VectorStore...')
index = VectorstoreIndexCreator().from_loaders(
    [PDFMinerLoader(path) for path in glob.glob('data/rp_docs/**/*.pdf')]
)

queries = [
    # 'Who authored the Revolution Performance Release notes?',
    # "Explain 'Aggregate Cash per Currency' setting in Revolution Performance." ,
    # 'Can Revolution Performance calculate returns from NAVs or GAVs?',
    # 'Can Revolution Performance calculate time-weighted and money-weighted returns from total asset values and transactions?',
    # 'Explain Price and Trading Returns in Revolution Performance.',
    # 'What returns can Revolution Performance calculate?',
    # 'Can Revolution Performance calculate Turnover Ratios?',
    # "Explain what the 'Closed Day Method' is and what it does in Revolution Performance",
    # "What are Abnormal Returns and why are they useful in Revolution Performance?",
    # "What does the setting 'Chained Results Calendar' in Revolution Performance do?",
    # "Where can I configure my portfolio's calculation periods in Revolution Performance?",
    # "Explain how automated exports of chained results are configured and run in Revolution Performance?",
    # "What are 'Chained Returns'? Explain how they are calculated in Revolution Performance.",
    # "How does Revolution Performance calculate derivatives like Futures and Options?",
    # 'How do I configure the "Workflow Notifications" file that goes to FTP? Give me all the details about that.',
    'How do I set up chained business returns and how do I export them to FTP?',
]

llm = OpenAI(
    openai_api_key=config.OPENAI_API_KEY,
    temperature=0.01,
)

print('Querying data...')
for q in queries:
    res = {
        'query': q,
        'response': index.query(q, llm)
    }
    print(json.dumps(res))
