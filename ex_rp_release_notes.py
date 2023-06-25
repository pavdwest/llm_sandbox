import glob
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PDFMinerLoader
import json

from langchain.llms import OpenAI

import config


# Load data
print('Loading data into VectorStore...')
index = VectorstoreIndexCreator().from_loaders(
    [PDFMinerLoader(path) for path in glob.glob('data/rp_docs/**/*.pdf')]
)

# queries = [
#     'Who authored the Revolution Performance Release notes?',
#     'What are Abnormal Returns and how are they handled in Revolution Performance?',
#     "What does the 'Closed Day Method' in Revolution Performance do?",
#     "What does the setting 'Chained Results Calendar' in Revolution Performance do?",
#     "Where can I configure my portfolio's calculation periods in Revolution Performance?",
#     "Explain how automated exports of chained results are configured and run in Revolution Performance?",
#     "What are 'Chained Returns'? Explain how they are calculated in Revolution Performance.",
# ]

queries = [
    'Who authored the Revolution Performance Release notes?',
    "Explain 'Aggregate Cash per Currency' setting in Revolution Performance." ,
    'Can Revolution Performance calculate returns from NAVs or GAVs?',
    'Can Revolution Performance calculate time-weighted and money-weighted returns from total asset values and transactions?',
    'Explain Price and Trading Returns in Revolution Performance.',
    'What returns can Revolution Performance calculate?',
    'Can Revolution Performance calculate Turnover Ratios?',
    "Explain what the 'Closed Day Method' is and what it does in Revolution Performance",
    "What are Abnormal Returns and why are they useful in Revolution Performance?",
    "What does the setting 'Chained Results Calendar' in Revolution Performance do?",
    "Where can I configure my portfolio's calculation periods in Revolution Performance?",
    "Explain how automated exports of chained results are configured and run in Revolution Performance?",
    "What are 'Chained Returns'? Explain how they are calculated in Revolution Performance.",
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


# print(index.query('Who authored the Revolution Performance Release notes?'))
# print(index.query('What are Abnormal Returns and how are they handled in Revolution Performance?'))   # Not quite accurate
# print(index.query("What does the 'Closed Day Method' in Revolution Performance do?"))
# print(index.query("What does the setting 'Chained Results Calendar' in Revolution Performance do?"))
# print(index.query("Where can I configure my portfolio's calculation periods in Revolution Performance?"))
# print(index.query("Explain how automated exports of chained results are configured and run in Revolution Performance?"))
# print(index.query("What are 'Chained Returns'? Explain how they are calculated in Revolution Performance."))
