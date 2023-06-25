import glob
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PDFMinerLoader

import config


# Load data
print('Loading data into VectorStore...')
index = VectorstoreIndexCreator().from_loaders(
    [PDFMinerLoader(path) for path in glob.glob('data/rp_release_notes_all/*.pdf')]
)

print('Querying data...')
# print(index.query('Who authored the Revolution Performance Release notes?'))
# print(index.query('What are Abnormal Returns and how are they handled in Revolution Performance?'))   # Not quite accurate
print(index.query("What does the 'Closed Day Method' in Revolution Performance do?"))
# print(index.query("What does the setting 'Chained Results Calendar' in Revolution Performance do?"))
# print(index.query("Where can I configure my portfolio's calculation periods in Revolution Performance?"))
# print(index.query("Explain how automated exports of chained results are configured and run in Revolution Performance?"))
# print(index.query("What are 'Chained Returns'? Explain how they are calculated in Revolution Performance."))
