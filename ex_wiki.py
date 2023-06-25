from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import WikipediaLoader

import config


# Load data
print('Loading data into VectorStore...')
index = VectorstoreIndexCreator().from_loaders(
    [
        WikipediaLoader('Stable Diffusion'),
        WikipediaLoader('Midjourney'),
    ]
)

print('Querying data...')
print(index.query('who authored the theoretical paper behind stable diffusion?'))
