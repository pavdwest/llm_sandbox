from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import WikipediaLoader

import config


# Load data
print('Loading data into VectorStore...')
index = VectorstoreIndexCreator().from_loaders(
    [
        WikipediaLoader('Stable Diffusion'),
        WikipediaLoader('Midjourney'),
        WikipediaLoader('2023 Titan submersible incident'),
    ]
)

print('Querying data...')
# print(index.query('who authored the theoretical paper behind stable diffusion?'))
print(index.query('How big was the Titan vessel trying to explore the wreck of the Titanic?'))